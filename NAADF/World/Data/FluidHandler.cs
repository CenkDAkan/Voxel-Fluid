using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Render;
using System;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * This class is meant to help simulate fluid behavior in a voxel-based world. 
    * The simulation state is kept separate from the rendering logic, allowing for future improvements to the display without affecting the simulation itself.
    * It uses engine's already existing editing pipeline to display the fluid voxel in the world.
    * Later on ApplyToWorld() can be swapped out for a more efficient method when the fluid simulation grows to many cells.
    */
    public class FluidHandler
    {
        private WorldData worldData;

        // Separate empty/fluid grid, decoupled from the engine's solid-voxel node tree. This is the
        // simulation's source of truth; ApplyToWorld only ever reads it to decide what to draw.
        private FluidGrid fluidGrid;

        // The render-table index of the material our fluid voxel uses. Registered once at construction.
        private uint fluidTypeRenderIndex;

        // Cells whose fluid state changed since the last ApplyToWorld call, drained (and re-drawn) each apply.
        private List<Point3> dirtyCells = new List<Point3>();

        // Simulation state
        private bool hasCell = false;
        private Point3 currentCell;          // the voxel currentPosition is currently floored/snapped to; what fluidGrid marks as fluid
        private Vector3 currentPosition;     // voxel position, integrated every physics step

        // Gravity is applied as the external-forces term of the Navier-Stokes equation (u(t+dt) = u(t) + F*dt). 
        // The engine has no defined real-world unit scale (a voxel isn't "1 meter" anywhere), 
        // so this magnitude is arbitrary and be changed at any point for fine tuning
        private Vector3 gravity = new Vector3(0f, -20f, 0f);

        // Physics runs on a fixed tick rather than the raw (variable) frame delta, so integration stays
        // stable regardless of framerate. The accumulator banks elapsed time between ticks, same as the demo
        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;

        public FluidHandler(WorldData worldData)
        {
            this.worldData = worldData;
            fluidGrid = new FluidGrid(worldData.sizeInVoxels);

            // A new voxel type is created with a unique ID, color, and material properties. Emissive so that it is properly visible regardless of the scene
            // The render index of this new voxel type is stored for later use when writing to the world.
            VoxelType fluidType = new VoxelType
            {
                ID = "fluid_demo",
                colorBase = new Vector3(0.2f, 0.5f, 1.0f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            // store the render-table index for later use, taken from the return value of ApplyVoxelType in voxelTypeHandler
            fluidTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(fluidType).renderIndex; 
        }

        public void Update(float gameTime)
        {
            HandleSpawnInput();

            if (!hasCell)
                return;

            // Advance the simulation on a fixed cadence.
            physicsAccumulatorMs += gameTime;
            bool moved = false;
            while (physicsAccumulatorMs >= physicsIntervalMs)
            {
                physicsAccumulatorMs -= physicsIntervalMs;
                StepPhysics(physicsIntervalMs / 1000f);
                moved = true;
                if (physicsIntervalMs <= 0f) break; // avoid an infinite loop if physicsIntervalMs is set to 0
            }

            // ApplyToWorld is only called when the voxel actually moves rather than every frame
            if (moved)
                ApplyToWorld();
        }

        // Spawn or respawn the voxel a few cells in front of the camera when G is pressed.
        private void HandleSpawnInput()
        {
            if (!IO.KBStates.IsKeyToggleDown(Keys.G))
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Point3 spawn = Point3.FromVector3(camPos + camDir * 20f);

            if (!fluidGrid.IsInside(spawn))
            {
                Console.WriteLine("FluidHandler: spawn point is outside the world, aim somewhere else and press G again.");
                return;
            }

            // If a voxel already exists, clear it from the grid before moving the origin so we don't leave a stray cell behind.
            // If commented out the previous voxel will remain still in the world and a new one will be spawned at the new location.
            if (hasCell)
            {
                fluidGrid.SetFluid(currentCell, false);
                dirtyCells.Add(currentCell);
            }

            currentCell = spawn;
            currentPosition = spawn.ToVector3();
            physicsAccumulatorMs = 0f;
            hasCell = true;
            fluidGrid.SetFluid(currentCell, true);
            dirtyCells.Add(currentCell);

            ApplyToWorld(); // make the freshly spawned voxel appear immediately
            Console.WriteLine($"FluidHandler: spawned voxel at ({spawn.X}, {spawn.Y}, {spawn.Z}).");
        }

        // Integrates gravity into the cell's velocity, then integrates velocity into its continuous position. 
        // Once the continuous position crosses into a neighboring voxel, the grid entry moves there too. 
        // fluidGrid stays the source of truth for both fluid state and velocity. No collision checks yet
        private void StepPhysics(float dt)
        {
            Vector3 velocity = fluidGrid.GetVelocity(currentCell);
            velocity += gravity * dt;

            Vector3 newPosition = currentPosition + velocity * dt;
            Point3 newCell = Point3.FromVector3(newPosition);

            if (!fluidGrid.IsInside(newCell))
            {
                newPosition = currentCell.ToVector3();
                newCell = currentCell;
                velocity = Vector3.Zero;
            }

            if (!newCell.Equals(currentCell))
            {
                fluidGrid.SetFluid(currentCell, false);
                dirtyCells.Add(currentCell);
                currentCell = newCell;
                fluidGrid.SetFluid(currentCell, true);
                dirtyCells.Add(currentCell);
            }

            currentPosition = newPosition;
            fluidGrid.SetVelocity(currentCell, velocity);
        }

        // Display/apply step: re-draws every cell whose fluid state changed since the last apply, then
        // pushes to the GPU. This is the part that is intentionally swappable for a more efficient
        // implementation later; it only ever reads fluidGrid, never decides fluid state itself.
        private void ApplyToWorld()
        {
            foreach (Point3 cell in dirtyCells)
                WriteCell(cell);
            dirtyCells.Clear();

            // Stage the edited chunks through the same way the editing tools use. ChangeHandler.Update() then uploads everything to the GPU.
            worldData.editingHandler.processChunks(false);
        }

        // Translate a world-voxel position into the engine's chunk + in-chunk coordinates and write one voxel,
        // reading fluidGrid to decide whether it should be drawn as fluid or erased back to empty.
        private void WriteCell(Point3 worldVoxel)
        {
            if (!fluidGrid.IsInside(worldVoxel))
                return;

            Point3 chunkPos = worldVoxel / 16;          // which chunk contains this voxel, in chunk-grid coordinate
            Point3 voxelPosInChunk = worldVoxel % 16;   // 0..15 on each axis, the voxel's position within the chunk

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            // Voxel encoding: bit 15 = "solid" flag, low 15 bits = material render index. 0 means empty/air.
            uint type = fluidGrid.IsFluid(worldVoxel) ? (1u << 15) | fluidTypeRenderIndex : 0u;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);
        }
    }
}
