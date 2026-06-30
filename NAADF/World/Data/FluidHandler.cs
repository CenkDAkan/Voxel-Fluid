using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Render;
using System;

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

        // The render-table index of the material our fluid voxel uses. Registered once at construction.
        private uint fluidTypeRenderIndex;

        // Simulation state
        private bool hasCell = false;
        private Point3 originCell;
        private Point3 currentCell;
        private int travel = 0;              // how far along the path we are, in voxels, from originCell
        private int stepDir = 1;             // +1 / -1, flips when we reach either end of the path
        private const int RangeSteps = 40;   // how many voxels the demo voxel travels before bouncing back

        // Movement of the fluid voxel is determined by the interval variable. 
        // The accumulator variable banks the elapsed time since the last movement, allowing for smooth and consistent updates regardless of frame rate.
        private float moveIntervalMs = 60f;
        private float moveAccumulatorMs = 0f;

        public FluidHandler(WorldData worldData)
        {
            this.worldData = worldData;

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
            moveAccumulatorMs += gameTime;
            bool moved = false;
            while (moveAccumulatorMs >= moveIntervalMs)
            {
                moveAccumulatorMs -= moveIntervalMs;
                StepSimulation();
                moved = true;
                if (moveIntervalMs <= 0f) break; // avoid an infinite loop if moveIntervalMs is set to 0
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

            // The check is done on the spawn point and the farthest point along the path, which is RangeSteps away in the +X direction.
            if (!IsInsideWorld(spawn) || !IsInsideWorld(spawn + new Point3(RangeSteps, 0, 0)))
            {
                Console.WriteLine("FluidHandler: spawn point is outside the world, aim somewhere else and press G again.");
                return;
            }

            // If a voxel already exists, erase it before moving the origin so we don't leave a stray cell behind.
            // If commented out the previous voxel will remain still in the world and a new one will be spawned at the new location.
            if (hasCell)
                WriteCell(currentCell, isErase: true);

            originCell = spawn;
            currentCell = spawn;
            travel = 0;
            stepDir = 1;
            moveAccumulatorMs = 0f;
            hasCell = true;

            ApplyToWorld(); // make the freshly spawned voxel appear immediately
            Console.WriteLine($"FluidHandler: spawned voxel at ({spawn.X}, {spawn.Y}, {spawn.Z}).");
        }

        // To decide where the voxel goes next. No engine interaction here on purpose. 
        // Will do more complex fluid simulation math later, but for now it just moves back and forth along the X axis.
        private void StepSimulation()
        {
            travel += stepDir;
            if (travel >= RangeSteps || travel <= 0)
                stepDir = -stepDir; // bounce at either end so the voxel stays near the spawn point

            currentCell = originCell + new Point3(travel, 0, 0);
        }

        // Display/apply step: erase the previous cell, draw the current cell, then push to the GPU.
        // This is the part that is intentionally swappable for a more efficient implementation later.
        private void ApplyToWorld()
        {
            // The voxel moved at most one cell, so the previous position is one step back along the path.
            Point3 previousCell = originCell + new Point3(travel - stepDir, 0, 0);

            if (!previousCell.Equals(currentCell))
                WriteCell(previousCell, isErase: true);
            WriteCell(currentCell, isErase: false);

            // Stage the edited chunks through the same way the editing tools use. ChangeHandler.Update() then uploads everything to the GPU.
            worldData.editingHandler.processChunks(false);
        }

        // Translate a world-voxel position into the engine's chunk + in-chunk coordinates and write one voxel.
        private void WriteCell(Point3 worldVoxel, bool isErase)
        {
            if (!IsInsideWorld(worldVoxel))
                return;

            Point3 chunkPos = worldVoxel / 16;          // which chunk contains this voxel, in chunk-grid coordinate
            Point3 voxelPosInChunk = worldVoxel % 16;   // 0..15 on each axis, the voxel's position within the chunk

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            // Voxel encoding: bit 15 = "solid" flag, low 15 bits = material render index. 0 means empty/air.
            uint type = isErase ? 0u : (1u << 15) | fluidTypeRenderIndex;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);
        }

        private bool IsInsideWorld(Point3 p)
        {
            return p.X >= 0 && p.Y >= 0 && p.Z >= 0
                && p.X < worldData.sizeInVoxels.X
                && p.Y < worldData.sizeInVoxels.Y
                && p.Z < worldData.sizeInVoxels.Z;
        }
    }
}
