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

        // One simulated voxel of fluid. Velocity itself lives on fluidGrid (keyed by currentCell), this just tracks which 
        // cell/continuous position belongs to which particle so multiple can be integrated independently per frame.
        private class Particle
        {
            public Point3 currentCell;       // the voxel currentPosition is currently floored/snapped to; what fluidGrid marks as fluid
            public Vector3 currentPosition;  // voxel position, integrated every physics step
        }

        private List<Particle> particles = new List<Particle>();

        // Gravity is applied as the external-forces term of the Navier-Stokes equation (u(t+dt) = u(t) + F*dt).
        // The engine has no defined real-world unit scale (a voxel isn't "1 meter" anywhere),
        // so this magnitude is arbitrary and be changed at any point for fine tuning
        private Vector3 gravity = new Vector3(0f, -20f, 0f);

        // Physics runs on a fixed tick rather than the raw (variable) frame delta, so integration stays
        // stable regardless of framerate. The accumulator banks elapsed time between ticks, same as the demo
        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;

        // Hose emitter: while held, spawns particles from the camera position at a fixed cadence
        private float hoseSpeed = 40f;
        private float hoseIntervalMs = 50f;
        private float hoseAccumulatorMs = 0f;

        // Sprinkler: a placed emitter. Its base is a static marker voxel, drawn directly and never touched by the fluid grid/dirty-cell system 
        // so the fluid stream can't erase it. Its launch direction rotates left and right 
        // around the facing direction it was placed with and is tilted upward, so gravity arcs each launch into the ground
        private class Sprinkler
        {
            public Point3 position;       // base cell, holds the static marker voxel
            public Vector3 forwardXZ;     // horizontal reference direction, fixed at placement time
            public float angleDeg;        // current offset from forwardXZ, rotates within +-(sprinklerArcDegrees/2)
            public int sweepDir;          // +1 or -1, flips when angleDeg hits an arc limit
            public float emitAccumulatorMs;
        }

        private List<Sprinkler> sprinklers = new List<Sprinkler>();
        private uint sprinklerTypeRenderIndex;
        private float sprinklerArcDegrees = 120f;
        private float sprinklerAngularSpeedDegPerSec = 90f;
        private float sprinklerTiltDegrees = 35f;

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

            // Distinct color from the fluid itself so a placed sprinkler's base is visually identifiable.
            VoxelType sprinklerType = new VoxelType
            {
                ID = "fluid_sprinkler_head",
                colorBase = new Vector3(1.0f, 0.5f, 0.1f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            sprinklerTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(sprinklerType).renderIndex;
        }

        public void Update(float gameTime)
        {
            HandleSpawnInput();
            HandleHoseInput(gameTime);
            HandlePlaceSprinklerInput();
            HandleSprinklers(gameTime);

            if (particles.Count == 0)
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

            // ApplyToWorld is only called when a voxel actually moved rather than every frame
            if (moved)
                ApplyToWorld();
        }

        // Drop one voxel a few cells in front of the camera when G is pressed, with no initial velocity
        private void HandleSpawnInput()
        {
            if (!IO.KBStates.IsKeyToggleDown(Keys.G))
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Point3 spawn = Point3.FromVector3(camPos + camDir * 20f);

            SpawnParticle(spawn, Vector3.Zero);
        }

        // While H is held, spawns a stream of voxels from the camera position at a fixed cadence
        private void HandleHoseInput(float gameTime)
        {
            if (!IO.KBStates.New.IsKeyDown(Keys.H))
            {
                hoseAccumulatorMs = 0f; // reset so the stream starts immediately next time H is held
                return;
            }

            hoseAccumulatorMs += gameTime;
            while (hoseAccumulatorMs >= hoseIntervalMs)
            {
                hoseAccumulatorMs -= hoseIntervalMs;

                Vector3 camPos = WorldRender.camera.GetPos().toVector3();
                Vector3 camDir = WorldRender.camera.GetDir();
                Point3 spawn = Point3.FromVector3(camPos + camDir * 4f);

                SpawnParticle(spawn, camDir * hoseSpeed);
            }
        }

        // Places a new sprinkler when J is pressed, the marker voxel is written once and is not part of the fluid grid
        private void HandlePlaceSprinklerInput()
        {
            if (!IO.KBStates.IsKeyToggleDown(Keys.J))
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Point3 spawn = Point3.FromVector3(camPos + camDir * 20f);

            if (!fluidGrid.IsInside(spawn))
            {
                Console.WriteLine("FluidHandler: sprinkler point is outside the world, aim somewhere else.");
                return;
            }

            Vector3 forwardXZ = new Vector3(camDir.X, 0f, camDir.Z);
            forwardXZ = forwardXZ.LengthSquared() > 1e-6f ? Vector3.Normalize(forwardXZ) : Vector3.UnitZ;

            sprinklers.Add(new Sprinkler
            {
                position = spawn,
                forwardXZ = forwardXZ,
                angleDeg = 0f,
                sweepDir = 1,
                emitAccumulatorMs = 0f,
            });

            PlaceMarkerVoxel(spawn, sprinklerTypeRenderIndex);
            Console.WriteLine($"FluidHandler: placed sprinkler at ({spawn.X}, {spawn.Y}, {spawn.Z}).");
        }

        // Advances each sprinkler's rotation and, on its own cadence, launches a particle in its current direction
        private void HandleSprinklers(float gameTime)
        {
            foreach (Sprinkler sprinkler in sprinklers)
            {
                float halfArc = sprinklerArcDegrees * 0.5f;
                sprinkler.angleDeg += sprinklerAngularSpeedDegPerSec * (gameTime / 1000f) * sprinkler.sweepDir;
                if (sprinkler.angleDeg > halfArc)
                {
                    sprinkler.angleDeg = halfArc;
                    sprinkler.sweepDir = -1;
                }
                else if (sprinkler.angleDeg < -halfArc)
                {
                    sprinkler.angleDeg = -halfArc;
                    sprinkler.sweepDir = 1;
                }

                sprinkler.emitAccumulatorMs += gameTime;
                while (sprinkler.emitAccumulatorMs >= hoseIntervalMs)
                {
                    sprinkler.emitAccumulatorMs -= hoseIntervalMs;

                    Vector3 horizontal = Vector3.Transform(sprinkler.forwardXZ, Matrix.CreateRotationY(MathHelper.ToRadians(sprinkler.angleDeg)));
                    float tiltRad = MathHelper.ToRadians(sprinklerTiltDegrees);
                    Vector3 launchDir = horizontal * (float)Math.Cos(tiltRad) + Vector3.Up * (float)Math.Sin(tiltRad);

                    // Emit from one cell above the marker so the stream's own dirty-cell writes never touch (and erase) the marker voxel.
                    Point3 emitCell = sprinkler.position + new Point3(0, 1, 0);
                    SpawnParticle(emitCell, launchDir * hoseSpeed);
                }
            }
        }

        // Writes a single static voxel directly, bypassing fluidGrid entirely, used for the sprinkler's
        // base marker, which must never be picked up/erased by the fluid simulation's dirty-cell system.
        private void PlaceMarkerVoxel(Point3 worldVoxel, uint typeRenderIndex)
        {
            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            uint type = (1u << 15) | typeRenderIndex;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);

            worldData.editingHandler.processChunks(false);
        }

        // Adds one new fluid particle at cell with the given initial velocity 
        // (the "impulse" is zero for a plain dropped voxel, camDir * hoseSpeed for the hose)
        // Skips cells already marked fluid rather than merging into them
        private void SpawnParticle(Point3 cell, Vector3 initialVelocity)
        {
            if (!fluidGrid.IsInside(cell))
            {
                Console.WriteLine("FluidHandler: spawn point is outside the world, aim somewhere else.");
                return;
            }

            if (fluidGrid.IsFluid(cell) || worldData.IsVoxelSolid(cell))
                return;

            particles.Add(new Particle { currentCell = cell, currentPosition = cell.ToVector3() });

            fluidGrid.SetFluid(cell, true);
            fluidGrid.SetVelocity(cell, initialVelocity);
            dirtyCells.Add(cell);

            ApplyToWorld(); // make the freshly spawned voxel appear immediately
        }

        // Integrates gravity into each particle's velocity, then integrates velocity into its continuous position
        // With collision resolution, see StepParticle. Once a particle's continuous position crosses into a neighboring voxel, 
        // its grid entry moves there too. fluidGrid stays the source of truth for both fluid state and velocity. 
        // Fluid-fluid collision isn't handled yet, if two particles' paths land in the same cell, the later write just wins
        private void StepPhysics(float dt)
        {
            foreach (Particle particle in particles)
                StepParticle(particle, dt);
        }

        private void StepParticle(Particle particle, float dt) // dt stands for delta time in seconds, not milliseconds
        {
            Vector3 velocity = fluidGrid.GetVelocity(particle.currentCell);
            velocity += gravity * dt;

            // Resolve collision one axis at a time against a running position, rather than testing the combined
            // destination in one shot. This lets a particle blocked on one axis keep moving on the others,
            // like sliding along a wall/floor, and it prevents a diagonal move from cutting through a solid corner
            // that neither pure-axis move would itself enter. Each axis step tests against the previous axes' 
            // already-resolved position, so the final cell always ends up checked by the time all three are done. 
            // World bounds are folded into the same per-axis check, so sliding along the world edge behaves
            // the same as sliding along solid geometry.
            Vector3 resolvedPosition = particle.currentPosition;

            Vector3 candidate = resolvedPosition;
            candidate.X += velocity.X * dt;
            if (IsBlocked(Point3.FromVector3(candidate)))
                velocity.X = 0f;
            else
                resolvedPosition.X = candidate.X;

            candidate = resolvedPosition;
            candidate.Y += velocity.Y * dt;
            if (IsBlocked(Point3.FromVector3(candidate)))
                velocity.Y = 0f;
            else
                resolvedPosition.Y = candidate.Y;

            candidate = resolvedPosition;
            candidate.Z += velocity.Z * dt;
            if (IsBlocked(Point3.FromVector3(candidate)))
                velocity.Z = 0f;
            else
                resolvedPosition.Z = candidate.Z;

            Point3 newCell = Point3.FromVector3(resolvedPosition);
            if (!newCell.Equals(particle.currentCell))
            {
                fluidGrid.SetFluid(particle.currentCell, false);
                dirtyCells.Add(particle.currentCell);
                particle.currentCell = newCell;
                fluidGrid.SetFluid(particle.currentCell, true);
                dirtyCells.Add(particle.currentCell);
            }

            particle.currentPosition = resolvedPosition;
            fluidGrid.SetVelocity(particle.currentCell, velocity);
        }

        // A cell blocks fluid movement if it's outside the addressable world, or occupied by solid world
        // geometry that isn't itself fluid. The fluidGrid check matters because fluid voxels are drawn through the
        // same shared world data as real terrain and use the same "solid" render bit like in the original grid, so
        // without excluding fluid cells here a particle would read its own just-flushed position as solid
        // terrain and immediately collide with itself. fluidGrid reflects live simulation state, which is updated the
        // instant a particle moves, so it's the correct source of truth for "is this actually fluid"
        // WorldData's voxel data only catches up once ApplyToWorld flushes.
        // This was a fix to a bug where a particle would collide with itself and stop moving if it was the only particle in a cell
        private bool IsBlocked(Point3 cell)
        {
            if (!fluidGrid.IsInside(cell))
                return true;
            if (fluidGrid.IsFluid(cell))
                return false;
            return worldData.IsVoxelSolid(cell);
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
