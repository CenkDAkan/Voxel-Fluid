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
    * Engine integration for the sparse Eulerian (Volume-of-Fluid) arm
    *
    * Owns a SparseFluidSolver for the physics and handles everything engine-side: input, emitters,
    * seeding, drawing visible cells into the world through the editing pipeline, and logging
    */
    public class SparseFluidHandler
    {
        private WorldData worldData;
        public readonly SparseFluidSolver solver;

        private uint fluidTypeRenderIndex;

        // Cells this handler has drawn as fluid; the dirty-diff baseline for ApplyToWorld, and excluded from IsSolid
        private HashSet<long> wasFluidVoxel = new HashSet<long>();

        private static readonly Point3[] NeighborOffsets =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;
        private const int maxPhysicsStepsPerFrame = 4;

        // Drive LogTick's finer-grained window after each seed or emitter placement
        private int ticksSinceSeed = 0;
        private bool loggedFirstContact = false;
        private int? lastKnownFloorY = null;
        private float volumeAtLastLog = 0f;

        private enum EmitterKind { Tap, Sprinkler }

        // A continuous fluid source: a marker voxel plus a nozzle of cells held full at the launch velocity every tick
        private class Emitter
        {
            public EmitterKind kind;
            public Point3 position;          // marker voxel
            public Point3 nozzleOffset;      // nozzle center relative to the marker
            public Vector3 forwardXZ;        // horizontal facing, fixed at placement
            public float angleDeg;           // sprinkler sweep offset from forwardXZ
            public int sweepDir = 1;
        }

        private List<Emitter> emitters = new List<Emitter>();
        private uint emitterTypeRenderIndex;
        private const int nozzleRadius = 1; // 3x3 nozzle cross-section

        public float emitterSpeed = 25f;
        public float sprinklerArcDegrees = 120f;
        public float sprinklerAngularSpeedDegPerSec = 90f;
        public float sprinklerTiltDegrees = 35f;

        public SparseFluidHandler(WorldData worldData)
        {
            this.worldData = worldData;
            solver = new SparseFluidSolver(IsSolid, IsWorldBoundary);

            VoxelType sparseEulerianFluidType = new VoxelType
            {
                ID = "fluid_sparse_eulerian_demo",
                colorBase = new Vector3(0.1f, 0.9f, 0.9f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            fluidTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(sparseEulerianFluidType).renderIndex;

            VoxelType emitterType = new VoxelType
            {
                ID = "fluid_sparse_emitter_head",
                colorBase = new Vector3(1.0f, 0.5f, 0.1f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            emitterTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(emitterType).renderIndex;
        }

        public void Update(float gameTime)
        {
            HandlePlaceInput();
            HandleEmitterInput();

            if (solver.Count == 0 && emitters.Count == 0)
                return;

            physicsAccumulatorMs += gameTime;
            bool stepped = false;
            int stepsThisFrame = 0;
            while (physicsAccumulatorMs >= physicsIntervalMs && stepsThisFrame < maxPhysicsStepsPerFrame)
            {
                physicsAccumulatorMs -= physicsIntervalMs;
                float dt = physicsIntervalMs / 1000f;
                solver.Step(dt, BuildSources(dt));
                ticksSinceSeed++;
                LogTick();
                stepped = true;
                stepsThisFrame++;
                if (physicsIntervalMs <= 0f) break;
            }
            if (stepsThisFrame >= maxPhysicsStepsPerFrame)
                physicsAccumulatorMs = 0f; // drop the backlog rather than compound it into the next frame

            if (stepped)
                ApplyToWorld();
        }

        // P drops a seed sphere aimed from the camera
        private void HandlePlaceInput()
        {
            if (!IO.KBStates.IsKeyToggleDown(Keys.P))
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            SeedSphere(Point3.FromVector3(camPos + camDir * 20f), 4);
        }

        // Seeds a sphere a few voxels above the floor below the aim point; no longer called when the mode is selected
        public void SeedDefaultScenario()
        {
            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Vector3 aimPoint = camPos + camDir * 20f;

            uint hitType = worldData.RayTraversal(aimPoint + new Vector3(0, 200, 0), new Vector3(0, -1, 0), out float hitLength, out Point3 hitVoxel, out Point3 hitNormal);
            int floorY = hitType != 0 ? hitVoxel.Y + 1 : (int)aimPoint.Y - 40;
            lastKnownFloorY = floorY;

            SeedSphere(new Point3((int)aimPoint.X, floorY + 6, (int)aimPoint.Z), 4);
        }

        // Fills a box (bounds inclusive) with fluid at rest, for seeding large volumes at once
        public void SeedBox(Point3 min, Point3 max)
        {
            solver.SeedBox(min, max);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Console.WriteLine($"SparseFluidHandler: filled box ({min.X}, {min.Y}, {min.Z}) to ({max.X}, {max.Y}, {max.Z}), tracked cells = {solver.Count}, volume = {volumeAtLastLog:F2}.");
        }

        private void SeedSphere(Point3 center, int radius)
        {
            solver.SeedSphere(center, radius);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Console.WriteLine($"SparseFluidHandler: seeded {solver.Count} fluid cells around ({center.X}, {center.Y}, {center.Z}), volume = {volumeAtLastLog:F2}.");
        }

        // T places a tap, J places a sprinkler, both aimed from the camera
        private void HandleEmitterInput()
        {
            if (IO.KBStates.IsKeyToggleDown(Keys.T))
                PlaceEmitter(EmitterKind.Tap);
            if (IO.KBStates.IsKeyToggleDown(Keys.J))
                PlaceEmitter(EmitterKind.Sprinkler);
        }

        private void PlaceEmitter(EmitterKind kind)
        {
            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Point3 position = Point3.FromVector3(camPos + camDir * 20f);

            if (IsSolid(position))
            {
                Console.WriteLine("SparseFluidHandler: emitter point is solid or outside the world, aim somewhere else.");
                return;
            }

            Vector3 forwardXZ = new Vector3(camDir.X, 0f, camDir.Z);
            forwardXZ = forwardXZ.LengthSquared() > 1e-6f ? Vector3.Normalize(forwardXZ) : Vector3.UnitZ;

            Point3 nozzleOffset = kind == EmitterKind.Tap
                ? DominantAxisStep(forwardXZ)
                : new Point3(0, nozzleRadius + 1, 0);

            solver.RemoveCell(position);
            wasFluidVoxel.Remove(SparseFluidGrid.Key(position));
            PlaceMarkerVoxel(position);

            emitters.Add(new Emitter { kind = kind, position = position, nozzleOffset = nozzleOffset, forwardXZ = forwardXZ });
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Console.WriteLine($"SparseFluidHandler: placed {kind} at ({position.X}, {position.Y}, {position.Z}).");
        }

        // Erases every emitter marker and stops all injection, leaving existing fluid in place
        public void RemoveAllEmitters()
        {
            foreach (Emitter emitter in emitters)
                WriteCell(emitter.position, false);
            emitters.Clear();
            CommitEdits();
        }

        private void PlaceMarkerVoxel(Point3 worldVoxel)
        {
            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, (1u << 15) | emitterTypeRenderIndex);
            CommitEdits();
        }

        // Advances each sprinkler's sweep and returns every nozzle cell with its emitter's launch velocity
        private List<SparseFluidSolver.Source> BuildSources(float dt)
        {
            List<SparseFluidSolver.Source> sources = new List<SparseFluidSolver.Source>();
            foreach (Emitter emitter in emitters)
            {
                float tiltDeg = 0f;
                if (emitter.kind == EmitterKind.Sprinkler)
                {
                    tiltDeg = sprinklerTiltDegrees;
                    float halfArc = sprinklerArcDegrees * 0.5f;
                    emitter.angleDeg += sprinklerAngularSpeedDegPerSec * dt * emitter.sweepDir;
                    if (emitter.angleDeg > halfArc)
                    {
                        emitter.angleDeg = halfArc;
                        emitter.sweepDir = -1;
                    }
                    else if (emitter.angleDeg < -halfArc)
                    {
                        emitter.angleDeg = -halfArc;
                        emitter.sweepDir = 1;
                    }
                }

                Vector3 horizontal = Vector3.Transform(emitter.forwardXZ, Matrix.CreateRotationY(MathHelper.ToRadians(emitter.angleDeg)));
                float tiltRad = MathHelper.ToRadians(tiltDeg);
                Vector3 launchVelocity = (horizontal * MathF.Cos(tiltRad) + Vector3.Up * MathF.Sin(tiltRad)) * emitterSpeed;

                // Nozzle cross-section spans the two axes perpendicular to the launch direction's dominant axis
                Point3 flowAxis = DominantAxisStep(launchVelocity);
                Point3 axisA = flowAxis.X != 0 ? new Point3(0, 1, 0) : new Point3(1, 0, 0);
                Point3 axisB = flowAxis.Z != 0 ? new Point3(0, 1, 0) : new Point3(0, 0, 1);

                Point3 center = emitter.position + emitter.nozzleOffset;
                for (int a = -nozzleRadius; a <= nozzleRadius; a++)
                    for (int b = -nozzleRadius; b <= nozzleRadius; b++)
                        sources.Add(new SparseFluidSolver.Source { cell = center + axisA * a + axisB * b, velocity = launchVelocity });
            }
            return sources;
        }

        // Unit step along whichever axis dominates v, with v's sign
        private static Point3 DominantAxisStep(Vector3 v)
        {
            float ax = MathF.Abs(v.X), ay = MathF.Abs(v.Y), az = MathF.Abs(v.Z);
            if (ax >= ay && ax >= az)
                return new Point3(MathF.Sign(v.X) >= 0 ? 1 : -1, 0, 0);
            if (ay >= az)
                return new Point3(0, MathF.Sign(v.Y) >= 0 ? 1 : -1, 0);
            return new Point3(0, 0, MathF.Sign(v.Z) >= 0 ? 1 : -1);
        }

        // Erases every cell and marker this handler has drawn and drops all simulation state
        public void ClearAll()
        {
            if (wasFluidVoxel.Count == 0 && solver.Count == 0 && emitters.Count == 0)
                return;

            foreach (long key in wasFluidVoxel)
                WriteCell(SparseFluidGrid.Unpack(key), false);
            wasFluidVoxel.Clear();

            foreach (Emitter emitter in emitters)
                WriteCell(emitter.position, false);
            emitters.Clear();

            CommitEdits();

            solver.Clear();
            physicsAccumulatorMs = 0f;
        }

        private bool IsSolid(Point3 worldVoxel)
        {
            if (!IsInsideWorld(worldVoxel))
                return true;
            if (wasFluidVoxel.Contains(SparseFluidGrid.Key(worldVoxel)))
                return false;
            return worldData.IsVoxelSolid(worldVoxel);
        }

        private bool IsInsideWorld(Point3 p)
        {
            return p.X >= 0 && p.Y >= 0 && p.Z >= 0
                && p.X < worldData.sizeInVoxels.X && p.Y < worldData.sizeInVoxels.Y && p.Z < worldData.sizeInVoxels.Z;
        }

        // The outermost voxel layer of the world; fluid reaching it leaves the simulation
        private bool IsWorldBoundary(Point3 p)
        {
            return p.X <= 0 || p.Y <= 0 || p.Z <= 0
                || p.X >= worldData.sizeInVoxels.X - 1 || p.Y >= worldData.sizeInVoxels.Y - 1 || p.Z >= worldData.sizeInVoxels.Z - 1;
        }

        // Processes the staged edits and uploads them as a GPU batch of their own, after flushing any batch still pending
        private void CommitEdits()
        {
            if (worldData.changeHandler.changedChunkCount > 0)
                worldData.changeHandler.UpdateWorld();
            worldData.editingHandler.processChunks(true);
        }

        // Draws cells above the visibility threshold and erases cells that dropped below it or left tracking
        private void ApplyToWorld()
        {
            SparseFluidGrid grid = solver.grid;
            HashSet<long> nowFluid = new HashSet<long>();
            foreach (long key in grid.SnapshotKeys())
                if (grid.GetFill(key) > SparseFluidSolver.visibilityThreshold)
                    nowFluid.Add(key);

            foreach (long key in nowFluid)
            {
                if (!wasFluidVoxel.Contains(key) && WriteCell(SparseFluidGrid.Unpack(key), true))
                    wasFluidVoxel.Add(key);
            }

            List<long> toErase = new List<long>();
            foreach (long key in wasFluidVoxel)
                if (!nowFluid.Contains(key))
                    toErase.Add(key);

            foreach (long key in toErase)
                if (WriteCell(SparseFluidGrid.Unpack(key), false))
                    wasFluidVoxel.Remove(key);

            CommitEdits();
        }

        private bool WriteCell(Point3 worldVoxel, bool visible)
        {
            if (!IsInsideWorld(worldVoxel))
                return false;
            if (visible && worldData.IsVoxelSolid(worldVoxel))
                return false;

            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            uint type = visible ? (1u << 15) | fluidTypeRenderIndex : 0u;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);
            return true;
        }

        private void ResetVolumeTracking()
        {
            volumeAtLastLog = solver.TotalVolume();
            solver.ClearVolumeAccumulators();
        }

        private void LogTick()
        {
            SparseFluidGrid grid = solver.grid;
            int visibleCellCount = 0;
            float totalSpeed = 0f;
            float totalVolume = 0f;
            bool contactThisTick = false;
            int minY = int.MaxValue, maxY = int.MinValue;

            foreach (long key in grid.SnapshotKeys())
            {
                Point3 trackedPos = SparseFluidGrid.Unpack(key);
                if (trackedPos.Y < minY) minY = trackedPos.Y;
                if (trackedPos.Y > maxY) maxY = trackedPos.Y;

                float fill = grid.GetFill(key);
                totalSpeed += grid.CenterVelocity(trackedPos).Length();
                totalVolume += fill;
                if (fill <= SparseFluidSolver.visibilityThreshold)
                    continue;

                visibleCellCount++;

                if (!loggedFirstContact && !contactThisTick)
                {
                    foreach (Point3 offset in NeighborOffsets)
                    {
                        if (IsSolid(trackedPos + offset))
                        {
                            contactThisTick = true;
                            break;
                        }
                    }
                }
            }

            if (contactThisTick && !loggedFirstContact)
            {
                loggedFirstContact = true;
                Console.WriteLine($"SparseFluidHandler: first solid contact at tick {ticksSinceSeed} ({ticksSinceSeed * physicsIntervalMs / 1000f:F2}s), visible cells = {visibleCellCount}, tracked cells = {grid.Count}.");
            }

            bool fineGrained = ticksSinceSeed <= 125;
            int logEveryTicks = fineGrained ? 4 : 62;
            if (ticksSinceSeed % logEveryTicks != 0)
                return;

            string bandYInfo = grid.Count > 0 ? $", Y range = [{minY}, {maxY}]" : "";
            string floorInfo = lastKnownFloorY.HasValue ? $", floorY = {lastKnownFloorY.Value}" : "";
            Console.WriteLine($"SparseFluidHandler: tick {ticksSinceSeed} ({ticksSinceSeed * physicsIntervalMs / 1000f:F2}s), visible cells = {visibleCellCount}, tracked cells = {grid.Count}, liquid cells = {solver.lastLiquidCells}, total speed = {totalSpeed:F3}, pressure iterations = {solver.lastPressureIterations} (residual {solver.lastPressureResidual:E1}), pruned this tick (solid/empty/drained) = {solver.lastPrunedBySolid}/{solver.lastPrunedByEmpty}/{solver.lastDrained}, emitters = {emitters.Count}{bandYInfo}{floorInfo}");

            float expectedChange = solver.volumeInjected + solver.volumeClampCreated - solver.volumeClampedAway - solver.volumePrunedSolid - solver.volumePrunedEmpty - solver.volumePrunedStagnant - solver.volumeDrained;
            float unaccounted = (totalVolume - volumeAtLastLog) - expectedChange;
            Console.WriteLine($"    volume = {totalVolume:F2}, since last log: injected +{solver.volumeInjected:F2}, clamp lost -{solver.volumeClampedAway:F3}, clamp created +{solver.volumeClampCreated:F3}, pruned solid -{solver.volumePrunedSolid:F3} / empty -{solver.volumePrunedEmpty:F3} / stagnant -{solver.volumePrunedStagnant:F3}, drained -{solver.volumeDrained:F3}, unaccounted {unaccounted:F3}");

            volumeAtLastLog = totalVolume;
            solver.ClearVolumeAccumulators();
        }
    }
}
