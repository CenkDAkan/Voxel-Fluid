using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Render;
using System;
using System.Collections.Generic;
using System.Diagnostics;

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

        // Cells this handler has drawn as fluid, one bit per cell, held per brick of the solver's size and keyed by brick position
        // The baseline ApplyToWorld compares against, and excluded from IsSolid
        private class DrawnBrick
        {
            public uint[] bits = new uint[SparseFluidGrid.BrickCells / 32];
            public bool seen;
        }

        private Dictionary<long, DrawnBrick> drawnBricks = new Dictionary<long, DrawnBrick>();
        private uint[] visibleMask = new uint[0];
        private readonly List<long> drawnBricksToRemove = new List<long>();

        private static readonly Point3[] NeighborOffsets =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;
        private const float physicsBudgetMsPerFrame = 12f; // once a frame's ticks have taken this long, the rest of the backlog is dropped

        // Drive LogTick's finer-grained window after each seed or emitter placement
        private int ticksSinceSeed = 0;
        private bool loggedFirstContact = false;
        private int? lastKnownFloorY = null;
        private double volumeAtLastLog = 0.0;

        // Above this many cells the log prints every 62 ticks from the start, instead of every 4 for the first 125
        private const int fineLogMaxCells = 20000;

        // Time spent outside the solver, and ticks counted, since the last timing line
        private const int timingLogEveryTicks = 125;
        private int ticksSinceTimingLog = 0, framesSinceTimingLog = 0;
        private double logMs = 0.0, applyToWorldMs = 0.0;

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

        // Run the solver through its compute shader stages; with the self-check on, the CPU version of each runs too and stays authoritative
        public bool gpuEnabled = false;
        public bool gpuSelfCheck = false;
        public bool gpuMeasureKernels = false; // logs the time the kernels of each stage take by waiting for the GPU around them
        private SparseFluidGpu gpu;

        public float emitterSpeed = 25f;
        public float sprinklerArcDegrees = 120f;
        public float sprinklerAngularSpeedDegPerSec = 90f;
        public float sprinklerTiltDegrees = 35f;

        // Creates the solver with this handler's solid and drain lookups, and registers the fluid and emitter voxel types
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

        // Tells the solver about every chunk edited since the last GPU upload (editing tools, scene buttons), so it re-queries solids there
        private void InvalidateEditedTerrain()
        {
            ChangeHandler changeHandler = worldData.changeHandler;
            for (int i = 0; i < changeHandler.changedChunkCount; i++)
            {
                uint packed = changeHandler.changedChunks[i].data1;
                Point3 min = new Point3((int)(packed & 0x7FF), (int)((packed >> 11) & 0x3FF), (int)(packed >> 21)) * 16;
                solver.InvalidateSolid(min, min + new Point3(15));
            }
        }

        // Per frame: reacts to terrain edits and input, runs the due physics ticks on the CPU or GPU within the frame budget, then draws the result
        public void Update(float gameTime)
        {
            InvalidateEditedTerrain();
            HandlePlaceInput();
            HandleEmitterInput();

            if (solver.Count == 0 && emitters.Count == 0)
                return;

            if (gpuEnabled)
            {
                gpu ??= new SparseFluidGpu(solver, App.graphicsDevice, App.contentManager.Load<Microsoft.Xna.Framework.Graphics.Effect>("shaders/fluid/sparseFluidSim"));
                gpu.selfCheck = gpuSelfCheck;
                gpu.measureKernels = gpuMeasureKernels;
            }
            else
                gpu?.Release();

            // At least one tick per frame when one is due, then more until the frame's time budget is used up
            physicsAccumulatorMs += gameTime;
            bool stepped = false;
            bool overBudget = false;
            long budgetStart = Stopwatch.GetTimestamp();
            while (physicsAccumulatorMs >= physicsIntervalMs && !overBudget)
            {
                physicsAccumulatorMs -= physicsIntervalMs;
                float dt = physicsIntervalMs / 1000f;
                List<SparseFluidSolver.Source> sources = BuildSources(dt);
                if (gpuEnabled)
                    gpu.Step(dt, sources);
                else
                    solver.Step(dt, sources);
                ticksSinceSeed++;
                long logStart = Stopwatch.GetTimestamp();
                LogTick();
                logMs += (Stopwatch.GetTimestamp() - logStart) * 1000.0 / Stopwatch.Frequency;
                ticksSinceTimingLog++;
                stepped = true;
                overBudget = (Stopwatch.GetTimestamp() - budgetStart) * 1000.0 / Stopwatch.Frequency >= physicsBudgetMsPerFrame;
                if (physicsIntervalMs <= 0f) break;
            }
            if (overBudget)
                physicsAccumulatorMs = 0f; // drop the backlog rather than compound it into the next frame

            if (stepped)
            {
                long applyStart = Stopwatch.GetTimestamp();
                ApplyToWorld();
                applyToWorldMs += (Stopwatch.GetTimestamp() - applyStart) * 1000.0 / Stopwatch.Frequency;
                framesSinceTimingLog++;
            }

            if (ticksSinceTimingLog >= timingLogEveryTicks)
                LogTiming();
        }

        // Average time per tick of every solver stage, plus this handler's own logging (per tick) and drawing (per frame)
        private void LogTiming()
        {
            double solverTotal = 0.0;
            string stages = "";
            for (int i = 0; i < solver.stageMs.Length; i++)
            {
                solverTotal += solver.stageMs[i];
                stages += $"{SparseFluidSolver.StageNames[i]} {solver.stageMs[i] / ticksSinceTimingLog:F2}, ";
                solver.stageMs[i] = 0.0;
            }
            Console.WriteLine($"SparseFluidHandler: timing over {ticksSinceTimingLog} ticks in {framesSinceTimingLog} frames, bricks = {solver.grid.BrickCount}, ms/tick: solver {solverTotal / ticksSinceTimingLog:F2} ({stages.TrimEnd(',', ' ')}), logging {logMs / ticksSinceTimingLog:F2}; ApplyToWorld {applyToWorldMs / Math.Max(framesSinceTimingLog, 1):F2} ms/frame");

            ticksSinceTimingLog = 0;
            framesSinceTimingLog = 0;
            logMs = applyToWorldMs = 0.0;
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
            gpu?.Release();
            solver.SeedBox(min, max);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Console.WriteLine($"SparseFluidHandler: filled box ({min.X}, {min.Y}, {min.Z}) to ({max.X}, {max.Y}, {max.Z}), tracked cells = {solver.Count}, volume = {volumeAtLastLog:F2}.");
        }

        // Fills a sphere with fluid at rest and restarts the log window and volume ledger
        private void SeedSphere(Point3 center, int radius)
        {
            gpu?.Release();
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

        // Places a tap or sprinkler at the aim point: its marker voxel replaces any fluid there and the stream starts next tick
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

            gpu?.Release();
            solver.RemoveCell(position);
            SetDrawn(position, false);
            PlaceMarkerVoxel(position);
            solver.InvalidateSolid(position, position);

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
            {
                WriteCell(emitter.position, false);
                solver.InvalidateSolid(emitter.position, emitter.position);
            }
            emitters.Clear();
            CommitEdits();
        }

        // Writes an emitter's marker voxel into the world
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
            if (drawnBricks.Count == 0 && solver.Count == 0 && emitters.Count == 0)
                return;
            gpu?.Release();

            foreach (KeyValuePair<long, DrawnBrick> drawn in drawnBricks)
                EraseDrawnBrick(drawn.Key, drawn.Value);
            drawnBricks.Clear();

            foreach (Emitter emitter in emitters)
                WriteCell(emitter.position, false);
            emitters.Clear();

            CommitEdits();

            solver.Clear();
            physicsAccumulatorMs = 0f;
        }

        // Solid lookup for the solver: outside the world is solid, voxels this handler drew as fluid are not
        private bool IsSolid(Point3 worldVoxel)
        {
            if (!IsInsideWorld(worldVoxel))
                return true;
            if (IsDrawn(worldVoxel))
                return false;
            return worldData.IsVoxelSolid(worldVoxel);
        }

        // Whether a voxel position lies inside the world
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

        // Key of the drawn-cell brick holding a cell
        private static long DrawnBrickKey(Point3 cell)
        {
            return SparseFluidGrid.Key(new Point3(cell.X >> SparseFluidGrid.BrickShift, cell.Y >> SparseFluidGrid.BrickShift, cell.Z >> SparseFluidGrid.BrickShift));
        }

        // Bit of a cell within its drawn-cell brick, in the same order as the grid's local cell index
        private static int DrawnBit(Point3 cell)
        {
            return (cell.X & SparseFluidGrid.BrickMask) | ((cell.Y & SparseFluidGrid.BrickMask) << SparseFluidGrid.BrickShift) | ((cell.Z & SparseFluidGrid.BrickMask) << (SparseFluidGrid.BrickShift * 2));
        }

        // Whether this handler currently has the cell drawn as fluid
        private bool IsDrawn(Point3 cell)
        {
            if (!drawnBricks.TryGetValue(DrawnBrickKey(cell), out DrawnBrick drawn))
                return false;
            int bit = DrawnBit(cell);
            return (drawn.bits[bit >> 5] & (1u << (bit & 31))) != 0;
        }

        // Records whether the cell is drawn as fluid, creating its brick entry only when setting a bit
        private void SetDrawn(Point3 cell, bool value)
        {
            long key = DrawnBrickKey(cell);
            if (!drawnBricks.TryGetValue(key, out DrawnBrick drawn))
            {
                if (!value)
                    return;
                drawn = new DrawnBrick { seen = true };
                drawnBricks[key] = drawn;
            }

            int bit = DrawnBit(cell);
            if (value)
                drawn.bits[bit >> 5] |= 1u << (bit & 31);
            else
                drawn.bits[bit >> 5] &= ~(1u << (bit & 31));
        }

        // Erases every cell drawn in one brick
        private void EraseDrawnBrick(long key, DrawnBrick drawn)
        {
            Point3 origin = SparseFluidGrid.Unpack(key) * SparseFluidGrid.BrickSize;
            for (int element = 0; element < drawn.bits.Length; element++)
            {
                uint bits = drawn.bits[element];
                while (bits != 0)
                {
                    int local = element * 32 + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    bits &= bits - 1;
                    WriteCell(origin + LocalCell(local), false);
                }
                drawn.bits[element] = 0;
            }
        }

        // Offset of a cell from its brick origin, from its local index
        private static Point3 LocalCell(int local)
        {
            return new Point3(local & SparseFluidGrid.BrickMask, (local >> SparseFluidGrid.BrickShift) & SparseFluidGrid.BrickMask, local >> (SparseFluidGrid.BrickShift * 2));
        }

        // One bit per cell of every brick slot: tracked and above the visibility threshold
        private void BuildVisibleMask()
        {
            SparseFluidGrid grid = solver.grid;
            int elements = grid.SlotCount * (SparseFluidGrid.BrickCells / 32);
            if (visibleMask.Length < elements)
                visibleMask = new uint[Math.Max(elements, visibleMask.Length * 2)];
            if (gpuEnabled && gpu != null && gpu.ReadVisibleMask(visibleMask))
                return;

            for (int element = 0; element < elements; element++)
            {
                uint bits = 0;
                int start = element * 32;
                for (int k = 0; k < 32; k++)
                    if (grid.tracked[start + k] != 0 && grid.fill[start + k] > SparseFluidSolver.visibilityThreshold)
                        bits |= 1u << k;
                visibleMask[element] = bits;
            }
        }

        // Draws cells above the visibility threshold and erases cells that dropped below it or left tracking,
        // going only through the cells whose visibility differs from what is drawn
        private void ApplyToWorld()
        {
            SparseFluidGrid grid = solver.grid;
            BuildVisibleMask();
            bool wrote = false;
            int elementsPerBrick = SparseFluidGrid.BrickCells / 32;

            foreach (DrawnBrick drawn in drawnBricks.Values)
                drawn.seen = false;

            for (int slot = 0; slot < grid.SlotCount; slot++)
            {
                if (!grid.IsSlotAllocated(slot))
                    continue;
                Point3 origin = grid.SlotOrigin(slot);
                if (origin.X < 0 || origin.Y < 0 || origin.Z < 0)
                    continue;

                long key = DrawnBrickKey(origin);
                bool hasDrawn = drawnBricks.TryGetValue(key, out DrawnBrick drawn);
                if (hasDrawn)
                    drawn.seen = true;

                for (int element = 0; element < elementsPerBrick; element++)
                {
                    uint now = visibleMask[slot * elementsPerBrick + element];
                    uint changed = now ^ (hasDrawn ? drawn.bits[element] : 0u);
                    while (changed != 0)
                    {
                        int bit = System.Numerics.BitOperations.TrailingZeroCount(changed);
                        changed &= changed - 1;
                        bool visible = (now & (1u << bit)) != 0;
                        if (!WriteCell(origin + LocalCell(element * 32 + bit), visible))
                            continue;

                        if (!hasDrawn)
                        {
                            drawn = new DrawnBrick { seen = true };
                            drawnBricks[key] = drawn;
                            hasDrawn = true;
                        }
                        if (visible)
                            drawn.bits[element] |= 1u << bit;
                        else
                            drawn.bits[element] &= ~(1u << bit);
                        wrote = true;
                    }
                }
            }

            // Bricks the solver has let go of
            drawnBricksToRemove.Clear();
            foreach (KeyValuePair<long, DrawnBrick> drawn in drawnBricks)
            {
                if (drawn.Value.seen)
                    continue;
                EraseDrawnBrick(drawn.Key, drawn.Value);
                drawnBricksToRemove.Add(drawn.Key);
                wrote = true;
            }
            foreach (long key in drawnBricksToRemove)
                drawnBricks.Remove(key);

            if (wrote)
                CommitEdits();
        }

        // Stages a fluid voxel or air at a world position; returns false when the voxel is outside the world, or when drawing it would overwrite solid terrain
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

        // Takes the current total volume as the ledger's baseline and zeroes the solver's volume counters
        private void ResetVolumeTracking()
        {
            volumeAtLastLog = solver.TotalVolume();
            solver.ClearVolumeAccumulators();
        }

        // Walks every cell, so it only does that on the ticks it prints (and until first solid contact has been reported)
        private void LogTick()
        {
            SparseFluidGrid grid = solver.grid;
            bool fineGrained = ticksSinceSeed <= 125 && grid.Count <= fineLogMaxCells;
            int logEveryTicks = fineGrained ? 4 : 62;
            bool printing = ticksSinceSeed % logEveryTicks == 0;
            if (!printing && loggedFirstContact)
                return;
            gpu?.Download(true);

            int visibleCellCount = 0;
            float totalSpeed = 0f;
            double totalVolume = 0.0;
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

            if (!printing)
                return;

            string bandYInfo = grid.Count > 0 ? $", Y range = [{minY}, {maxY}]" : "";
            string floorInfo = lastKnownFloorY.HasValue ? $", floorY = {lastKnownFloorY.Value}" : "";
            Console.WriteLine($"SparseFluidHandler: tick {ticksSinceSeed} ({ticksSinceSeed * physicsIntervalMs / 1000f:F2}s), visible cells = {visibleCellCount}, tracked cells = {grid.Count}, liquid cells = {solver.lastLiquidCells}, total speed = {totalSpeed:F3}, pressure iterations = {solver.lastPressureIterations} (residual {solver.lastPressureResidual:E1}), pruned this tick (solid/empty/drained) = {solver.lastPrunedBySolid}/{solver.lastPrunedByEmpty}/{solver.lastDrained}, emitters = {emitters.Count}{bandYInfo}{floorInfo}");

            float expectedChange = solver.volumeInjected + solver.volumeClampCreated - solver.volumeClampedAway - solver.volumePrunedSolid - solver.volumePrunedEmpty - solver.volumePrunedStagnant - solver.volumeDrained;
            double unaccounted = (totalVolume - volumeAtLastLog) - expectedChange;
            Console.WriteLine($"    volume = {totalVolume:F2}, since last log: injected +{solver.volumeInjected:F2}, clamp lost -{solver.volumeClampedAway:F3}, clamp created +{solver.volumeClampCreated:F3}, pruned solid -{solver.volumePrunedSolid:F3} / empty -{solver.volumePrunedEmpty:F3} / stagnant -{solver.volumePrunedStagnant:F3}, drained -{solver.volumeDrained:F3}, unaccounted {unaccounted:F3}");

            volumeAtLastLog = totalVolume;
            solver.ClearVolumeAccumulators();
        }
    }
}
