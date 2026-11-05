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

        // The fluid's shades from deep blue to foam, as registered voxel types; calm cells pick from the first four, fast ones from the last three
        private const int ShadeCount = 5, PlainShade = 2;
        private static readonly Vector3[] ShadeColors =
        {
            new Vector3(0.04f, 0.35f, 0.7f), new Vector3(0.07f, 0.6f, 0.85f), new Vector3(0.1f, 0.9f, 0.9f), new Vector3(0.5f, 0.95f, 0.95f), new Vector3(0.85f, 0.97f, 1.0f),
        };
        private static readonly float[] CalmShadeWeights = { 0.2f, 0.3f, 0.4f, 0.1f, 0f };
        private static readonly float[] FastShadeWeights = { 0f, 0f, 0.15f, 0.35f, 0.5f };
        private readonly uint[] shadeRenderIndex = new uint[ShadeCount];
        private readonly Random shadeRandom = new Random();

        // Drawn cells take a random shade, fast ones (center speed above foamSpeed) leaning to foam, and change it when they speed up or settle;
        // shimmerPerSecond of the drawn cells with nothing drawn above pick a new shade each second. Off, every cell is drawn in the plain shade
        public bool colorVariation = true;
        public float foamSpeed = 8f;
        public float shimmerPerSecond = 0.05f;
        private long lastApplyTimestamp;

        // Cells this handler has drawn as fluid, one bit per cell, held per brick of the solver's size and keyed by brick position
        // The baseline ApplyToWorld compares against, and excluded from IsSolid; also the shade each cell was drawn in and whether it counted as fast
        private class DrawnBrick
        {
            public uint[] bits = new uint[SparseFluidGrid.BrickCells / 32];
            public uint[] fast = new uint[SparseFluidGrid.BrickCells / 32];
            public byte[] shade = new byte[SparseFluidGrid.BrickCells];
            public bool seen;
        }

        private Dictionary<long, DrawnBrick> drawnBricks = new Dictionary<long, DrawnBrick>();
        private uint[] drawMask = new uint[0], keepMask = new uint[0], fastMask = new uint[0];

        // A cell is drawn once its fill rises above visibilityThreshold + drawHysteresis and erased once it falls below
        // visibilityThreshold - drawHysteresis, so a surface resting right at the threshold doesn't flicker
        public float drawHysteresis = 0.05f;
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
        private double volumeAtLastLog = 0.0;

        // Above this many cells the log prints every 62 ticks from the start, instead of every 4 for the first 125
        private const int fineLogMaxCells = 20000;

        // Time spent outside the solver, and ticks counted, since the last timing line
        private const int timingLogEveryTicks = 125;
        private int ticksSinceTimingLog = 0, framesSinceTimingLog = 0;
        private double logMs = 0.0, applyToWorldMs = 0.0;

        public enum EmitterKind { Tap, Sprinkler }

        // A continuous fluid source: a marker voxel plus a nozzle of cells held full at the launch velocity while it runs
        public class Emitter
        {
            public EmitterKind kind;
            public Point3 position;          // marker voxel
            public Vector3 forwardXZ;        // horizontal facing
            public float angleDeg;           // sprinkler sweep offset from forwardXZ
            public int sweepDir = 1;
            public bool enabled = true;
            public float speed;              // launch speed in voxels/s
            public float tiltDeg;            // launch angle above the horizontal, negative aims down
            public float arcDeg, sweepSpeedDeg; // sprinklers: sweep arc in degrees and sweep speed in degrees/s
            public int nozzleRadius;         // the nozzle is (2 * nozzleRadius + 1)^2 cells across the launch direction
            public float onSeconds, offSeconds, startDelaySeconds; // after the delay, runs onSeconds then pauses offSeconds, repeating; onSeconds 0 = always on
            public float runSeconds;         // time since placement or the last timer restart
        }

        private readonly List<Emitter> emitters = new List<Emitter>();
        public IReadOnlyList<Emitter> Emitters => emitters;
        public int SelectedEmitter { get; private set; } = -1;
        private uint emitterTypeRenderIndex, selectedEmitterTypeRenderIndex;

        // Run the solver through its compute shader stages; with the self-check on, the CPU version of each runs too and stays authoritative
        public bool gpuEnabled = true;
        public bool gpuSelfCheck = false;
        public bool gpuMeasureKernels = false; // logs the time the kernels of each stage take by waiting for the GPU around them
        private SparseFluidGpu gpu;

        // Console logging of ticks, volumes, timings and events; while off, none of the work behind it is done
        public bool loggingEnabled = false;
        private bool wasLogging = false;

        // Settings given to newly placed emitters
        public float emitterSpeed = 25f;
        public int emitterNozzleRadius = 0;
        public float sprinklerArcDegrees = 120f;
        public float sprinklerAngularSpeedDegPerSec = 90f;
        public float sprinklerTiltDegrees = 35f;

        // Rain: cubic drops of rainDropSize cells, each held full for one tick at rainHeight above the ground, falling at rainDropSpeed,
        // at random points of a square of half side rainHalfSize around the point that was aimed at when rain was switched on
        public bool rainEnabled = false;
        public float rainDropsPerSecond = 10f;
        public int rainDropSize = 3;
        public int rainHalfSize = 40;
        public int rainHeight = 40;
        public float rainDropSpeed = 10f;
        private Point3 rainCenter;
        private int rainGroundY;
        private float rainDropBacklog = 0f;
        private readonly Random rainRandom = new Random();

        // Creates the solver with this handler's solid and drain lookups, and registers the fluid and emitter voxel types
        public SparseFluidHandler(WorldData worldData)
        {
            this.worldData = worldData;
            solver = new SparseFluidSolver(IsSolid, IsWorldBoundary);

            for (int shade = 0; shade < ShadeCount; shade++)
            {
                VoxelType shadeType = new VoxelType
                {
                    ID = shade == PlainShade ? "fluid_sparse_eulerian_demo" : $"fluid_sparse_eulerian_shade{shade}",
                    colorBase = ShadeColors[shade],
                    colorLayered = Vector3.Zero,
                    materialBase = MaterialTypeBase.Emissive,
                    materialLayer = MaterialTypeLayer.None,
                    roughness = 1.0f,
                };
                shadeRenderIndex[shade] = App.worldHandler.voxelTypeHandler.ApplyVoxelType(shadeType).renderIndex;
            }

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

            VoxelType selectedEmitterType = new VoxelType
            {
                ID = "fluid_sparse_emitter_head_selected",
                colorBase = new Vector3(1.0f, 1.0f, 0.35f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            selectedEmitterTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(selectedEmitterType).renderIndex;
        }

        // Tells the solver about every chunk edited since the last GPU upload (editing tools, scene buttons), so it re-queries solids there
        private void InvalidateEditedTerrain()
        {
            ChangeHandler changeHandler = worldData.changeHandler;
            for (int i = 0; i < changeHandler.changedChunkCount; i++)
            {
                uint packed = changeHandler.changedChunks[i].data1;
                Point3 min = new Point3((int)(packed & 0x7FF), (int)((packed >> 11) & 0x3FF), (int)(packed >> 21)) * 16;
                ForgetOverwrittenCells(min);
                solver.InvalidateSolid(min, min + new Point3(15));
            }
        }

        // Whether a voxel type is one of the fluid's shades
        private bool IsFluidShade(uint type)
        {
            foreach (uint shadeType in shadeRenderIndex)
                if (type == shadeType)
                    return true;
            return false;
        }

        // Clears the drawn bit of every cell in a 16^3 chunk whose voxel no longer holds one of the fluid's shades,
        // so that IsSolid sees terrain an editing tool placed there and erasing never writes air over it
        private void ForgetOverwrittenCells(Point3 chunkMin)
        {
            for (int z = 0; z < 16; z += SparseFluidGrid.BrickSize)
                for (int y = 0; y < 16; y += SparseFluidGrid.BrickSize)
                    for (int x = 0; x < 16; x += SparseFluidGrid.BrickSize)
                    {
                        Point3 origin = chunkMin + new Point3(x, y, z);
                        if (!drawnBricks.TryGetValue(DrawnBrickKey(origin), out DrawnBrick drawn))
                            continue;

                        for (int element = 0; element < drawn.bits.Length; element++)
                        {
                            uint bits = drawn.bits[element];
                            while (bits != 0)
                            {
                                int bit = System.Numerics.BitOperations.TrailingZeroCount(bits);
                                bits &= bits - 1;
                                if (!IsFluidShade(worldData.GetVoxelType(origin + LocalCell(element * 32 + bit))))
                                    drawn.bits[element] &= ~(1u << bit);
                            }
                        }
                    }
        }

        // Per frame: reacts to terrain edits and input, runs the due physics ticks on the CPU or GPU within the frame budget, then draws the result
        public void Update(float gameTime)
        {
            InvalidateEditedTerrain();
            HandlePlaceInput();
            HandleEmitterInput();

            if (solver.Count == 0 && emitters.Count == 0 && !rainEnabled)
                return;

            if (gpuEnabled && gpu == null)
            {
                try
                {
                    gpu = new SparseFluidGpu(solver, App.graphicsDevice, App.contentManager.Load<Microsoft.Xna.Framework.Graphics.Effect>("shaders/fluid/sparseFluidSim"));
                }
                catch (Exception e)
                {
                    // Without compute shader support, or if the shader fails to load, the simulation stays on the CPU
                    Console.WriteLine($"SparseFluidHandler: GPU simulation unavailable, running on the CPU ({e.Message})");
                    gpuEnabled = false;
                }
            }

            if (gpuEnabled)
            {
                gpu.selfCheck = gpuSelfCheck;
                gpu.measureKernels = gpuMeasureKernels;
                gpu.logging = loggingEnabled;
            }
            else if (gpu != null)
            {
                // Hands the state back to the CPU and frees the GPU buffers, switching the GPU back on creates them again
                gpu.Dispose();
                gpu = null;
            }

            if (loggingEnabled && !wasLogging)
                StartLogging();
            wasLogging = loggingEnabled;

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
                if (loggingEnabled)
                {
                    long logStart = Stopwatch.GetTimestamp();
                    LogTick();
                    logMs += (Stopwatch.GetTimestamp() - logStart) * 1000.0 / Stopwatch.Frequency;
                    ticksSinceTimingLog++;
                }
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
                if (loggingEnabled)
                {
                    applyToWorldMs += (Stopwatch.GetTimestamp() - applyStart) * 1000.0 / Stopwatch.Frequency;
                    framesSinceTimingLog++;
                }
            }

            if (loggingEnabled && ticksSinceTimingLog >= timingLogEveryTicks)
                LogTiming();
        }

        // Called when logging is switched on: takes the current volume as the ledger's baseline and starts the timing counters afresh
        private void StartLogging()
        {
            gpu?.Download(false);
            ResetVolumeTracking();
            Array.Clear(solver.stageMs);
            ticksSinceTimingLog = framesSinceTimingLog = 0;
            logMs = applyToWorldMs = 0.0;
        }

        // Prints a line when logging is on
        private void Log(string line)
        {
            if (loggingEnabled)
                Console.WriteLine(line);
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

        // Fills a box (bounds inclusive) with fluid at rest, the top layer to topFill, for seeding large volumes at once
        public void SeedBox(Point3 min, Point3 max, float topFill = 1f)
        {
            gpu?.Release();
            solver.SeedBox(min, max, topFill);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Log($"SparseFluidHandler: filled box ({min.X}, {min.Y}, {min.Z}) to ({max.X}, {max.Y}, {max.Z}), tracked cells = {solver.Count}, volume = {volumeAtLastLog:F2}.");
        }

        // Fills a sphere with fluid at rest and restarts the log window and volume ledger
        private void SeedSphere(Point3 center, int radius)
        {
            gpu?.Release();
            solver.SeedSphere(center, radius);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Log($"SparseFluidHandler: seeded {solver.Count} fluid cells around ({center.X}, {center.Y}, {center.Z}), volume = {volumeAtLastLog:F2}.");
        }

        // T places a tap, J places a sprinkler, both aimed from the camera
        private void HandleEmitterInput()
        {
            if (IO.KBStates.IsKeyToggleDown(Keys.T))
                PlaceEmitter(EmitterKind.Tap);
            if (IO.KBStates.IsKeyToggleDown(Keys.J))
                PlaceEmitter(EmitterKind.Sprinkler);
        }

        // Places a tap or sprinkler at the aim point with the settings for new emitters and selects it; its marker voxel
        // replaces any fluid there and the stream starts next tick
        private void PlaceEmitter(EmitterKind kind)
        {
            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Point3 position = Point3.FromVector3(camPos + camDir * 20f);

            if (IsSolid(position))
            {
                Log("SparseFluidHandler: emitter point is solid or outside the world, aim somewhere else.");
                return;
            }

            gpu?.Release();
            solver.RemoveCell(position);
            SetDrawn(position, false);
            PlaceMarkerVoxel(position, false);
            solver.InvalidateSolid(position, position);

            emitters.Add(new Emitter
            {
                kind = kind,
                position = position,
                forwardXZ = HorizontalFacing(camDir),
                speed = emitterSpeed,
                tiltDeg = kind == EmitterKind.Sprinkler ? sprinklerTiltDegrees : 0f,
                arcDeg = sprinklerArcDegrees,
                sweepSpeedDeg = sprinklerAngularSpeedDegPerSec,
                nozzleRadius = emitterNozzleRadius,
            });
            SelectEmitter(emitters.Count - 1);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Log($"SparseFluidHandler: placed {kind} at ({position.X}, {position.Y}, {position.Z}).");
        }

        // Switches rain on around the ground below the aim point, or off
        public void SetRain(bool enabled)
        {
            rainEnabled = enabled;
            rainDropBacklog = 0f;
            if (!enabled)
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 aimPoint = camPos + WorldRender.camera.GetDir() * 20f;
            uint hitType = worldData.RayTraversal(aimPoint + new Vector3(0, 200, 0), new Vector3(0, -1, 0), out float hitLength, out Point3 hitVoxel, out Point3 hitNormal);
            rainGroundY = hitType != 0 ? hitVoxel.Y + 1 : (int)aimPoint.Y - 40;
            rainCenter = new Point3((int)aimPoint.X, rainGroundY, (int)aimPoint.Z);
            ticksSinceSeed = 0;
            loggedFirstContact = false;
            ResetVolumeTracking();
            Log($"SparseFluidHandler: rain over ({rainCenter.X}, {rainCenter.Z}) +-{rainHalfSize}, ground at y{rainGroundY}.");
        }

        // Adds this tick's raindrops to the sources: whole drops at the rate set, the fraction carried over to the next tick
        private void AddRaindrops(List<SparseFluidSolver.Source> sources, float dt)
        {
            rainDropBacklog += rainDropsPerSecond * dt;
            int size = Math.Max(1, rainDropSize);
            int topY = Math.Min(rainGroundY + rainHeight, worldData.sizeInVoxels.Y - size - 2);
            Vector3 velocity = new Vector3(0f, -rainDropSpeed, 0f);
            while (rainDropBacklog >= 1f)
            {
                rainDropBacklog -= 1f;
                Point3 corner = new Point3(rainCenter.X + rainRandom.Next(-rainHalfSize, rainHalfSize + 1), topY, rainCenter.Z + rainRandom.Next(-rainHalfSize, rainHalfSize + 1));
                for (int x = 0; x < size; x++)
                    for (int y = 0; y < size; y++)
                        for (int z = 0; z < size; z++)
                            sources.Add(new SparseFluidSolver.Source { cell = corner + new Point3(x, y, z), velocity = velocity });
            }
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
            SelectedEmitter = -1;
            CommitEdits();
        }

        // Erases one emitter's marker and stops its injection, leaving existing fluid in place
        public void RemoveEmitter(int index)
        {
            if (index < 0 || index >= emitters.Count)
                return;
            Emitter emitter = emitters[index];
            WriteCell(emitter.position, false);
            solver.InvalidateSolid(emitter.position, emitter.position);
            emitters.RemoveAt(index);
            if (SelectedEmitter == index)
                SelectedEmitter = -1;
            else if (SelectedEmitter > index)
                SelectedEmitter--;
            CommitEdits();
        }

        // Makes an emitter the selected one (-1 for none), its marker drawn in the highlight colour
        public void SelectEmitter(int index)
        {
            if (index < 0 || index >= emitters.Count)
                index = -1;
            if (index == SelectedEmitter)
                return;
            if (SelectedEmitter >= 0)
                PlaceMarkerVoxel(emitters[SelectedEmitter].position, false);
            SelectedEmitter = index;
            if (index >= 0)
                PlaceMarkerVoxel(emitters[index].position, true);
        }

        // Points an emitter along the camera: its horizontal facing and, within its kind's range, its tilt; a sprinkler restarts its sweep
        public void AimEmitterAlongCamera(int index)
        {
            if (index < 0 || index >= emitters.Count)
                return;
            Emitter emitter = emitters[index];
            Vector3 camDir = WorldRender.camera.GetDir();
            emitter.forwardXZ = HorizontalFacing(camDir);
            float pitch = MathHelper.ToDegrees(MathF.Asin(Math.Clamp(camDir.Y, -1f, 1f)));
            emitter.tiltDeg = Math.Clamp(pitch, MinTilt(emitter.kind), 89f);
            emitter.angleDeg = 0f;
            emitter.sweepDir = 1;
        }

        // Lowest tilt an emitter kind can be given: taps can aim down, sprinklers only spray upwards
        public static float MinTilt(EmitterKind kind) => kind == EmitterKind.Tap ? -89f : 0f;

        // Unit horizontal part of a direction, +Z when it points straight up or down
        private static Vector3 HorizontalFacing(Vector3 direction)
        {
            Vector3 forwardXZ = new Vector3(direction.X, 0f, direction.Z);
            return forwardXZ.LengthSquared() > 1e-6f ? Vector3.Normalize(forwardXZ) : Vector3.UnitZ;
        }

        // Whether an emitter injects this tick, going by its switch, start delay and on/off cycle
        private static bool IsEmitting(Emitter emitter)
        {
            if (!emitter.enabled)
                return false;
            float t = emitter.runSeconds - emitter.startDelaySeconds;
            if (t < 0f)
                return false;
            if (emitter.onSeconds <= 0f)
                return true;
            float cycle = emitter.onSeconds + MathF.Max(emitter.offSeconds, 0f);
            return t % cycle < emitter.onSeconds;
        }

        // Writes an emitter's marker voxel into the world, in the highlight colour when selected
        private void PlaceMarkerVoxel(Point3 worldVoxel, bool selected)
        {
            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, (1u << 15) | (selected ? selectedEmitterTypeRenderIndex : emitterTypeRenderIndex));
            CommitEdits();
        }

        // Advances each emitter's timer and each running sprinkler's sweep, and returns every nozzle cell of the emitters running
        // with its emitter's launch velocity, plus this tick's raindrops
        private List<SparseFluidSolver.Source> BuildSources(float dt)
        {
            List<SparseFluidSolver.Source> sources = new List<SparseFluidSolver.Source>();
            foreach (Emitter emitter in emitters)
            {
                emitter.runSeconds += dt;
                if (!IsEmitting(emitter))
                    continue;

                if (emitter.kind == EmitterKind.Sprinkler)
                {
                    float halfArc = emitter.arcDeg * 0.5f;
                    emitter.angleDeg += emitter.sweepSpeedDeg * dt * emitter.sweepDir;
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
                float tiltRad = MathHelper.ToRadians(emitter.tiltDeg);
                Vector3 launchVelocity = (horizontal * MathF.Cos(tiltRad) + Vector3.Up * MathF.Sin(tiltRad)) * emitter.speed;

                // Nozzle cross-section spans the two axes perpendicular to the launch direction's dominant axis
                Point3 flowAxis = DominantAxisStep(launchVelocity);
                Point3 axisA = flowAxis.X != 0 ? new Point3(0, 1, 0) : new Point3(1, 0, 0);
                Point3 axisB = flowAxis.Z != 0 ? new Point3(0, 1, 0) : new Point3(0, 0, 1);

                // The nozzle sits in front of a tap and on top of a sprinkler, far enough out that its cross-section clears the marker
                int radius = Math.Max(emitter.nozzleRadius, 0);
                Point3 offsetStep = emitter.kind == EmitterKind.Tap ? DominantAxisStep(emitter.forwardXZ) : new Point3(0, 1, 0);
                bool offsetAlongFlow = Math.Abs(offsetStep.X) == Math.Abs(flowAxis.X) && Math.Abs(offsetStep.Y) == Math.Abs(flowAxis.Y);
                Point3 center = emitter.position + offsetStep * (offsetAlongFlow ? 1 : radius + 1);
                for (int a = -radius; a <= radius; a++)
                    for (int b = -radius; b <= radius; b++)
                        sources.Add(new SparseFluidSolver.Source { cell = center + axisA * a + axisB * b, velocity = launchVelocity });
            }
            if (rainEnabled)
                AddRaindrops(sources, dt);
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

        // Frees the GPU simulation's buffers, when the world this handler belongs to is disposed
        public void Dispose()
        {
            gpu?.Dispose();
            gpu = null;
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
            SelectedEmitter = -1;
            rainEnabled = false;

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

        // One bit per cell of every brick slot: tracked and its fill above threshold
        private void BuildVisibleMask(ref uint[] mask, float threshold)
        {
            SparseFluidGrid grid = solver.grid;
            int elements = grid.SlotCount * (SparseFluidGrid.BrickCells / 32);
            if (mask.Length < elements)
                mask = new uint[Math.Max(elements, mask.Length * 2)];
            if (gpuEnabled && gpu != null && gpu.ReadVisibleMask(mask, threshold))
                return;

            for (int element = 0; element < elements; element++)
            {
                uint bits = 0;
                int start = element * 32;
                for (int k = 0; k < 32; k++)
                    if (grid.tracked[start + k] != 0 && grid.fill[start + k] > threshold)
                        bits |= 1u << k;
                mask[element] = bits;
            }
        }

        // One bit per cell of every brick slot: tracked, its fill above threshold, and its center speed above foamSpeed
        private void BuildFastMask(ref uint[] mask, float threshold)
        {
            SparseFluidGrid grid = solver.grid;
            int elements = grid.SlotCount * (SparseFluidGrid.BrickCells / 32);
            if (mask.Length < elements)
                mask = new uint[Math.Max(elements, mask.Length * 2)];
            if (gpuEnabled && gpu != null && gpu.ReadFastMask(mask, threshold, foamSpeed))
                return;

            float fastSquared = foamSpeed * foamSpeed;
            for (int element = 0; element < elements; element++)
            {
                uint bits = 0;
                int start = element * 32;
                for (int k = 0; k < 32; k++)
                    if (grid.tracked[start + k] != 0 && grid.fill[start + k] > threshold && grid.CenterVelocity(grid.CellPosition(start + k)).LengthSquared() > fastSquared)
                        bits |= 1u << k;
                mask[element] = bits;
            }
        }

        // A random shade, from the fast or the calm weights
        private byte PickShade(bool fast)
        {
            float[] weights = fast ? FastShadeWeights : CalmShadeWeights;
            float roll = (float)shadeRandom.NextDouble();
            for (int shade = 0; shade < ShadeCount - 1; shade++)
            {
                roll -= weights[shade];
                if (roll < 0f)
                    return (byte)shade;
            }
            return ShadeCount - 1;
        }

        // Draws cells that rose above the visibility threshold and erases cells that dropped below it or left tracking (with drawHysteresis),
        // going only through the cells whose visibility differs from what is drawn; with colorVariation, also gives a new shade to drawn cells
        // that became fast or calm, and to a random few for the shimmer
        private void ApplyToWorld()
        {
            SparseFluidGrid grid = solver.grid;
            BuildVisibleMask(ref drawMask, SparseFluidSolver.visibilityThreshold + drawHysteresis);
            BuildVisibleMask(ref keepMask, SparseFluidSolver.visibilityThreshold - drawHysteresis);
            if (colorVariation)
                BuildFastMask(ref fastMask, SparseFluidSolver.visibilityThreshold - drawHysteresis);
            bool wrote = false;
            int elementsPerBrick = SparseFluidGrid.BrickCells / 32;

            long nowTimestamp = Stopwatch.GetTimestamp();
            float elapsed = lastApplyTimestamp == 0 ? 0f : MathF.Min((float)((nowTimestamp - lastApplyTimestamp) / (double)Stopwatch.Frequency), 0.25f);
            lastApplyTimestamp = nowTimestamp;
            float shimmerChancePerCell = colorVariation ? MathF.Max(shimmerPerSecond, 0f) * elapsed : 0f;

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
                    uint drawnBits = hasDrawn ? drawn.bits[element] : 0u;
                    uint now = drawMask[slot * elementsPerBrick + element] | (drawnBits & keepMask[slot * elementsPerBrick + element]);
                    uint fastBits = colorVariation ? fastMask[slot * elementsPerBrick + element] : 0u;
                    uint changed = now ^ drawnBits;
                    while (changed != 0)
                    {
                        int bit = System.Numerics.BitOperations.TrailingZeroCount(changed);
                        changed &= changed - 1;
                        bool visible = (now & (1u << bit)) != 0;
                        bool fast = (fastBits & (1u << bit)) != 0;
                        byte shade = colorVariation ? PickShade(fast) : (byte)PlainShade;
                        if (!WriteCell(origin + LocalCell(element * 32 + bit), visible, shade))
                            continue;

                        if (!hasDrawn)
                        {
                            drawn = new DrawnBrick { seen = true };
                            drawnBricks[key] = drawn;
                            hasDrawn = true;
                        }
                        if (visible)
                        {
                            drawn.bits[element] |= 1u << bit;
                            drawn.shade[element * 32 + bit] = shade;
                        }
                        else
                            drawn.bits[element] &= ~(1u << bit);
                        if (visible && fast)
                            drawn.fast[element] |= 1u << bit;
                        else
                            drawn.fast[element] &= ~(1u << bit);
                        wrote = true;
                    }

                    if (!colorVariation || drawnBits == 0)
                        continue;

                    // Cells that stay drawn: a new shade when they became fast or calm, and now and then for one random cell with nothing drawn above it
                    uint stay = now & drawnBits;
                    uint recolor = stay & (fastBits ^ drawn.fast[element]);
                    int stayCount = System.Numerics.BitOperations.PopCount(stay);
                    if (stayCount > 0 && shadeRandom.NextDouble() < stayCount * shimmerChancePerCell)
                    {
                        int pick = shadeRandom.Next(stayCount);
                        uint rest = stay;
                        for (int k = 0; k < pick; k++)
                            rest &= rest - 1;
                        uint picked = rest & (~rest + 1);
                        int pickedLocal = element * 32 + System.Numerics.BitOperations.TrailingZeroCount(picked);
                        if (!IsDrawn(origin + LocalCell(pickedLocal) + new Point3(0, 1, 0)))
                            recolor |= picked;
                    }
                    while (recolor != 0)
                    {
                        int bit = System.Numerics.BitOperations.TrailingZeroCount(recolor);
                        recolor &= recolor - 1;
                        bool fast = (fastBits & (1u << bit)) != 0;
                        byte shade = PickShade(fast);
                        if (fast)
                            drawn.fast[element] |= 1u << bit;
                        else
                            drawn.fast[element] &= ~(1u << bit);
                        if (shade == drawn.shade[element * 32 + bit])
                            continue;
                        RecolorCell(origin + LocalCell(element * 32 + bit), shade);
                        drawn.shade[element * 32 + bit] = shade;
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

        // Stages a fluid voxel in the given shade or air at a world position; returns false when the voxel is outside the world, or when
        // drawing it would overwrite solid terrain
        private bool WriteCell(Point3 worldVoxel, bool visible, int shade = PlainShade)
        {
            if (!IsInsideWorld(worldVoxel))
                return false;
            if (visible && worldData.IsVoxelSolid(worldVoxel))
                return false;

            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            uint type = visible ? (1u << 15) | shadeRenderIndex[shade] : 0u;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);
            return true;
        }

        // Stages a new shade for a cell this handler has drawn
        private void RecolorCell(Point3 worldVoxel, int shade)
        {
            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;
            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, (1u << 15) | shadeRenderIndex[shade]);
        }

        // Takes the current total volume as the ledger's baseline and zeroes the solver's volume counters
        private void ResetVolumeTracking()
        {
            if (!loggingEnabled)
                return;
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
            Console.WriteLine($"SparseFluidHandler: tick {ticksSinceSeed} ({ticksSinceSeed * physicsIntervalMs / 1000f:F2}s), visible cells = {visibleCellCount}, tracked cells = {grid.Count}, liquid cells = {solver.lastLiquidCells}, total speed = {totalSpeed:F3}, pressure iterations = {solver.lastPressureIterations} (residual {solver.lastPressureResidual:E1}), pruned this tick (solid/empty or stagnant/drained) = {solver.lastPrunedBySolid}/{solver.lastPrunedEmptyOrStagnant}/{solver.lastDrained}, emitters = {emitters.Count}{bandYInfo}");

            float expectedChange = solver.volumeInjected + solver.volumeClampCreated - solver.volumeClampedAway - solver.volumePrunedSolid - solver.volumePrunedEmpty - solver.volumePrunedStagnant - solver.volumeDrained;
            double unaccounted = (totalVolume - volumeAtLastLog) - expectedChange;
            Console.WriteLine($"    volume = {totalVolume:F2}, since last log: injected +{solver.volumeInjected:F2}, clamp lost -{solver.volumeClampedAway:F3}, clamp created +{solver.volumeClampCreated:F3}, pruned solid -{solver.volumePrunedSolid:F3} / empty -{solver.volumePrunedEmpty:F3} / stagnant -{solver.volumePrunedStagnant:F3}, drained -{solver.volumeDrained:F3}, unaccounted {unaccounted:F3}");

            volumeAtLastLog = totalVolume;
            solver.ClearVolumeAccumulators();
        }
    }
}
