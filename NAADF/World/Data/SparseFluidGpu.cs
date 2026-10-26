using Microsoft.Xna.Framework.Graphics;
using NAADF.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NAADF.World.Data
{
    /*
    * Runs SparseFluidSolver's tick as compute shaders (shaders/fluid/sparseFluidSim.fx)
    *
    * The per-cell arrays exist both in SparseFluidGrid and in GPU buffers, and only the side that last wrote one is current
    * A tick leaves everything on the GPU, the grid is brought up to date on request (Download) or handed the state back (Release)
    * What stays on the CPU every tick: the brick layout, which works from a per-brick summary read back at the end of the tick,
    * and the solid lookups, which are uploaded as a mask
    * With selfCheck on, every stage is followed by its CPU version on the same input, the CPU result is the one kept, and the differences are logged
    */
    public class SparseFluidGpu : IDisposable
    {
        [Flags]
        private enum Arrays { None = 0, Faces = 1, Pressure = 2, Fill = 4, State = 8 }

        // One per-cell array of the grid together with its GPU buffer
        private class Mirror
        {
            public StructuredBuffer buffer;
            public bool cpuCurrent = true, gpuCurrent = false;
            public bool neverUploaded = true;

            // Marks the grid's array as the only current copy
            public void WrittenOnCpu()
            {
                cpuCurrent = true;
                gpuCurrent = false;
            }

            // Marks the GPU buffer as the only current copy
            public void WrittenOnGpu()
            {
                gpuCurrent = true;
                cpuCurrent = false;
            }
        }

        // Timing and self-check figures of one stage since its last log line
        private class StageStatistics
        {
            public string name;
            public float tolerance;
            public bool comparesState;
            public int ticks;
            public double gpuMs, cpuMs, kernelMs;
            public float maxDifference;
            public long compared, overTolerance;
            public long gpuIterations, cpuIterations;
            public int unconvergedTicks, extraBatches;
            public long stateMismatches;
        }

        private readonly SparseFluidSolver solver;
        private readonly SparseFluidGrid grid;
        private readonly GraphicsDevice device;
        private readonly Effect effect;

        private int cellCapacity = 0;
        private int slots, cells; // of the stage being run

        // Slots the buffers that keep their contents between ticks may hold data for, from there on they are zero
        private int gpuSlots = 0;
        private static readonly float[] zeroFloats = new float[65536];
        private static readonly uint[] zeroUints = new uint[65536];

        // Face velocities, pressure, fill fractions, and one state word per cell (SparseFluidGrid.PackState)
        private readonly Mirror[] face = { new Mirror(), new Mirror(), new Mirror() };
        private readonly Mirror pressure = new Mirror(), fill = new Mirror(), state = new Mirror();
        private readonly List<Mirror> mirrors = new List<Mirror>();

        private StructuredBuffer brickInfoBuffer, brickNeighborBuffer, solidBuffer;
        private bool layoutCurrent = false, solidCurrent = false;
        private StructuredBuffer[] faceScratchBuffer = new StructuredBuffer[3];
        private StructuredBuffer diagBuffer, rBuffer, sBuffer, qBuffer;
        private StructuredBuffer brickSumBuffer, brickMaxBuffer, scalarsBuffer;
        private StructuredBuffer fillOutTotalBuffer, fillOutScaledBuffer, fillInTotalBuffer, fillRawBuffer, fillStatusBuffer;
        private StructuredBuffer ledgerBuffer, brickLedgerBuffer, brickSummaryBuffer, visibleMaskBuffer;
        private StructuredBuffer sourceBuffer, previousSourceBuffer;
        private int sourceCapacity = 0;

        private int[] brickInfoUpload;
        private uint[] byteUpload;
        private uint[] stateUpload, stateReadback;
        private float[][] faceReadback = new float[3][];
        private float[] pressureReadback, fillReadback;
        private uint[] summaryReadback;
        private int[] sourceUpload = new int[0];
        private readonly List<int> previousSourceCells = new List<int>();
        private int sourceCount;
        private readonly float[] scalarsReadback = new float[8];

        // Same slots as the shader's ledger
        private const int LedgerInjected = 0, LedgerClampedAway = 1, LedgerClampCreated = 2, LedgerPrunedSolid = 3, LedgerPrunedEmpty = 4,
            LedgerPrunedStagnant = 5, LedgerDrained = 6, LedgerCountSolid = 7, LedgerCountEmpty = 8, LedgerCountDrained = 9, LedgerSize = 10;
        private readonly float[] ledgerReadback = new float[LedgerSize];
        private readonly float[] ledgerZero = new float[LedgerSize];

        // Bits of a state word each stage is answerable for, for the self-check
        private const uint StateTracked = SparseFluidGrid.StateTracked, StateLiquid = 2, StateSource = 4, StateGained = 8;
        private const uint StatePresent = SparseFluidGrid.StatePresent * 7, StateStagnant = 0xFFFF0000;

        private static readonly string[] FaceNames = { "faceX", "faceY", "faceZ" };
        private static readonly string[] FaceRwNames = { "faceRwX", "faceRwY", "faceRwZ" };

        public bool selfCheck = false;

        // Prints the per-stage lines, with selfCheck or measureKernels on they are printed regardless
        public bool logging = false;

        // Waits for the GPU before and after each stage's kernels, so the time they take can be logged on its own; slows the tick down
        public bool measureKernels = false;

        // The pressure solve dispatches conjugate gradient iterations in batches, iterations after convergence do nothing
        // The first batch is sized from what the previous tick needed, and further batches follow until the solve has converged
        private const int minPressureIterations = 16, maxPressureIterations = 300;
        private int pressureIterations = 32;

        private const int logEveryTicks = 125;
        private readonly StageStatistics gravityStatistics = new StageStatistics { name = "Gravity", tolerance = 1e-6f };
        private readonly StageStatistics advectStatistics = new StageStatistics { name = "AdvectVelocity", tolerance = 1e-4f, comparesState = true };
        private readonly StageStatistics sourceStatistics = new StageStatistics { name = "Sources", tolerance = 1e-6f, comparesState = true };
        private readonly StageStatistics projectStatistics = new StageStatistics { name = "Project", tolerance = 1e-3f, comparesState = true };
        private readonly StageStatistics extrapolateStatistics = new StageStatistics { name = "Extrapolate", tolerance = 1e-6f, comparesState = true };
        private readonly StageStatistics fillStatistics = new StageStatistics { name = "AdvectFill", tolerance = 1e-6f, comparesState = true };
        private readonly StageStatistics pruneStatistics = new StageStatistics { name = "Prune", tolerance = 1e-6f, comparesState = true };
        private readonly StageStatistics dampingStatistics = new StageStatistics { name = "Damping", tolerance = 1e-6f };

        // Drives the given solver's grid on the GPU
        // effect is the compiled shaders/fluid/sparseFluidSim
        public SparseFluidGpu(SparseFluidSolver solver, GraphicsDevice device, Effect effect)
        {
            this.solver = solver;
            this.grid = solver.grid;
            this.device = device;
            this.effect = effect;

            mirrors.AddRange(face);
            mirrors.AddRange(new[] { pressure, fill, state });
        }

        // ---- Buffers ----

        // Creates a buffer the kernels can read and write
        private StructuredBuffer CreateBuffer(Type type, int count)
        {
            return new StructuredBuffer(device, type, count, BufferUsage.None, ShaderAccess.ReadWrite);
        }

        // Creates a buffer the kernels only read, filled from the CPU
        private StructuredBuffer CreateUploadBuffer(Type type, int count)
        {
            return new StructuredBuffer(device, type, count, BufferUsage.None, ShaderAccess.Read, StructuredBufferType.Basic, -1, true);
        }

        // Makes the buffers as large as the grid's arrays, what was only current on the GPU is brought back to the grid first
        private void EnsureCapacity()
        {
            int gridCells = grid.tracked.Length;
            if (gridCells <= cellCapacity)
                return;

            Download(true);
            DisposeBuffers();
            cellCapacity = gridCells;
            int slotCapacity = gridCells / SparseFluidGrid.BrickCells;

            brickInfoBuffer = CreateUploadBuffer(typeof(int), slotCapacity * 4);
            brickNeighborBuffer = CreateUploadBuffer(typeof(int), slotCapacity * 27);
            solidBuffer = CreateUploadBuffer(typeof(uint), gridCells / 4);
            layoutCurrent = solidCurrent = false;

            for (int axis = 0; axis < 3; axis++)
            {
                face[axis].buffer = CreateBuffer(typeof(float), gridCells);
                faceScratchBuffer[axis] = CreateBuffer(typeof(float), gridCells);
                ZeroCells(faceScratchBuffer[axis], zeroFloats, 0, gridCells);
                faceReadback[axis] = new float[gridCells];
            }
            pressure.buffer = CreateBuffer(typeof(float), gridCells);
            fill.buffer = CreateBuffer(typeof(float), gridCells);
            state.buffer = CreateBuffer(typeof(uint), gridCells);
            foreach (Mirror mirror in mirrors)
            {
                mirror.gpuCurrent = false;
                mirror.neverUploaded = true;
            }
            gpuSlots = 0;

            diagBuffer = CreateBuffer(typeof(float), gridCells);
            rBuffer = CreateBuffer(typeof(float), gridCells);
            sBuffer = CreateBuffer(typeof(float), gridCells);
            qBuffer = CreateBuffer(typeof(float), gridCells);
            brickSumBuffer = CreateBuffer(typeof(float), slotCapacity);
            brickMaxBuffer = CreateBuffer(typeof(float), slotCapacity);
            scalarsBuffer = CreateBuffer(typeof(float), scalarsReadback.Length);
            fillOutTotalBuffer = CreateBuffer(typeof(float), gridCells);
            fillOutScaledBuffer = CreateBuffer(typeof(float), gridCells);
            fillInTotalBuffer = CreateBuffer(typeof(float), gridCells);
            fillRawBuffer = CreateBuffer(typeof(float), gridCells);
            fillStatusBuffer = CreateBuffer(typeof(uint), gridCells);
            ledgerBuffer = CreateBuffer(typeof(float), LedgerSize);
            brickLedgerBuffer = CreateBuffer(typeof(float), slotCapacity * LedgerSize);
            brickSummaryBuffer = CreateBuffer(typeof(uint), slotCapacity);
            visibleMaskBuffer = CreateBuffer(typeof(uint), gridCells / 32);

            brickInfoUpload = new int[slotCapacity * 4];
            byteUpload = new uint[gridCells / 4];
            stateUpload = new uint[gridCells];
            stateReadback = new uint[gridCells];
            pressureReadback = new float[gridCells];
            fillReadback = new float[gridCells];
            summaryReadback = new uint[slotCapacity];
        }

        // Writes zeros into count cells of a buffer from cell start on
        private static void ZeroCells<T>(StructuredBuffer buffer, T[] zeros, int start, int count) where T : struct
        {
            for (int done = 0; done < count; done += zeros.Length)
                buffer.SetData((start + done) * 4, zeros, 0, Math.Min(zeros.Length, count - done), 0);
        }

        // Zeroes the slots from firstSlot up to gpuSlots in every buffer that keeps its contents between ticks,
        // so that slots the grid hands out again after it was cleared don't start with the previous cells
        private void ZeroSlotsFrom(int firstSlot)
        {
            int start = firstSlot * SparseFluidGrid.BrickCells;
            int count = gpuSlots * SparseFluidGrid.BrickCells - start;
            for (int axis = 0; axis < 3; axis++)
            {
                ZeroCells(face[axis].buffer, zeroFloats, start, count);
                ZeroCells(faceScratchBuffer[axis], zeroFloats, start, count);
            }
            ZeroCells(pressure.buffer, zeroFloats, start, count);
            ZeroCells(fill.buffer, zeroFloats, start, count);
            ZeroCells(state.buffer, zeroUints, start, count);
        }

        // The first upload of an array covers the whole buffer, so slots that come into use later start out as they are on the CPU
        private int UploadCount(Mirror mirror)
        {
            int count = mirror.neverUploaded ? cellCapacity : cells;
            mirror.neverUploaded = false;
            mirror.gpuCurrent = true;
            return count;
        }

        // Number of cells to read back: every cell of the grid's slots, as far as the buffers reach
        private int DownloadCount => Math.Min(grid.SlotCount * SparseFluidGrid.BrickCells, cellCapacity);

        // Uploads one float array unless the GPU copy is already current
        private void ToGpu(Mirror mirror, float[] source)
        {
            if (!mirror.gpuCurrent)
                mirror.buffer.SetData(source, 0, UploadCount(mirror));
        }

        // Reads one float array back into the grid unless the grid's copy is already current
        private void ToCpu(Mirror mirror, float[] target)
        {
            if (mirror.cpuCurrent)
                return;

            mirror.buffer.GetData(target, 0, DownloadCount);
            mirror.cpuCurrent = true;
        }

        // Brings the given arrays up to date on the GPU; the state words are packed from the grid on the way
        private void ToGpu(Arrays arrays)
        {
            if ((arrays & Arrays.Faces) != 0)
                for (int axis = 0; axis < 3; axis++)
                    ToGpu(face[axis], grid.face[axis]);
            if ((arrays & Arrays.Pressure) != 0)
                ToGpu(pressure, grid.pressure);
            if ((arrays & Arrays.Fill) != 0)
                ToGpu(fill, grid.fill);
            if ((arrays & Arrays.State) != 0 && !state.gpuCurrent)
            {
                int count = UploadCount(state);
                grid.PackState(stateUpload, count);
                state.buffer.SetData(stateUpload, 0, count);

                // The cells the grid holds as sources are the ones the next source stage has to release
                previousSourceCells.Clear();
                for (int i = 0; i < count; i++)
                    if ((stateUpload[i] & StateSource) != 0)
                        previousSourceCells.Add(i);
            }
        }

        // Brings the given arrays up to date in the grid; the state words are unpacked and the solver's cell lists rebuilt on the way
        private void ToCpu(Arrays arrays)
        {
            if ((arrays & Arrays.Faces) != 0)
                for (int axis = 0; axis < 3; axis++)
                    ToCpu(face[axis], grid.face[axis]);
            if ((arrays & Arrays.Pressure) != 0)
                ToCpu(pressure, grid.pressure);
            if ((arrays & Arrays.Fill) != 0)
                ToCpu(fill, grid.fill);
            if ((arrays & Arrays.State) != 0 && !state.cpuCurrent)
            {
                int count = DownloadCount;
                state.buffer.GetData(stateReadback, 0, count);
                grid.UnpackState(stateReadback, count);
                solver.RebuildCellLists();
                state.cpuCurrent = true;
            }
        }

        // Marks the given arrays as changed in the grid
        private void WrittenOnCpu(Arrays arrays)
        {
            if ((arrays & Arrays.Faces) != 0)
                for (int axis = 0; axis < 3; axis++)
                    face[axis].WrittenOnCpu();
            if ((arrays & Arrays.Pressure) != 0) pressure.WrittenOnCpu();
            if ((arrays & Arrays.Fill) != 0) fill.WrittenOnCpu();
            if ((arrays & Arrays.State) != 0) state.WrittenOnCpu();
        }

        // Marks the given arrays as changed on the GPU
        private void WrittenOnGpu(Arrays arrays)
        {
            if ((arrays & Arrays.Faces) != 0)
                for (int axis = 0; axis < 3; axis++)
                    face[axis].WrittenOnGpu();
            if ((arrays & Arrays.Pressure) != 0) pressure.WrittenOnGpu();
            if ((arrays & Arrays.Fill) != 0) fill.WrittenOnGpu();
            if ((arrays & Arrays.State) != 0) state.WrittenOnGpu();
        }

        // Brick layout, read by the kernels that look at neighboring cells
        private void LayoutToGpu()
        {
            if (layoutCurrent)
                return;

            grid.FillBrickInfo(brickInfoUpload);
            brickInfoBuffer.SetData(brickInfoUpload, 0, slots * 4);
            brickNeighborBuffer.SetData(grid.NeighborTable, 0, slots * 27);
            layoutCurrent = true;
        }

        // Solid and drain lookups for every cell of every brick, one byte per cell uploaded four to an element
        private void SolidToGpu()
        {
            if (solidCurrent)
                return;

            solver.FillSolid();
            Buffer.BlockCopy(grid.solid, 0, byteUpload, 0, cells);
            solidBuffer.SetData(byteUpload, 0, cells / 4);
            solidCurrent = true;
        }

        // Brings the grid's arrays up to date for code that reads them directly. Without velocities, face velocities and pressure are left where they are
        public void Download(bool velocities)
        {
            if (cellCapacity == 0)
                return;

            ToCpu(Arrays.State | Arrays.Fill);
            if (velocities)
                ToCpu(Arrays.Faces | Arrays.Pressure);
        }

        // Fills in one bit per cell (32 cells per element, in cell index order) saying whether the cell is tracked and its fill above threshold
        // Returns false without doing so when the grid holds the current cells, which the caller can then read itself
        public bool ReadVisibleMask(uint[] target, float threshold)
        {
            if (cellCapacity == 0 || state.cpuCurrent || fill.cpuCurrent)
                return false;

            BindBuffers();
            effect.Parameters["visibilityThreshold"].SetValue(threshold);
            int elements = slots * (SparseFluidGrid.BrickCells / 32);
            Dispatch("BuildVisibleMask", (elements + 63) / 64);
            visibleMaskBuffer.GetData(target, 0, elements);
            return true;
        }

        // Hands the whole state back to the grid, so that it may be changed there
        public void Release()
        {
            Download(true);
            foreach (Mirror mirror in mirrors)
                mirror.gpuCurrent = false;
            layoutCurrent = solidCurrent = false;
        }

        // Binds every buffer to its shader name, and the current slot count
        private void BindBuffers()
        {
            effect.Parameters["brickInfo"].SetValue(brickInfoBuffer);
            effect.Parameters["brickNeighbor"].SetValue(brickNeighborBuffer);
            effect.Parameters["solidMask"].SetValue(solidBuffer);
            for (int axis = 0; axis < 3; axis++)
            {
                effect.Parameters[FaceNames[axis]].SetValue(face[axis].buffer);
                effect.Parameters[FaceRwNames[axis]].SetValue(face[axis].buffer);
            }
            effect.Parameters["pressure"].SetValue(pressure.buffer);
            effect.Parameters["fill"].SetValue(fill.buffer);
            effect.Parameters["fillRw"].SetValue(fill.buffer);
            effect.Parameters["cellState"].SetValue(state.buffer);
            effect.Parameters["cellStateRw"].SetValue(state.buffer);

            effect.Parameters["pcgDiag"].SetValue(diagBuffer);
            effect.Parameters["pcgR"].SetValue(rBuffer);
            effect.Parameters["pcgS"].SetValue(sBuffer);
            effect.Parameters["pcgQ"].SetValue(qBuffer);
            effect.Parameters["brickSum"].SetValue(brickSumBuffer);
            effect.Parameters["brickMax"].SetValue(brickMaxBuffer);
            effect.Parameters["scalars"].SetValue(scalarsBuffer);
            effect.Parameters["fillOutTotal"].SetValue(fillOutTotalBuffer);
            effect.Parameters["fillOutScaled"].SetValue(fillOutScaledBuffer);
            effect.Parameters["fillInTotal"].SetValue(fillInTotalBuffer);
            effect.Parameters["fillRaw"].SetValue(fillRawBuffer);
            effect.Parameters["fillStatus"].SetValue(fillStatusBuffer);
            effect.Parameters["ledger"].SetValue(ledgerBuffer);
            effect.Parameters["brickLedger"].SetValue(brickLedgerBuffer);
            effect.Parameters["brickSummary"].SetValue(brickSummaryBuffer);
            effect.Parameters["visibleMask"].SetValue(visibleMaskBuffer);
            effect.Parameters["slotCount"].SetValue(slots);
        }

        // Passes the solver's settings for this tick to the shader
        private void SetTickConstants(float dt)
        {
            float damping = solver.velocityDampingPerSecond;
            effect.Parameters["dt"].SetValue(dt);
            effect.Parameters["gravityStep"].SetValue(solver.gravityStrength * dt);
            effect.Parameters["velocityScale"].SetValue(damping >= 1f ? 1f : MathF.Pow(Math.Clamp(damping, 0f, 1f), dt));
            effect.Parameters["maxSpeed"].SetValue(SparseFluidSolver.cflMaxFraction / dt);
            effect.Parameters["compactionRate"].SetValue(solver.compactionRate);
            effect.Parameters["liquidThreshold"].SetValue(SparseFluidSolver.liquidThreshold);
            effect.Parameters["visibilityThreshold"].SetValue(SparseFluidSolver.visibilityThreshold);
            effect.Parameters["minInterfaceTheta"].SetValue(SparseFluidSolver.minInterfaceTheta);
            effect.Parameters["pressureTolerance"].SetValue(SparseFluidSolver.pressureTolerance);
            effect.Parameters["minFillEpsilon"].SetValue(SparseFluidSolver.minFillEpsilon);
            effect.Parameters["stagnantLimit"].SetValue(solver.stagnantTicksBeforeDestroy);
            effect.Parameters["keepSurfaceFilm"].SetValue(solver.keepSurfaceFilm ? 1 : 0);
        }

        // Runs one kernel of the effect with the given number of compute groups
        private void Dispatch(string pass, int groups)
        {
            effect.Techniques[0].Passes[pass].ApplyCompute();
            device.DispatchCompute(groups, 1, 1);
        }

        // Start of a kernel timing: with measureKernels on, first waits until the GPU has finished all earlier work
        private long KernelsStart()
        {
            if (measureKernels)
                scalarsBuffer.GetData(scalarsReadback, 0, 1);
            return Stopwatch.GetTimestamp();
        }

        // End of a kernel timing: with measureKernels on, waits for the kernels to finish and adds their time to the stage
        private void KernelsEnd(StageStatistics statistics, long start)
        {
            if (!measureKernels)
                return;
            scalarsBuffer.GetData(scalarsReadback, 0, 1);
            statistics.kernelMs += (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }

        // ---- Tick ----

        // Replacement for SparseFluidSolver.Step
        public void Step(float dt, IReadOnlyList<SparseFluidSolver.Source> sources)
        {
            if (grid.SlotCount == 0)
            {
                Release();
                solver.Step(dt, sources);
                return;
            }

            // While the cells are on the GPU, the layout goes by the summary the previous tick ended with
            long start = Stopwatch.GetTimestamp();
            if (state.cpuCurrent)
                solver.BeginTick();
            else
                solver.UpdateLayout();
            AfterLayoutChange();
            SetTickConstants(dt);
            ledgerBuffer.SetData(ledgerZero, 0, LedgerSize);
            start = AddStageTime(0, start);

            if (solver.enableGravity)
                Stage(gravityStatistics, Arrays.State, Arrays.Faces, 0,
                    () => Dispatch("ApplyGravity", slots),
                    () => solver.ApplyGravity(dt));
            start = AddStageTime(1, start);

            Stage(advectStatistics, Arrays.None, Arrays.Faces | Arrays.State, StatePresent, AdvectVelocityKernels, () => solver.AdvectVelocity(dt));
            start = AddStageTime(2, start);

            if (PrepareSources(sources))
                Stage(sourceStatistics, Arrays.None, Arrays.Faces | Arrays.Fill | Arrays.State, StateTracked | StateSource | StatePresent | StateStagnant,
                    SourceKernels,
                    () => solver.ApplySources(sources));
            start = AddStageTime(3, start);

            Stage(projectStatistics, Arrays.Fill, Arrays.Faces | Arrays.Pressure | Arrays.State, StateLiquid, ProjectKernels,
                () =>
                {
                    solver.Project(dt);
                    projectStatistics.cpuIterations += solver.lastPressureIterations;
                });
            start = AddStageTime(4, start);

            Stage(extrapolateStatistics, Arrays.Fill, Arrays.Faces | Arrays.State, StatePresent,
                () => Dispatch("ExtrapolateIntoSupportedAir", slots),
                () => solver.ExtrapolateIntoSupportedAir());
            start = AddStageTime(5, start);

            Stage(fillStatistics, Arrays.Faces, Arrays.Fill | Arrays.State, StateTracked | StateGained,
                () =>
                {
                    Dispatch("FillOutflow", slots);
                    Dispatch("FillInflow", slots);
                    Dispatch("FillApply", slots);
                    Dispatch("FinishFill", slots);
                    SumBrickLedger(LedgerClampedAway, 2);
                },
                () => solver.AdvectFill(dt));
            start = AddStageTime(6, start);

            Stage(pruneStatistics, Arrays.None, Arrays.Fill | Arrays.State, StateTracked | StateStagnant,
                () =>
                {
                    Dispatch("PruneStaleCells", slots);
                    SumBrickLedger(LedgerPrunedSolid, LedgerSize - LedgerPrunedSolid);
                },
                () => solver.PruneStaleCells());
            start = AddStageTime(7, start);

            Stage(dampingStatistics, Arrays.None, Arrays.Faces, 0,
                () => Dispatch("DampAndClampVelocity", slots),
                () =>
                {
                    solver.ApplyDamping(dt);
                    solver.ClampVelocity(dt);
                });
            start = AddStageTime(8, start);

            if (!selfCheck)
                ReadTickSummary();
            AddStageTime(0, start);
        }

        // Adds the time since start to a stage's total in the solver and returns the current timestamp
        private long AddStageTime(int stage, long start)
        {
            long now = Stopwatch.GetTimestamp();
            solver.stageMs[stage] += (now - start) * 1000.0 / Stopwatch.Frequency;
            return now;
        }

        // Bricks may have been added or freed on the CPU
        private void AfterLayoutChange()
        {
            EnsureCapacity();
            slots = grid.SlotCount;
            cells = slots * SparseFluidGrid.BrickCells;
            layoutCurrent = solidCurrent = false;

            // Fewer slots than before only happens when the grid was cleared
            if (slots < gpuSlots)
                ZeroSlotsFrom(slots);
            gpuSlots = slots;
        }

        // Runs one stage: its kernels, and with the self-check on its CPU version afterwards, which is then the result kept
        // reads/writes are the arrays the stage needs and changes; stateMask the state bits it is answerable for
        private void Stage(StageStatistics statistics, Arrays reads, Arrays writes, uint stateMask, Action kernels, Action cpuVersion)
        {
            Stopwatch stopwatch = Stopwatch.StartNew();
            Arrays all = reads | writes;
            if (selfCheck)
                ToCpu(all);
            ToGpu(all);
            SolidToGpu();
            LayoutToGpu();
            BindBuffers();

            long kernelsStart = KernelsStart();
            kernels();
            KernelsEnd(statistics, kernelsStart);

            if (!selfCheck)
            {
                WrittenOnGpu(writes);
                statistics.gpuMs += stopwatch.Elapsed.TotalMilliseconds;
            }
            else
            {
                if ((writes & Arrays.Faces) != 0)
                    for (int axis = 0; axis < 3; axis++)
                        face[axis].buffer.GetData(faceReadback[axis], 0, cells);
                if ((writes & Arrays.Pressure) != 0)
                    pressure.buffer.GetData(pressureReadback, 0, cells);
                if ((writes & Arrays.Fill) != 0)
                    fill.buffer.GetData(fillReadback, 0, cells);
                if ((writes & Arrays.State) != 0)
                    state.buffer.GetData(stateReadback, 0, cells);
                statistics.gpuMs += stopwatch.Elapsed.TotalMilliseconds;

                stopwatch.Restart();
                cpuVersion();
                statistics.cpuMs += stopwatch.Elapsed.TotalMilliseconds;

                if ((writes & Arrays.Faces) != 0)
                    for (int axis = 0; axis < 3; axis++)
                        Compare(statistics, grid.face[axis], faceReadback[axis], cells);
                if ((writes & Arrays.Pressure) != 0)
                    Compare(statistics, grid.pressure, pressureReadback, cells);
                if ((writes & Arrays.Fill) != 0)
                    Compare(statistics, grid.fill, fillReadback, cells);
                if ((writes & Arrays.State) != 0)
                    CompareState(statistics, stateMask);

                WrittenOnCpu(writes);
                AfterLayoutChange();
            }
            LogStatistics(statistics);
        }

        // Folds one array pair into a stage's self-check figures
        private static void Compare(StageStatistics statistics, float[] cpu, float[] gpu, int count)
        {
            for (int i = 0; i < count; i++)
            {
                float difference = MathF.Abs(cpu[i] - gpu[i]);
                if (!(difference <= statistics.maxDifference))
                    statistics.maxDifference = difference;
                if (!(difference <= statistics.tolerance))
                    statistics.overTolerance++;
            }
            statistics.compared += count;
        }

        // Counts the cells whose state read back from the GPU differs from the grid's in the given bits
        // a cell tracked on neither side only has its present faces compared
        private void CompareState(StageStatistics statistics, uint mask)
        {
            grid.PackState(stateUpload, cells);
            for (int i = 0; i < cells; i++)
            {
                uint cpu = stateUpload[i], gpu = stateReadback[i];
                uint difference = (cpu ^ gpu) & (mask | StateTracked);
                if (((cpu | gpu) & StateTracked) == 0)
                    difference &= StatePresent;
                if (difference != 0)
                    statistics.stateMismatches++;
            }
        }

        // Advects the face velocities into the scratch buffers, which then become the face buffers
        private void AdvectVelocityKernels()
        {
            for (int axis = 0; axis < 3; axis++)
                effect.Parameters[FaceRwNames[axis]].SetValue(faceScratchBuffer[axis]);
            Dispatch("AdvectVelocity", slots);

            for (int axis = 0; axis < 3; axis++)
                (face[axis].buffer, faceScratchBuffer[axis]) = (faceScratchBuffer[axis], face[axis].buffer);
        }

        // Makes sure the bricks the sources touch exist and lists the cells for the source kernel. false when there is nothing to do
        private bool PrepareSources(IReadOnlyList<SparseFluidSolver.Source> sources)
        {
            sourceCount = 0;
            int given = sources?.Count ?? 0;
            if (sourceUpload.Length < given * 8)
                sourceUpload = new int[given * 8];

            for (int i = 0; i < given; i++)
            {
                SparseFluidSolver.Source source = sources[i];
                if (solver.IsSolid(source.cell))
                    continue;

                int entry = sourceCount * 8;
                sourceUpload[entry] = grid.EnsureCellIndex(source.cell);
                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                    sourceUpload[entry + 1 + axis] = grid.EnsureCellIndex(source.cell - offset);
                    grid.EnsureCellIndex(source.cell + offset);
                }
                sourceUpload[entry + 4] = BitConverter.SingleToInt32Bits(source.velocity.X);
                sourceUpload[entry + 5] = BitConverter.SingleToInt32Bits(source.velocity.Y);
                sourceUpload[entry + 6] = BitConverter.SingleToInt32Bits(source.velocity.Z);
                sourceCount++;
            }

            if (sourceCount == 0 && previousSourceCells.Count == 0)
                return false;
            if (given > 0)
                AfterLayoutChange();
            return true;
        }

        // Uploads this tick's source cells and last tick's, runs the source kernel, and remembers this tick's cells for the next one
        private void SourceKernels()
        {
            int needed = Math.Max(sourceCount, previousSourceCells.Count);
            if (sourceBuffer == null || needed > sourceCapacity)
            {
                sourceBuffer?.Dispose();
                previousSourceBuffer?.Dispose();
                sourceCapacity = Math.Max(needed, Math.Max(64, sourceCapacity * 2));
                sourceBuffer = CreateUploadBuffer(typeof(int), sourceCapacity * 8);
                previousSourceBuffer = CreateUploadBuffer(typeof(int), sourceCapacity);
            }

            if (sourceCount > 0)
                sourceBuffer.SetData(sourceUpload, 0, sourceCount * 8);
            if (previousSourceCells.Count > 0)
                previousSourceBuffer.SetData(previousSourceCells.ToArray(), 0, previousSourceCells.Count);

            effect.Parameters["sourceCells"].SetValue(sourceBuffer);
            effect.Parameters["previousSourceCells"].SetValue(previousSourceBuffer);
            effect.Parameters["sourceCount"].SetValue(sourceCount);
            effect.Parameters["previousSourceCount"].SetValue(previousSourceCells.Count);
            Dispatch("ApplySources", 1);

            previousSourceCells.Clear();
            for (int i = 0; i < sourceCount; i++)
                previousSourceCells.Add(sourceUpload[i * 8]);
        }

        // Classifies the liquid cells, solves for pressure in batches of iterations until converged, and applies the pressure to the faces
        private void ProjectKernels()
        {
            Dispatch("ClassifyLiquid", slots);
            Dispatch("ZeroSolidFaces", slots);
            Dispatch("BuildSystem", slots);
            Dispatch("FinalizeStart", 1);

            // The starting residual decides whether any iterations are needed at all
            scalarsBuffer.GetData(scalarsReadback, 0, scalarsReadback.Length);
            bool converged = scalarsReadback[4] != 0f;
            int dispatched = 0;
            while (!converged && dispatched < maxPressureIterations)
            {
                if (dispatched > 0)
                    projectStatistics.extraBatches++;

                int batch = Math.Min(pressureIterations, maxPressureIterations - dispatched);
                for (int iteration = 0; iteration < batch; iteration++)
                {
                    Dispatch("ApplyA", slots);
                    Dispatch("FinalizeAlpha", 1);
                    Dispatch("UpdateSolution", slots);
                    Dispatch("FinalizeBeta", 1);
                    Dispatch("UpdateDirection", slots);
                }
                dispatched += batch;

                scalarsBuffer.GetData(scalarsReadback, 0, scalarsReadback.Length);
                converged = scalarsReadback[4] != 0f;
            }
            Dispatch("SubtractPressureGradient", slots);

            int gpuIterations = (int)scalarsReadback[5];
            projectStatistics.gpuIterations += gpuIterations;
            if (!converged)
                projectStatistics.unconvergedTicks++;
            pressureIterations = Math.Clamp(gpuIterations + gpuIterations / 2 + 8, minPressureIterations, maxPressureIterations);

            if (!selfCheck)
            {
                solver.lastPressureIterations = gpuIterations;
                solver.lastPressureResidual = scalarsReadback[3];
            }
        }

        // Adds the per-brick parts of a range of ledger slots to the tick's ledger
        private void SumBrickLedger(int first, int count)
        {
            effect.Parameters["ledgerFirst"].SetValue(first);
            effect.Parameters["ledgerCount"].SetValue(count);
            Dispatch("SumBrickLedger", 1);
        }

        // What the CPU needs to know about a tick that ran on the GPU. per-brick figures for the layout, and the volume bookkeeping
        private void ReadTickSummary()
        {
            BindBuffers();
            Dispatch("SummarizeBricks", (slots + 63) / 64);
            brickSummaryBuffer.GetData(summaryReadback, 0, slots);
            ledgerBuffer.GetData(ledgerReadback, 0, LedgerSize);

            grid.SetBrickSummary(summaryReadback, slots);
            int liquidCells = 0;
            for (int slot = 0; slot < slots; slot++)
                if (grid.IsSlotAllocated(slot))
                    liquidCells += (int)((summaryReadback[slot] >> 18) & 0x3FF);
            solver.lastLiquidCells = liquidCells;

            solver.volumeInjected += ledgerReadback[LedgerInjected];
            solver.volumeClampedAway += ledgerReadback[LedgerClampedAway];
            solver.volumeClampCreated += ledgerReadback[LedgerClampCreated];
            solver.volumePrunedSolid += ledgerReadback[LedgerPrunedSolid];
            solver.volumePrunedEmpty += ledgerReadback[LedgerPrunedEmpty];
            solver.volumePrunedStagnant += ledgerReadback[LedgerPrunedStagnant];
            solver.volumeDrained += ledgerReadback[LedgerDrained];
            solver.lastPrunedBySolid = (int)ledgerReadback[LedgerCountSolid];
            solver.lastPrunedByEmpty = (int)ledgerReadback[LedgerCountEmpty];
            solver.lastDrained = (int)ledgerReadback[LedgerCountDrained];
        }

        // Counts a tick for a stage, and every logEveryTicks ticks prints and resets its timing and self-check figures
        private void LogStatistics(StageStatistics statistics)
        {
            if (++statistics.ticks < logEveryTicks)
                return;

            if (logging || selfCheck || measureKernels)
                PrintStatistics(statistics);

            statistics.ticks = 0;
            statistics.gpuMs = statistics.cpuMs = statistics.kernelMs = 0.0;
            statistics.maxDifference = 0f;
            statistics.compared = statistics.overTolerance = 0;
            statistics.gpuIterations = statistics.cpuIterations = 0;
            statistics.unconvergedTicks = statistics.extraBatches = 0;
            statistics.stateMismatches = 0;
        }

        // Prints one stage's timing and self-check figures
        private void PrintStatistics(StageStatistics statistics)
        {
            string line = $"SparseFluidGpu: {statistics.name} over {statistics.ticks} ticks, {slots} brick slots, GPU {statistics.gpuMs / statistics.ticks:F2} ms/tick";
            if (measureKernels)
                line += $" of which kernels {statistics.kernelMs / statistics.ticks:F2}";
            if (statistics.gpuIterations > 0)
                line += $", GPU iterations {(double)statistics.gpuIterations / statistics.ticks:F1}/tick ({statistics.extraBatches} extra batches, {statistics.unconvergedTicks} ticks hit the iteration limit)";
            if (selfCheck)
            {
                line += $", CPU {statistics.cpuMs / statistics.ticks:F2} ms/tick";
                if (statistics == projectStatistics)
                    line += $", CPU iterations {(double)statistics.cpuIterations / statistics.ticks:F1}/tick";
                line += $", self-check: max difference {statistics.maxDifference:E2}, {statistics.overTolerance} of {statistics.compared} values above {statistics.tolerance:E0}";
                if (statistics.comparesState)
                    line += $", {statistics.stateMismatches} cells with a different state";
            }
            Console.WriteLine(line);
        }

        // Frees every buffer whose size follows the grid's capacity
        private void DisposeBuffers()
        {
            brickInfoBuffer?.Dispose();
            brickNeighborBuffer?.Dispose();
            solidBuffer?.Dispose();
            foreach (Mirror mirror in mirrors)
                mirror.buffer?.Dispose();
            for (int axis = 0; axis < 3; axis++)
                faceScratchBuffer[axis]?.Dispose();
            diagBuffer?.Dispose();
            rBuffer?.Dispose();
            sBuffer?.Dispose();
            qBuffer?.Dispose();
            brickSumBuffer?.Dispose();
            brickMaxBuffer?.Dispose();
            scalarsBuffer?.Dispose();
            fillOutTotalBuffer?.Dispose();
            fillOutScaledBuffer?.Dispose();
            fillInTotalBuffer?.Dispose();
            fillRawBuffer?.Dispose();
            fillStatusBuffer?.Dispose();
            ledgerBuffer?.Dispose();
            brickLedgerBuffer?.Dispose();
            brickSummaryBuffer?.Dispose();
            visibleMaskBuffer?.Dispose();
        }

        // Hands the state back to the grid and frees all GPU buffers
        public void Dispose()
        {
            Release();
            DisposeBuffers();
            sourceBuffer?.Dispose();
            previousSourceBuffer?.Dispose();
            cellCapacity = 0;
        }
    }
}
