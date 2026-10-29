using Microsoft.Xna.Framework;
using NAADF.Common;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace NAADF.World.Data
{
    /*
    * Sparse Volume-of-Fluid solver on a staggered (MAC) grid, independent of the engine
    *
    * Per tick: gravity -> velocity advection -> sources -> pressure projection -> supported-air velocities -> fill transport -> thin water flow -> pruning -> damping/clamp
    * A cell counts as liquid for the pressure solve when fill >= liquidThreshold; every other non-solid cell is air
    * Solid geometry is queried through the isSolid delegate supplied by the owner
    * Cells for which the optional isDrain delegate returns true are emptied every tick, acting as an open outflow
    */
    public class SparseFluidSolver
    {
        public struct Source
        {
            public Point3 cell;
            public Vector3 velocity;
        }

        private struct Transfer
        {
            public int donor, receiver;
            public float amount;
        }

        private struct FaceUpdate
        {
            public int axis, index;
            public Point3 lower;
            public float value;
        }

        public readonly SparseFluidGrid grid = new SparseFluidGrid();

        private readonly Func<Point3, bool> isSolid;
        private readonly Func<Point3, bool> isDrain;
        private readonly Dictionary<long, bool> solidOutsideBricks = new Dictionary<long, bool>();
        private const int solidRefreshTicks = 64;
        private int solidRefreshSlot = 0;

        private const byte FlagLiquid = 1, FlagSource = 2, FlagGained = 4, FlagTouched = 8, FlagThinWater = 16;

        // Cell indices carrying a flag, kept so the flags can be cleared without sweeping the grid
        private readonly List<int> sourceList = new List<int>();
        private List<int> liquidList = new List<int>(), previousLiquidList = new List<int>();
        private readonly List<int> gainedList = new List<int>();
        private readonly List<int> touchedList = new List<int>();
        private readonly List<Transfer> transfers = new List<Transfer>();
        private readonly List<FaceUpdate> faceUpdates = new List<FaceUpdate>();

        // Pressure system buffers, grown as needed and reused across ticks
        private int[] neighbor = new int[0];
        private float[] airWeight = new float[0];
        private float[] diag = new float[0];
        private double[] b = new double[0], x = new double[0], r = new double[0], z = new double[0], s = new double[0], q = new double[0];

        public const float visibilityThreshold = 0.5f;
        public const float liquidThreshold = 0.5f;
        public const float minFillEpsilon = 0.001f;
        public const float cflMaxFraction = 0.5f;
        public const float minInterfaceTheta = 0.05f;
        public const float maxThinWaterRate = 0.2f;
        public const float thinWaterMinSpreadFill = 0.05f;

        private const int maxPressureIterations = 300;

        private static readonly Point3[] LatticeNeighbors =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        public const float pressureTolerance = 1e-4f;

        public bool enableGravity = true;
        public float gravityStrength = 20f;
        public float velocityDampingPerSecond = 0.3f; // fraction of velocity retained per second
        public int stagnantTicksBeforeDestroy = 150;
        public float compactionRate = 0.05f; // fraction of a liquid cell's missing fill drawn in per tick

        // Liquid cells pass on even negligible amounts, and dim cells resting on liquid are never pruned as empty or stagnant,
        // so a thin layer of fluid spreading over a surface is kept instead of being clamped or pruned away
        public bool keepSurfaceFilm = true;

        // Dim cells resting on solid ground, away from liquid, pass this fraction per second of their fill difference to neighbors holding less
        // (only while above thinWaterMinSpreadFill when the neighbor also rests on solid), and over edges onto unsupported cells; 0 turns it off
        public float thinWaterFlowPerSecond = 3f;

        // Stagnant dim cells lose evaporationPerSecond of fill per second instead of being removed all at once
        public bool gradualEvaporation = true;
        public float evaporationPerSecond = 0.05f;

        // Diagnostics
        public static readonly string[] StageNames = { "BeginTick", "Gravity", "AdvectVelocity", "Sources", "Project", "Extrapolate", "AdvectFill", "ThinWater", "Prune", "Damping", "Clamp" };
        public readonly double[] stageMs = new double[11]; // time spent per stage, accumulated until the owner clears it
        public int lastPrunedBySolid, lastPrunedByEmpty, lastDrained;
        public int lastLiquidCells, lastPressureIterations;
        public float lastPressureResidual;
        public float volumeInjected, volumeClampedAway, volumeClampCreated;
        public float volumePrunedSolid, volumePrunedEmpty, volumePrunedStagnant, volumeDrained;

        // Creates an empty solver; isSolid says which world cells block fluid, the optional isDrain which cells empty themselves
        public SparseFluidSolver(Func<Point3, bool> isSolid, Func<Point3, bool> isDrain = null)
        {
            this.isSolid = isSolid;
            this.isDrain = isDrain;
        }

        // Number of tracked cells
        public int Count => grid.Count;


        // Drops all fluid, cached solid lookups and flagged-cell lists
        public void Clear()
        {
            grid.Clear();
            solidOutsideBricks.Clear();
            sourceList.Clear();
            liquidList.Clear();
            previousLiquidList.Clear();
            gainedList.Clear();
        }

        // Removes the fluid in one cell, if it has any
        public void RemoveCell(Point3 p) => grid.RemoveKey(SparseFluidGrid.Key(p));

        // Sum of the fill of every cell
        public double TotalVolume()
        {
            double volume = 0.0;
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
                volume += grid.fill[i];
            return volume;
        }

        // Pressure from the last solve; false for cells that weren't liquid in it
        public bool TryGetPressure(Point3 p, out float value)
        {
            int index = grid.CellIndex(p);
            bool liquid = index >= 0 && grid.liquidIndex[index] != 0;
            value = liquid ? grid.pressure[index] : 0f;
            return liquid;
        }

        // Zeroes the volume bookkeeping that the owner reads and resets between log lines
        public void ClearVolumeAccumulators()
        {
            volumeInjected = volumeClampedAway = volumeClampCreated = 0f;
            volumePrunedSolid = volumePrunedEmpty = volumePrunedStagnant = volumeDrained = 0f;
        }

        // Adds a sphere of fill-fraction data to the cells it covers, each capped at full; cells inside solid are skipped
        public void SeedSphere(Point3 center, int radius)
        {
            ClearSolidCache();
            int bound = radius + 1;
            for (int x = -bound; x <= bound; x++)
                for (int y = -bound; y <= bound; y++)
                    for (int z = -bound; z <= bound; z++)
                    {
                        float distance = MathF.Sqrt(x * x + y * y + z * z);
                        float fill = MathHelper.Clamp(radius - distance + 0.5f, 0f, 1f);
                        if (fill <= 0f)
                            continue;

                        Point3 pos = center + new Point3(x, y, z);
                        int index = grid.EnsureCellIndex(pos);
                        if (IsSolid(index, pos))
                            continue;

                        grid.Track(index);
                        grid.fill[index] = MathF.Min(grid.fill[index] + fill, 1f);
                        grid.ticksStagnant[index] = 0;
                    }
        }

        // Adds fluid to every non-solid cell of a box (bounds inclusive), topFill in the top layer (y = max.Y) and full elsewhere, each capped at full
        public void SeedBox(Point3 min, Point3 max, float topFill = 1f)
        {
            ClearSolidCache();
            for (int z = min.Z; z <= max.Z; z++)
                for (int y = min.Y; y <= max.Y; y++)
                    for (int x = min.X; x <= max.X; x++)
                    {
                        Point3 pos = new Point3(x, y, z);
                        int index = grid.EnsureCellIndex(pos);
                        if (IsSolid(index, pos))
                            continue;

                        grid.Track(index);
                        grid.fill[index] = MathF.Min(grid.fill[index] + (y == max.Y ? topFill : 1f), 1f);
                        grid.ticksStagnant[index] = 0;
                    }
        }

        // Advances the simulation by one tick on the CPU, timing each stage into stageMs
        public void Step(float dt, IReadOnlyList<Source> sources)
        {
            long start = Stopwatch.GetTimestamp();
            BeginTick();
            start = AddStageTime(0, start);

            if (enableGravity)
                ApplyGravity(dt);
            start = AddStageTime(1, start);
            AdvectVelocity(dt);
            start = AddStageTime(2, start);
            ApplySources(sources);
            start = AddStageTime(3, start);
            Project(dt);
            start = AddStageTime(4, start);
            ExtrapolateIntoSupportedAir();
            start = AddStageTime(5, start);
            AdvectFill(dt);
            start = AddStageTime(6, start);
            FlowThinWater(dt);
            start = AddStageTime(7, start);
            PruneStaleCells(dt);
            start = AddStageTime(8, start);
            ApplyDamping(dt);
            start = AddStageTime(9, start);
            ClampVelocity(dt);
            AddStageTime(10, start);
        }

        // Adds the time since start to a stage's total and returns the current timestamp, for timing the next stage
        private long AddStageTime(int stage, long start)
        {
            long now = Stopwatch.GetTimestamp();
            stageMs[stage] += (now - start) * 1000.0 / Stopwatch.Frequency;
            return now;
        }

        // Works out the brick borders from the cells, then updates the brick layout
        public void BeginTick()
        {
            grid.ScanBorders();
            UpdateLayout();
        }

        // Brings the brick layout in line with where the fluid is, going by the grid's per-brick figures
        // Solid lookups are kept from tick to tick; a few bricks are forgotten each tick so every brick is re-queried about once a second
        public void UpdateLayout()
        {
            grid.UpdateTopology();
            solidOutsideBricks.Clear();

            int slots = grid.SlotCount;
            int refreshCount = (slots + solidRefreshTicks - 1) / solidRefreshTicks;
            for (int k = 0; k < refreshCount; k++)
            {
                if (solidRefreshSlot >= slots)
                    solidRefreshSlot = 0;
                Array.Clear(grid.solid, solidRefreshSlot << SparseFluidGrid.CellShift, SparseFluidGrid.BrickCells);
                solidRefreshSlot++;
            }
        }

        // Forgets every cached solid lookup
        private void ClearSolidCache()
        {
            Array.Clear(grid.solid, 0, grid.SlotCount * SparseFluidGrid.BrickCells);
            solidOutsideBricks.Clear();
        }

        // To be called when solid geometry inside a box of cells (bounds inclusive) has changed
        public void InvalidateSolid(Point3 min, Point3 max)
        {
            grid.ClearSolid(min, max);
            solidOutsideBricks.Clear();
        }

        // Queries the solid state of every cell of every brick that isn't known yet; returns whether there were any
        public bool FillSolid()
        {
            bool anyUnknown = false;
            for (int slot = 0; slot < grid.SlotCount; slot++)
            {
                if (!grid.IsSlotAllocated(slot))
                    continue;

                int start = slot << SparseFluidGrid.CellShift;
                for (int local = 0; local < SparseFluidGrid.BrickCells; local++)
                    if (grid.solid[start + local] == 0)
                    {
                        grid.solid[start + local] = SolidState(grid.CellPosition(start + local));
                        anyUnknown = true;
                    }
            }
            return anyUnknown;
        }

        // Queries the owner for a cell's solid and drain state, packed as the grid's solid cache byte
        private byte SolidState(Point3 p)
        {
            byte state = isSolid(p) ? SparseFluidGrid.SolidBlocked : SparseFluidGrid.SolidFree;
            if (isDrain != null && isDrain(p))
                state |= SparseFluidGrid.SolidDrain;
            return state;
        }

        // Whether the cell at a world position is solid, using the cache
        public bool IsSolid(Point3 p) => IsSolid(grid.CellIndex(p), p);

        // Whether the cell below is a liquid cell that this tick's pruning keeps (neither solid nor a drain)
        private bool RestsOnLiquid(int index)
        {
            int below = grid.NeighborIndex(index, 1, -1);
            if (below < 0 || (grid.flags[below] & FlagLiquid) == 0)
                return false;
            byte state = grid.solid[below];
            if (state == 0)
            {
                state = SolidState(grid.CellPosition(below));
                grid.solid[below] = state;
            }
            return (state & (SparseFluidGrid.SolidBlocked | SparseFluidGrid.SolidDrain)) == 0;
        }

        // Whether a cell is solid: cached per cell inside bricks, cached for the tick outside them
        // index is the cell's index, or -1 when no brick covers it
        private bool IsSolid(int index, Point3 p)
        {
            if (index >= 0)
            {
                byte state = grid.solid[index];
                if (state == 0)
                {
                    state = SolidState(p);
                    grid.solid[index] = state;
                }
                return (state & SparseFluidGrid.SolidBlocked) != 0;
            }

            long key = SparseFluidGrid.Key(p);
            if (!solidOutsideBricks.TryGetValue(key, out bool solid))
            {
                solid = isSolid(p);
                solidOutsideBricks[key] = solid;
            }
            return solid;
        }

        // Index of the cell one step from a cell that may itself lie outside every brick (index -1)
        private int StepIndex(int index, Point3 target, int axis, int direction)
        {
            return index >= 0 ? grid.NeighborIndex(index, axis, direction) : grid.CellIndex(target);
        }

        // Accelerates every present vertical face downwards by one tick of gravity
        public void ApplyGravity(float dt)
        {
            float[] faceY = grid.face[1];
            byte[] present = grid.facePresent[1];
            float dv = gravityStrength * dt;
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
                if (present[i] != 0)
                    faceY[i] -= dv;
        }

        // Semi-Lagrangian: every face of every tracked cell takes the velocity found by tracing back from its center
        public void AdvectVelocity(float dt)
        {
            byte[] tracked = grid.tracked;
            for (int slot = 0; slot < grid.SlotCount; slot++)
            {
                if (!grid.IsSlotAllocated(slot))
                    continue;

                Point3 origin = grid.SlotOrigin(slot);
                int start = slot << SparseFluidGrid.CellShift;
                int presentCount = 0;
                for (int local = 0; local < SparseFluidGrid.BrickCells; local++)
                {
                    int i = start + local;
                    Vector3 cellPos = new Vector3(
                        origin.X + (local & SparseFluidGrid.BrickMask),
                        origin.Y + ((local >> SparseFluidGrid.BrickShift) & SparseFluidGrid.BrickMask),
                        origin.Z + (local >> (SparseFluidGrid.BrickShift * 2)));

                    for (int axis = 0; axis < 3; axis++)
                    {
                        bool active = tracked[i] != 0;
                        if (!active)
                        {
                            int upper = grid.NeighborIndex(i, axis, 1);
                            active = upper >= 0 && tracked[upper] != 0;
                        }

                        if (!active)
                        {
                            grid.faceScratch[axis][i] = 0f;
                            grid.facePresent[axis][i] = 0;
                            continue;
                        }

                        Vector3 facePos = cellPos + SparseFluidGrid.AxisOffsets[axis].ToVector3() * 0.5f;
                        Vector3 backPos = facePos - grid.SampleVelocity(facePos, slot) * dt;
                        grid.faceScratch[axis][i] = grid.SampleComponent(axis, backPos, slot);
                        grid.facePresent[axis][i] = 1;
                        presentCount++;
                    }
                }
                grid.brickFacesPresent[slot] = presentCount;
            }

            (grid.face, grid.faceScratch) = (grid.faceScratch, grid.face);
        }

        // Source cells are held full, and all their faces are held at the source velocity through the pressure solve
        public void ApplySources(IReadOnlyList<Source> sources)
        {
            foreach (int index in sourceList)
                grid.flags[index] &= unchecked((byte)~FlagSource);
            sourceList.Clear();
            if (sources == null)
                return;

            foreach (Source source in sources)
            {
                if (IsSolid(source.cell))
                    continue;

                int index = grid.EnsureCellIndex(source.cell);
                volumeInjected += 1f - grid.fill[index];
                grid.Track(index);
                grid.fill[index] = 1f;
                grid.ticksStagnant[index] = 0;

                for (int axis = 0; axis < 3; axis++)
                {
                    float component = axis == 0 ? source.velocity.X : (axis == 1 ? source.velocity.Y : source.velocity.Z);
                    grid.SetFaceAt(axis, index, component);
                    grid.SetFace(axis, source.cell - SparseFluidGrid.AxisOffsets[axis], component);
                }
                grid.flags[index] |= FlagSource;
                sourceList.Add(index);

                // A source cell can appear anywhere mid-tick, so the bricks around it are made sure to exist here
                foreach (Point3 offset in LatticeNeighbors)
                    grid.EnsureCellIndex(source.cell + offset);
            }
        }

        // Zeroes the velocity of every present face that touches a solid cell
        private void ZeroSolidFaces()
        {
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int axis = 0; axis < 3; axis++)
            {
                Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                byte[] present = grid.facePresent[axis];
                float[] f = grid.face[axis];
                for (int i = 0; i < end; i++)
                {
                    if (present[i] == 0)
                        continue;

                    Point3 lower = grid.CellPosition(i);
                    if (IsSolid(i, lower) || IsSolid(grid.NeighborIndex(i, axis, 1), lower + offset))
                        f[i] = 0f;
                }
            }
        }

        // Solves for pressure on liquid cells so the face velocities around them become divergence-free,
        // except for a small inflow into liquid cells that aren't full, which packs trapped air back out
        // Neighbor rules: liquid = unknown, air = zero pressure at the interface (ghost fluid), solid or source = no flow through the face
        // The interface sits between a liquid and an air cell where fill crosses liquidThreshold, so air faces are weighted by 1/theta
        public void Project(float dt)
        {
            ClassifyLiquidCells();
            SolveProjection(dt);
        }

        // Rebuilds the lists of flagged cells from the flags themselves, after the per-cell arrays were filled in from elsewhere
        public void RebuildCellLists()
        {
            sourceList.Clear();
            liquidList.Clear();
            previousLiquidList.Clear();
            gainedList.Clear();

            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
            {
                byte cellFlags = grid.flags[i];
                grid.liquidIndex[i] = 0;
                if ((cellFlags & FlagSource) != 0)
                    sourceList.Add(i);
                if ((cellFlags & FlagGained) != 0)
                    gainedList.Add(i);
                if ((cellFlags & FlagLiquid) != 0)
                {
                    liquidList.Add(i);
                    grid.liquidIndex[i] = liquidList.Count;
                }
            }
            lastLiquidCells = liquidList.Count;
        }

        // Decides which cells are liquid for this tick's pressure solve
        public void ClassifyLiquidCells()
        {
            // Last tick's liquid cells keep their pressure, as the warm start, until the new system has read it
            (liquidList, previousLiquidList) = (previousLiquidList, liquidList);
            foreach (int index in previousLiquidList)
            {
                grid.flags[index] &= unchecked((byte)~FlagLiquid);
                grid.liquidIndex[index] = 0;
            }
            liquidList.Clear();

            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
            {
                if (grid.tracked[i] == 0 || grid.fill[i] < liquidThreshold || (grid.flags[i] & FlagSource) != 0)
                    continue;
                if (IsSolid(i, grid.CellPosition(i)))
                    continue;

                liquidList.Add(i);
                grid.flags[i] |= FlagLiquid;
                grid.liquidIndex[i] = liquidList.Count;
            }

            lastLiquidCells = liquidList.Count;
        }

        // The pressure solve itself, for the cells ClassifyLiquidCells marked
        public void SolveProjection(float dt)
        {
            ZeroSolidFaces();

            int n = liquidList.Count;
            lastPressureIterations = 0;
            lastPressureResidual = 0f;
            if (n == 0)
            {
                foreach (int index in previousLiquidList)
                    grid.pressure[index] = 0f;
                return;
            }

            if (diag.Length < n)
            {
                int capacity = Math.Max(n, diag.Length * 2);
                neighbor = new int[capacity * 6];
                airWeight = new float[capacity * 6];
                diag = new float[capacity];
                b = new double[capacity];
                x = new double[capacity];
                r = new double[capacity];
                z = new double[capacity];
                s = new double[capacity];
                q = new double[capacity];
            }

            // neighbor[i*6+d]: liquid index, -1 air, -2 solid/source; direction d = axis*2 + (0 = +, 1 = -)
            for (int i = 0; i < n; i++)
            {
                int cell = liquidList[i];
                Point3 pos = grid.CellPosition(cell);
                float fill = grid.fill[cell];
                float diagonal = 0f;
                float divergence = 0f;
                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                    int lowerCell = grid.NeighborIndex(cell, axis, -1);
                    divergence += grid.face[axis][cell] - (lowerCell >= 0 ? grid.face[axis][lowerCell] : 0f);

                    for (int side = 0; side < 2; side++)
                    {
                        Point3 np = side == 0 ? pos + offset : pos - offset;
                        int nCell = side == 0 ? grid.NeighborIndex(cell, axis, 1) : lowerCell;
                        int type;
                        if (IsSolid(nCell, np) || (nCell >= 0 && (grid.flags[nCell] & FlagSource) != 0))
                            type = -2;
                        else if (nCell >= 0 && grid.liquidIndex[nCell] != 0)
                            type = grid.liquidIndex[nCell] - 1;
                        else
                            type = -1;

                        neighbor[i * 6 + axis * 2 + side] = type;
                        airWeight[i * 6 + axis * 2 + side] = 0f;
                        if (type >= 0)
                            diagonal += 1f;
                        else if (type == -1)
                        {
                            float nFill = nCell >= 0 ? grid.fill[nCell] : 0f;
                            float theta = MathHelper.Clamp((fill - liquidThreshold) / MathF.Max(fill - nFill, 1e-6f), minInterfaceTheta, 1f);
                            airWeight[i * 6 + axis * 2 + side] = 1f / theta;
                            diagonal += 1f / theta;
                        }
                    }
                }
                diag[i] = diagonal > 0f ? diagonal : 1f;
                float targetDivergence = -compactionRate * (1f - fill) / dt;
                b[i] = targetDivergence - divergence;
                x[i] = grid.pressure[cell];
            }

            SolvePressure(n);

            foreach (int index in previousLiquidList)
                grid.pressure[index] = 0f;
            for (int i = 0; i < n; i++)
                grid.pressure[liquidList[i]] = (float)x[i];

            // u -= p(upper) - p(lower) on every face bordering liquid that isn't solid/source; liquid-liquid faces are handled once, from their lower cell
            for (int i = 0; i < n; i++)
            {
                int cell = liquidList[i];
                for (int axis = 0; axis < 3; axis++)
                {
                    int up = neighbor[i * 6 + axis * 2];
                    if (up != -2)
                    {
                        double pUp = up >= 0 ? x[up] : 0.0;
                        double weight = up >= 0 ? 1.0 : airWeight[i * 6 + axis * 2];
                        grid.SetFaceAt(axis, cell, grid.face[axis][cell] - (float)((pUp - x[i]) * weight));
                    }

                    int down = neighbor[i * 6 + axis * 2 + 1];
                    if (down == -1)
                    {
                        int lowerCell = grid.NeighborIndex(cell, axis, -1);
                        if (lowerCell < 0)
                            lowerCell = grid.EnsureCellIndex(grid.CellPosition(cell) - SparseFluidGrid.AxisOffsets[axis]);
                        grid.SetFaceAt(axis, lowerCell, grid.face[axis][lowerCell] - (float)(x[i] * airWeight[i * 6 + axis * 2 + 1]));
                    }
                }
            }
        }

        // Whether a cell is liquid or a source in this tick's pressure solve
        private bool IsFluidCell(int index) => index >= 0 && (grid.flags[index] & (FlagLiquid | FlagSource)) != 0;

        // A face the pressure solve has made consistent: not against solid, and bordering a liquid or source cell
        private bool IsValidFace(int axis, int lowerIndex, Point3 lower)
        {
            Point3 upper = lower + SparseFluidGrid.AxisOffsets[axis];
            int upperIndex = StepIndex(lowerIndex, upper, axis, 1);
            if (IsSolid(lowerIndex, lower) || IsSolid(upperIndex, upper))
                return false;
            return IsFluidCell(lowerIndex) || IsFluidCell(upperIndex);
        }

        // Below-threshold cells resting directly on liquid or solid take the velocity of the liquid around them on their free faces
        // (zero when resting on solid alone), so gravity can't build up on them; unsupported cells keep their own ballistic velocity
        public void ExtrapolateIntoSupportedAir()
        {
            faceUpdates.Clear();
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
            {
                if (grid.tracked[i] == 0 || IsFluidCell(i) || grid.fill[i] <= 0f)
                    continue;

                Point3 pos = grid.CellPosition(i);
                Point3 below = pos - SparseFluidGrid.AxisOffsets[1];
                int belowIndex = grid.NeighborIndex(i, 1, -1);
                bool belowLiquid = belowIndex >= 0 && (grid.flags[belowIndex] & FlagLiquid) != 0;
                if (!belowLiquid && !IsSolid(belowIndex, below))
                    continue;

                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                    for (int side = 0; side < 2; side++)
                    {
                        Point3 lower = side == 0 ? pos : pos - offset;
                        int lowerIndex = side == 0 ? i : grid.NeighborIndex(i, axis, -1);
                        int upperIndex = side == 0 ? grid.NeighborIndex(i, axis, 1) : i;
                        if (IsSolid(lowerIndex, lower) || IsSolid(upperIndex, lower + offset) || IsValidFace(axis, lowerIndex, lower))
                            continue;

                        float sum = 0f;
                        int count = 0;
                        for (int d = 0; d < 6; d++)
                        {
                            Point3 neighborLower = lower + LatticeNeighbors[d];
                            int neighborIndex = StepIndex(lowerIndex, neighborLower, d >> 1, (d & 1) == 0 ? 1 : -1);
                            if (IsValidFace(axis, neighborIndex, neighborLower))
                            {
                                sum += neighborIndex >= 0 ? grid.face[axis][neighborIndex] : 0f;
                                count++;
                            }
                        }
                        faceUpdates.Add(new FaceUpdate { axis = axis, index = lowerIndex, lower = lower, value = count > 0 ? sum / count : 0f });
                    }
                }
            }

            foreach (FaceUpdate update in faceUpdates)
            {
                int index = update.index >= 0 ? update.index : grid.EnsureCellIndex(update.lower);
                grid.SetFaceAt(update.axis, index, update.value);
            }
        }

        // Jacobi-preconditioned conjugate gradient for A p = b over the first n entries, with (A p)_i = diag_i p_i - sum of liquid neighbor p
        private void SolvePressure(int n)
        {
            void ApplyA(double[] v, double[] result)
            {
                for (int i = 0; i < n; i++)
                {
                    double sum = diag[i] * v[i];
                    for (int d = 0; d < 6; d++)
                    {
                        int j = neighbor[i * 6 + d];
                        if (j >= 0)
                            sum -= v[j];
                    }
                    result[i] = sum;
                }
            }

            double MaxAbs(double[] v)
            {
                double m = 0.0;
                for (int i = 0; i < n; i++)
                    m = Math.Max(m, Math.Abs(v[i]));
                return m;
            }

            ApplyA(x, q);
            for (int i = 0; i < n; i++)
            {
                r[i] = b[i] - q[i];
                z[i] = r[i] / diag[i];
                s[i] = z[i];
            }

            double rho = 0.0;
            for (int i = 0; i < n; i++)
                rho += r[i] * z[i];

            int iteration = 0;
            double residual = MaxAbs(r);
            while (residual > pressureTolerance && iteration < maxPressureIterations)
            {
                ApplyA(s, q);
                double sq = 0.0;
                for (int i = 0; i < n; i++)
                    sq += s[i] * q[i];
                if (Math.Abs(sq) < 1e-30)
                    break;

                double alpha = rho / sq;
                for (int i = 0; i < n; i++)
                {
                    x[i] += alpha * s[i];
                    r[i] -= alpha * q[i];
                }

                residual = MaxAbs(r);
                iteration++;
                if (residual <= pressureTolerance)
                    break;

                double rhoNew = 0.0;
                for (int i = 0; i < n; i++)
                {
                    z[i] = r[i] / diag[i];
                    rhoNew += r[i] * z[i];
                }
                double beta = rhoNew / rho;
                rho = rhoNew;
                for (int i = 0; i < n; i++)
                    s[i] = z[i] + beta * s[i];
            }

            lastPressureIterations = iteration;
            lastPressureResidual = (float)residual;
        }

        // Donor-cell face fluxes using the face velocity directly; whatever leaves one side of a face enters the other
        // Bounded: each donor's total outflow is scaled to what it holds, then each receiver's total inflow is scaled to its free space
        public void AdvectFill(float dt)
        {
            foreach (int index in gainedList)
                grid.flags[index] &= unchecked((byte)~FlagGained);
            gainedList.Clear();

            // Transfers as (donor, receiver, amount > 0), one per face whose upwind cell holds fluid
            transfers.Clear();
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
            {
                for (int axis = 0; axis < 3; axis++)
                {
                    float flux = MathHelper.Clamp(grid.face[axis][i] * dt, -1f, 1f);
                    if (flux == 0f)
                        continue;

                    int upper = grid.NeighborIndex(i, axis, 1);
                    float lowerFill = grid.fill[i];
                    float upperFill = upper >= 0 ? grid.fill[upper] : 0f;
                    if (flux > 0f ? lowerFill <= minFillEpsilon : upperFill <= minFillEpsilon)
                        continue;

                    Point3 lowerPos = grid.CellPosition(i);
                    Point3 upperPos = lowerPos + SparseFluidGrid.AxisOffsets[axis];
                    if (IsSolid(i, lowerPos) || IsSolid(upper, upperPos))
                        continue;

                    if (flux > 0f)
                    {
                        if (upper < 0)
                            upper = grid.EnsureCellIndex(upperPos);
                        transfers.Add(new Transfer { donor = i, receiver = upper, amount = flux * lowerFill });
                    }
                    else
                        transfers.Add(new Transfer { donor = upper, receiver = i, amount = -flux * upperFill });
                }
            }

            float[] fill = grid.fill, outTotal = grid.outTotal, outScaled = grid.outScaled, inTotal = grid.inTotal, delta = grid.delta;
            byte[] flags = grid.flags;
            int count = transfers.Count;

            for (int k = 0; k < count; k++)
                outTotal[transfers[k].donor] += transfers[k].amount;
            for (int k = 0; k < count; k++)
            {
                Transfer t = transfers[k];
                float available = fill[t.donor];
                float total = outTotal[t.donor];
                if (total > available)
                {
                    t.amount = t.amount * available / total;
                    transfers[k] = t;
                }
            }

            for (int k = 0; k < count; k++)
            {
                Transfer t = transfers[k];
                outScaled[t.donor] += t.amount;
                inTotal[t.receiver] += t.amount;
            }
            for (int k = 0; k < count; k++)
            {
                Transfer t = transfers[k];
                float incoming = inTotal[t.receiver];
                float space = 1f - fill[t.receiver] + outScaled[t.receiver];
                if (incoming > space)
                {
                    t.amount = t.amount * MathF.Max(space, 0f) / incoming;
                    transfers[k] = t;
                }
            }

            touchedList.Clear();
            for (int k = 0; k < count; k++)
            {
                Transfer t = transfers[k];
                if (t.amount < minFillEpsilon && fill[t.receiver] <= minFillEpsilon && !(keepSurfaceFilm && (flags[t.donor] & FlagLiquid) != 0))
                    continue; // don't create new cells from negligible amounts

                delta[t.donor] -= t.amount;
                delta[t.receiver] += t.amount;
                if ((flags[t.donor] & FlagTouched) == 0)
                {
                    flags[t.donor] |= FlagTouched;
                    touchedList.Add(t.donor);
                }
                if ((flags[t.receiver] & FlagTouched) == 0)
                {
                    flags[t.receiver] |= FlagTouched;
                    touchedList.Add(t.receiver);
                }
            }

            foreach (int index in touchedList)
            {
                float rawFill = fill[index] + delta[index];
                if (rawFill > 1f)
                    volumeClampedAway += rawFill - 1f;
                else if (rawFill < 0f)
                    volumeClampCreated += -rawFill;
                grid.Track(index);
                fill[index] = MathHelper.Clamp(rawFill, 0f, 1f);

                flags[index] &= unchecked((byte)~FlagTouched);
                if (delta[index] > 0f)
                {
                    flags[index] |= FlagGained;
                    gainedList.Add(index);
                }
                delta[index] = 0f;
            }

            for (int k = 0; k < count; k++)
            {
                outTotal[transfers[k].donor] = 0f;
                outScaled[transfers[k].donor] = 0f;
                inTotal[transfers[k].receiver] = 0f;
            }
        }

        // Rate per tick at which thin water flows, 0 when thin water flow is off
        public float ThinWaterRate(float dt) => thinWaterFlowPerSecond <= 0f ? 0f : MathF.Min(thinWaterFlowPerSecond * dt, maxThinWaterRate);

        // Fill a stagnant dim cell loses per tick, 0 when stagnant cells are removed all at once
        public float EvaporationStep(float dt) => gradualEvaporation ? MathF.Max(evaporationPerSecond * dt, 0f) : 0f;

        // Lets dim cells resting on solid ground flow towards neighbors holding less and over edges, volume only moves between cells
        public void FlowThinWater(float dt)
        {
            float rate = ThinWaterRate(dt);
            if (rate <= 0f)
                return;

            touchedList.Clear();
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
                if (IsThinWater(i))
                    grid.flags[i] |= FlagThinWater;

            for (int i = 0; i < end; i++)
            {
                if ((grid.flags[i] & FlagThinWater) == 0)
                    continue;
                MarkTouched(i);
                for (int d = 0; d < 4; d++)
                {
                    int j = HorizontalNeighbor(i, d);
                    if (j >= 0 && ThinWaterFlow(i, j, rate) > 0f)
                        MarkTouched(j);
                }
            }

            foreach (int index in touchedList)
            {
                float delta = 0f;
                for (int d = 0; d < 4; d++)
                {
                    int j = HorizontalNeighbor(index, d);
                    if (j >= 0)
                        delta += ThinWaterFlow(j, index, rate) - ThinWaterFlow(index, j, rate);
                }
                grid.delta[index] = delta;
            }

            foreach (int index in touchedList)
            {
                grid.fill[index] += grid.delta[index];
                grid.delta[index] = 0f;
                if (grid.fill[index] > 0f)
                    grid.Track(index);
                grid.flags[index] &= unchecked((byte)~FlagTouched);
            }
            for (int i = 0; i < end; i++)
                grid.flags[i] &= unchecked((byte)~FlagThinWater);
        }

        // Adds a cell to touchedList, once
        private void MarkTouched(int index)
        {
            if ((grid.flags[index] & FlagTouched) != 0)
                return;
            grid.flags[index] |= FlagTouched;
            touchedList.Add(index);
        }

        // Neighbor in the horizontal direction d: 0 = +X, 1 = -X, 2 = +Z, 3 = -Z
        private int HorizontalNeighbor(int index, int d) => grid.NeighborIndex(index, d < 2 ? 0 : 2, (d & 1) == 0 ? 1 : -1);

        // A dim cell resting on solid with no liquid or source cell beside or above it, so nothing else gives it a velocity
        private bool IsThinWater(int index)
        {
            if (grid.tracked[index] == 0 || (grid.flags[index] & (FlagLiquid | FlagSource)) != 0)
                return false;
            float f = grid.fill[index];
            if (f <= minFillEpsilon || f >= liquidThreshold)
                return false;
            if (IsSolid(index, grid.CellPosition(index)) || !SolidBelow(index, false))
                return false;

            for (int d = 0; d < 4; d++)
                if (IsFluidCell(HorizontalNeighbor(index, d)))
                    return false;
            return !IsFluidCell(grid.NeighborIndex(index, 1, 1));
        }

        // Fill the thin water cell from passes to its horizontal neighbor to this tick; 0 when from isn't thin water or to can't take it
        private float ThinWaterFlow(int from, int to, float rate)
        {
            if ((grid.flags[from] & FlagThinWater) == 0 || (grid.flags[to] & (FlagLiquid | FlagSource)) != 0)
                return 0f;
            float fromFill = grid.fill[from], toFill = grid.fill[to];
            if (toFill >= fromFill)
                return 0f;
            if (IsSolid(to, grid.CellPosition(to)))
                return 0f;
            if (fromFill <= thinWaterMinSpreadFill && SolidBelow(to, true))
                return 0f;
            return rate * (fromFill - toFill);
        }

        // Whether the cell below is solid; whenNoBrick is the answer when no brick covers it, as the GPU can't look it up
        private bool SolidBelow(int index, bool whenNoBrick)
        {
            int below = grid.NeighborIndex(index, 1, -1);
            return below < 0 ? whenNoBrick : IsSolid(below, grid.CellPosition(below));
        }

        // Removes cells inside solid or a drain, cells holding (almost) nothing, and dim cells that have stopped gaining fill for too long
        // (with gradualEvaporation, those lose evaporationPerSecond of fill per second instead, until nearly empty)
        public void PruneStaleCells(float dt)
        {
            float evaporation = EvaporationStep(dt);
            lastPrunedBySolid = 0;
            lastPrunedByEmpty = 0;
            lastDrained = 0;

            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            for (int i = 0; i < end; i++)
            {
                if (grid.tracked[i] == 0)
                    continue;

                float fill = grid.fill[i];
                Point3 pos = grid.CellPosition(i);
                if (isDrain != null && isDrain(pos))
                {
                    grid.Untrack(i);
                    lastDrained++;
                    volumeDrained += fill;
                    continue;
                }

                if (IsSolid(i, pos))
                {
                    grid.Untrack(i);
                    lastPrunedBySolid++;
                    volumePrunedSolid += fill;
                    continue;
                }

                if (keepSurfaceFilm && fill > 0f && fill <= visibilityThreshold && RestsOnLiquid(i))
                {
                    grid.ticksStagnant[i] = 0;
                    continue;
                }

                if (fill <= minFillEpsilon)
                {
                    grid.Untrack(i);
                    lastPrunedByEmpty++;
                    volumePrunedEmpty += fill;
                    continue;
                }

                bool dim = fill <= visibilityThreshold;
                bool gained = (grid.flags[i] & FlagGained) != 0;
                if (!dim || gained)
                {
                    if (gained)
                        grid.ticksStagnant[i] = 0;
                    continue;
                }

                grid.ticksStagnant[i]++;
                if (grid.ticksStagnant[i] >= stagnantTicksBeforeDestroy)
                {
                    float left = fill - evaporation;
                    if (evaporation > 0f && left > minFillEpsilon)
                    {
                        grid.fill[i] = left;
                        volumePrunedStagnant += fill - left;
                        continue;
                    }
                    grid.Untrack(i);
                    lastPrunedByEmpty++;
                    volumePrunedStagnant += fill;
                }
            }
        }

        // Scales every face velocity so that velocityDampingPerSecond of it remains after one second
        public void ApplyDamping(float dt)
        {
            if (velocityDampingPerSecond >= 1f)
                return;

            float factor = MathF.Pow(MathHelper.Clamp(velocityDampingPerSecond, 0f, 1f), dt);
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            foreach (float[] f in grid.face)
                for (int i = 0; i < end; i++)
                    f[i] *= factor;
        }

        // No face may carry more than cflMaxFraction of a cell per tick
        public void ClampVelocity(float dt)
        {
            float maxSpeed = cflMaxFraction / dt;
            int end = grid.SlotCount * SparseFluidGrid.BrickCells;
            foreach (float[] f in grid.face)
                for (int i = 0; i < end; i++)
                    f[i] = MathHelper.Clamp(f[i], -maxSpeed, maxSpeed);
        }
    }
}
