using Microsoft.Xna.Framework;
using NAADF.Common;
using System;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * Sparse Volume-of-Fluid solver on a staggered (MAC) grid, independent of the engine
    *
    * Per tick: gravity -> velocity advection -> sources -> pressure projection -> supported-air velocities -> fill transport -> pruning -> damping/clamp
    * A cell counts as liquid for the pressure solve when fill >= liquidThreshold; every other non-solid cell is air
    * Solid geometry is queried through the isSolid delegate supplied by the owner
    */
    public class SparseFluidSolver
    {
        public struct Source
        {
            public Point3 cell;
            public Vector3 velocity;
        }

        public readonly SparseFluidGrid grid = new SparseFluidGrid();

        private readonly Func<Point3, bool> isSolid;
        private readonly Dictionary<long, bool> solidCache = new Dictionary<long, bool>();
        private readonly HashSet<long> sourceCells = new HashSet<long>();
        private readonly HashSet<long> gainedFillThisTick = new HashSet<long>();
        private Dictionary<long, float> pressure = new Dictionary<long, float>();
        private HashSet<long> liquidCells = new HashSet<long>();

        public const float visibilityThreshold = 0.5f;
        public const float liquidThreshold = 0.5f;
        private const float minFillEpsilon = 0.001f;
        private const float cflMaxFraction = 0.5f;
        private const float minInterfaceTheta = 0.05f;

        private const float pressureTolerance = 1e-4f;
        private const int maxPressureIterations = 300;

        private static readonly Point3[] LatticeNeighbors =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        public bool enableGravity = true;
        public float gravityStrength = 20f;
        public float velocityDampingPerSecond = 0.3f; // fraction of velocity retained per second
        public int stagnantTicksBeforeDestroy = 150;
        public float compactionRate = 0.05f; // fraction of a liquid cell's missing fill drawn in per tick

        // Diagnostics
        public int lastPrunedBySolid, lastPrunedByEmpty;
        public int lastLiquidCells, lastPressureIterations;
        public float lastPressureResidual;
        public float volumeInjected, volumeClampedAway, volumeClampCreated;
        public float volumePrunedSolid, volumePrunedEmpty, volumePrunedStagnant;

        public SparseFluidSolver(Func<Point3, bool> isSolid)
        {
            this.isSolid = isSolid;
        }

        public int Count => grid.Count;

        public void Clear()
        {
            grid.Clear();
            pressure.Clear();
            solidCache.Clear();
        }

        public void RemoveCell(Point3 p) => grid.RemoveKey(SparseFluidGrid.Key(p));

        public float TotalVolume()
        {
            float volume = 0f;
            foreach (long key in grid.SnapshotKeys())
                volume += grid.GetFill(key);
            return volume;
        }

        public void ClearVolumeAccumulators()
        {
            volumeInjected = volumeClampedAway = volumeClampCreated = 0f;
            volumePrunedSolid = volumePrunedEmpty = volumePrunedStagnant = 0f;
        }

        // Fills a sphere with fill-fraction data at rest; cells inside solid are skipped
        public void SeedSphere(Point3 center, int radius)
        {
            solidCache.Clear();
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
                        if (IsSolid(pos))
                            continue;

                        grid.SetCell(SparseFluidGrid.Key(pos), new SparseFluidGrid.Cell { fill = fill });
                    }
        }

        public void Step(float dt, IReadOnlyList<Source> sources)
        {
            solidCache.Clear();

            if (enableGravity)
                ApplyGravity(dt);
            AdvectVelocity(dt);
            ApplySources(sources);
            Project(dt);
            ExtrapolateIntoSupportedAir();
            AdvectFill(dt);
            PruneStaleCells();
            ApplyDamping(dt);
            ClampVelocity(dt);
        }

        public bool IsSolid(Point3 p)
        {
            long key = SparseFluidGrid.Key(p);
            if (!solidCache.TryGetValue(key, out bool solid))
            {
                solid = isSolid(p);
                solidCache[key] = solid;
            }
            return solid;
        }

        private bool IsLiquid(long key, Point3 pos)
        {
            return grid.TryGetCell(key, out SparseFluidGrid.Cell cell) && cell.fill >= liquidThreshold
                && !sourceCells.Contains(key) && !IsSolid(pos);
        }

        private void ApplyGravity(float dt)
        {
            Dictionary<long, float> faceY = grid.faces[1];
            float dv = gravityStrength * dt;
            foreach (long key in new List<long>(faceY.Keys))
                faceY[key] -= dv;
        }

        // Semi-Lagrangian: every face of every tracked cell takes the velocity found by tracing back from its center
        private void AdvectVelocity(float dt)
        {
            long[] keys = grid.SnapshotKeys();
            Dictionary<long, float>[] advected = new Dictionary<long, float>[3];

            for (int axis = 0; axis < 3; axis++)
            {
                Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                HashSet<long> faceKeys = new HashSet<long>();
                foreach (long key in keys)
                {
                    faceKeys.Add(key);
                    faceKeys.Add(SparseFluidGrid.Key(SparseFluidGrid.Unpack(key) - offset));
                }

                Vector3 half = offset.ToVector3() * 0.5f;
                Dictionary<long, float> result = new Dictionary<long, float>(faceKeys.Count);
                foreach (long faceKey in faceKeys)
                {
                    Vector3 facePos = SparseFluidGrid.Unpack(faceKey).ToVector3() + half;
                    Vector3 backPos = facePos - grid.SampleVelocity(facePos) * dt;
                    result[faceKey] = grid.SampleComponent(axis, backPos);
                }
                advected[axis] = result;
            }

            grid.faces = advected;
        }

        // Source cells are held full, and all their faces are held at the source velocity through the pressure solve
        private void ApplySources(IReadOnlyList<Source> sources)
        {
            sourceCells.Clear();
            if (sources == null)
                return;

            foreach (Source source in sources)
            {
                if (IsSolid(source.cell))
                    continue;

                long key = SparseFluidGrid.Key(source.cell);
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                volumeInjected += 1f - cell.fill;
                cell.fill = 1f;
                cell.ticksStagnant = 0;
                grid.SetCell(key, cell);

                for (int axis = 0; axis < 3; axis++)
                {
                    float component = axis == 0 ? source.velocity.X : (axis == 1 ? source.velocity.Y : source.velocity.Z);
                    grid.faces[axis][key] = component;
                    grid.faces[axis][SparseFluidGrid.Key(source.cell - SparseFluidGrid.AxisOffsets[axis])] = component;
                }
                sourceCells.Add(key);
            }
        }

        private void ZeroSolidFaces()
        {
            for (int axis = 0; axis < 3; axis++)
            {
                Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                Dictionary<long, float> f = grid.faces[axis];
                foreach (long faceKey in new List<long>(f.Keys))
                {
                    Point3 lower = SparseFluidGrid.Unpack(faceKey);
                    if (IsSolid(lower) || IsSolid(lower + offset))
                        f[faceKey] = 0f;
                }
            }
        }

        // Solves for pressure on liquid cells so the face velocities around them become divergence-free,
        // except for a small inflow into liquid cells that aren't full, which packs trapped air back out
        // Neighbor rules: liquid = unknown, air = zero pressure at the interface (ghost fluid), solid or source = no flow through the face
        // The interface sits between a liquid and an air cell where fill crosses liquidThreshold, so air faces are weighted by 1/theta
        private void Project(float dt)
        {
            ZeroSolidFaces();

            List<long> liquidKeys = new List<long>();
            foreach (long key in grid.SnapshotKeys())
                if (IsLiquid(key, SparseFluidGrid.Unpack(key)))
                    liquidKeys.Add(key);

            int n = liquidKeys.Count;
            liquidCells = new HashSet<long>(liquidKeys);
            lastLiquidCells = n;
            lastPressureIterations = 0;
            lastPressureResidual = 0f;
            if (n == 0)
            {
                pressure.Clear();
                return;
            }

            Dictionary<long, int> index = new Dictionary<long, int>(n);
            for (int i = 0; i < n; i++)
                index[liquidKeys[i]] = i;

            // neighbor[i*6+d]: liquid index, -1 air, -2 solid/source; direction d = axis*2 + (0 = +, 1 = -)
            int[] neighbor = new int[n * 6];
            float[] airWeight = new float[n * 6];
            float[] diag = new float[n];
            double[] b = new double[n];

            for (int i = 0; i < n; i++)
            {
                Point3 pos = SparseFluidGrid.Unpack(liquidKeys[i]);
                float fill = grid.GetFill(liquidKeys[i]);
                float diagonal = 0f;
                float divergence = 0f;
                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                    divergence += grid.GetFace(axis, liquidKeys[i]) - grid.GetFace(axis, SparseFluidGrid.Key(pos - offset));

                    for (int side = 0; side < 2; side++)
                    {
                        Point3 np = side == 0 ? pos + offset : pos - offset;
                        long nKey = SparseFluidGrid.Key(np);
                        int type;
                        if (IsSolid(np) || sourceCells.Contains(nKey))
                            type = -2;
                        else if (index.TryGetValue(nKey, out int ni))
                            type = ni;
                        else
                            type = -1;

                        neighbor[i * 6 + axis * 2 + side] = type;
                        if (type >= 0)
                            diagonal += 1f;
                        else if (type == -1)
                        {
                            float nFill = grid.GetFill(nKey);
                            float theta = MathHelper.Clamp((fill - liquidThreshold) / MathF.Max(fill - nFill, 1e-6f), minInterfaceTheta, 1f);
                            airWeight[i * 6 + axis * 2 + side] = 1f / theta;
                            diagonal += 1f / theta;
                        }
                    }
                }
                diag[i] = diagonal > 0f ? diagonal : 1f;
                float targetDivergence = -compactionRate * (1f - fill) / dt;
                b[i] = targetDivergence - divergence;
            }

            double[] x = new double[n];
            for (int i = 0; i < n; i++)
                x[i] = pressure.TryGetValue(liquidKeys[i], out float p) ? p : 0f;

            SolvePressure(neighbor, diag, b, x);

            Dictionary<long, float> newPressure = new Dictionary<long, float>(n);
            for (int i = 0; i < n; i++)
                newPressure[liquidKeys[i]] = (float)x[i];
            pressure = newPressure;

            // u -= p(upper) - p(lower) on every face bordering liquid that isn't solid/source; liquid-liquid faces are handled once, from their lower cell
            for (int i = 0; i < n; i++)
            {
                Point3 pos = SparseFluidGrid.Unpack(liquidKeys[i]);
                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];

                    int up = neighbor[i * 6 + axis * 2];
                    if (up != -2)
                    {
                        double pUp = up >= 0 ? x[up] : 0.0;
                        double weight = up >= 0 ? 1.0 : airWeight[i * 6 + axis * 2];
                        grid.faces[axis][liquidKeys[i]] = grid.GetFace(axis, liquidKeys[i]) - (float)((pUp - x[i]) * weight);
                    }

                    int down = neighbor[i * 6 + axis * 2 + 1];
                    if (down == -1)
                    {
                        long faceKey = SparseFluidGrid.Key(pos - offset);
                        grid.faces[axis][faceKey] = grid.GetFace(axis, faceKey) - (float)(x[i] * airWeight[i * 6 + axis * 2 + 1]);
                    }
                }
            }
        }

        // Below-threshold cells resting directly on liquid or solid take the velocity of the liquid around them on their free faces
        // (zero when resting on solid alone), so gravity can't build up on them; unsupported cells keep their own ballistic velocity
        private void ExtrapolateIntoSupportedAir()
        {
            bool IsFluidCell(long key) => liquidCells.Contains(key) || sourceCells.Contains(key);

            bool IsValidFace(int axis, long faceKey)
            {
                Point3 lower = SparseFluidGrid.Unpack(faceKey);
                Point3 upper = lower + SparseFluidGrid.AxisOffsets[axis];
                if (IsSolid(lower) || IsSolid(upper))
                    return false;
                return IsFluidCell(faceKey) || IsFluidCell(SparseFluidGrid.Key(upper));
            }

            List<(int axis, long faceKey, float value)> updates = new List<(int, long, float)>();
            foreach (long key in grid.SnapshotKeys())
            {
                if (IsFluidCell(key) || grid.GetFill(key) <= 0f)
                    continue;

                Point3 pos = SparseFluidGrid.Unpack(key);
                Point3 below = pos - SparseFluidGrid.AxisOffsets[1];
                if (!liquidCells.Contains(SparseFluidGrid.Key(below)) && !IsSolid(below))
                    continue;

                for (int axis = 0; axis < 3; axis++)
                {
                    Point3 offset = SparseFluidGrid.AxisOffsets[axis];
                    for (int side = 0; side < 2; side++)
                    {
                        long faceKey = side == 0 ? key : SparseFluidGrid.Key(pos - offset);
                        Point3 lower = SparseFluidGrid.Unpack(faceKey);
                        if (IsSolid(lower) || IsSolid(lower + offset) || IsValidFace(axis, faceKey))
                            continue;

                        float sum = 0f;
                        int count = 0;
                        foreach (Point3 o in LatticeNeighbors)
                        {
                            long neighborFace = SparseFluidGrid.Key(lower + o);
                            if (IsValidFace(axis, neighborFace))
                            {
                                sum += grid.GetFace(axis, neighborFace);
                                count++;
                            }
                        }
                        updates.Add((axis, faceKey, count > 0 ? sum / count : 0f));
                    }
                }
            }

            foreach (var (axis, faceKey, value) in updates)
                grid.faces[axis][faceKey] = value;
        }

        // Jacobi-preconditioned conjugate gradient for A p = b, with (A p)_i = diag_i p_i - sum of liquid neighbor p
        private void SolvePressure(int[] neighbor, float[] diag, double[] b, double[] x)
        {
            int n = diag.Length;
            double[] r = new double[n], z = new double[n], s = new double[n], q = new double[n];

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
        private void AdvectFill(float dt)
        {
            long[] keys = grid.SnapshotKeys();
            Dictionary<long, float> currentFill = new Dictionary<long, float>(keys.Length);
            foreach (long key in keys)
                currentFill[key] = grid.GetFill(key);
            float FillOf(long key) => currentFill.TryGetValue(key, out float f) ? f : 0f;

            HashSet<(long lowerKey, int axis)> faceSet = new HashSet<(long, int)>();
            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                for (int axis = 0; axis < 3; axis++)
                {
                    faceSet.Add((key, axis));
                    faceSet.Add((SparseFluidGrid.Key(pos - SparseFluidGrid.AxisOffsets[axis]), axis));
                }
            }

            // Transfers as (donor, receiver, amount > 0)
            List<(long donor, long receiver, float amount)> transfers = new List<(long, long, float)>();
            foreach ((long lowerKey, int axis) in faceSet)
            {
                Point3 lowerPos = SparseFluidGrid.Unpack(lowerKey);
                Point3 upperPos = lowerPos + SparseFluidGrid.AxisOffsets[axis];
                if (IsSolid(lowerPos) || IsSolid(upperPos))
                    continue;

                long upperKey = SparseFluidGrid.Key(upperPos);
                float flux = MathHelper.Clamp(grid.GetFace(axis, lowerKey) * dt, -1f, 1f);
                if (flux > 0f && FillOf(lowerKey) > minFillEpsilon)
                    transfers.Add((lowerKey, upperKey, flux * FillOf(lowerKey)));
                else if (flux < 0f && FillOf(upperKey) > minFillEpsilon)
                    transfers.Add((upperKey, lowerKey, -flux * FillOf(upperKey)));
            }

            Dictionary<long, float> outTotal = new Dictionary<long, float>();
            foreach (var t in transfers)
                outTotal[t.donor] = (outTotal.TryGetValue(t.donor, out float o) ? o : 0f) + t.amount;
            for (int i = 0; i < transfers.Count; i++)
            {
                var t = transfers[i];
                float available = FillOf(t.donor);
                float total = outTotal[t.donor];
                if (total > available)
                    transfers[i] = (t.donor, t.receiver, t.amount * available / total);
            }

            Dictionary<long, float> outScaled = new Dictionary<long, float>();
            Dictionary<long, float> inTotal = new Dictionary<long, float>();
            foreach (var t in transfers)
            {
                outScaled[t.donor] = (outScaled.TryGetValue(t.donor, out float o) ? o : 0f) + t.amount;
                inTotal[t.receiver] = (inTotal.TryGetValue(t.receiver, out float n) ? n : 0f) + t.amount;
            }
            for (int i = 0; i < transfers.Count; i++)
            {
                var t = transfers[i];
                float incoming = inTotal[t.receiver];
                float space = 1f - FillOf(t.receiver) + (outScaled.TryGetValue(t.receiver, out float o) ? o : 0f);
                if (incoming > space)
                    transfers[i] = (t.donor, t.receiver, t.amount * MathF.Max(space, 0f) / incoming);
            }

            Dictionary<long, float> delta = new Dictionary<long, float>();
            void AddDelta(long key, float amount)
            {
                delta.TryGetValue(key, out float existing);
                delta[key] = existing + amount;
            }
            foreach (var t in transfers)
            {
                if (t.amount < minFillEpsilon && FillOf(t.receiver) <= minFillEpsilon)
                    continue; // don't create new cells from negligible amounts
                AddDelta(t.donor, -t.amount);
                AddDelta(t.receiver, t.amount);
            }

            gainedFillThisTick.Clear();
            foreach (KeyValuePair<long, float> kvp in delta)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(kvp.Key);
                float rawFill = cell.fill + kvp.Value;
                if (rawFill > 1f)
                    volumeClampedAway += rawFill - 1f;
                else if (rawFill < 0f)
                    volumeClampCreated += -rawFill;
                cell.fill = MathHelper.Clamp(rawFill, 0f, 1f);
                grid.SetCell(kvp.Key, cell);

                if (kvp.Value > 0f)
                    gainedFillThisTick.Add(kvp.Key);
            }
        }

        // Removes cells inside solid, cells holding (almost) nothing, and dim cells that have stopped gaining fill for too long
        private void PruneStaleCells()
        {
            lastPrunedBySolid = 0;
            lastPrunedByEmpty = 0;

            foreach (long key in grid.SnapshotKeys())
            {
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                if (IsSolid(SparseFluidGrid.Unpack(key)))
                {
                    grid.RemoveKey(key);
                    lastPrunedBySolid++;
                    volumePrunedSolid += cell.fill;
                    continue;
                }

                if (cell.fill <= minFillEpsilon)
                {
                    grid.RemoveKey(key);
                    lastPrunedByEmpty++;
                    volumePrunedEmpty += cell.fill;
                    continue;
                }

                bool dim = cell.fill <= visibilityThreshold;
                bool gained = gainedFillThisTick.Contains(key);
                if (!dim || gained)
                {
                    if (gained && cell.ticksStagnant != 0)
                    {
                        cell.ticksStagnant = 0;
                        grid.SetCell(key, cell);
                    }
                    continue;
                }

                cell.ticksStagnant++;
                if (cell.ticksStagnant >= stagnantTicksBeforeDestroy)
                {
                    grid.RemoveKey(key);
                    lastPrunedByEmpty++;
                    volumePrunedStagnant += cell.fill;
                }
                else
                {
                    grid.SetCell(key, cell);
                }
            }
        }

        private void ApplyDamping(float dt)
        {
            if (velocityDampingPerSecond >= 1f)
                return;

            float factor = MathF.Pow(MathHelper.Clamp(velocityDampingPerSecond, 0f, 1f), dt);
            foreach (Dictionary<long, float> f in grid.faces)
                foreach (long key in new List<long>(f.Keys))
                    f[key] *= factor;
        }

        // No face may carry more than cflMaxFraction of a cell per tick
        private void ClampVelocity(float dt)
        {
            float maxSpeed = cflMaxFraction / dt;
            foreach (Dictionary<long, float> f in grid.faces)
                foreach (long key in new List<long>(f.Keys))
                    f[key] = MathHelper.Clamp(f[key], -maxSpeed, maxSpeed);
        }
    }
}
