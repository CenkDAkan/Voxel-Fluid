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
    * Sparse Eulerian fluid solver - Volume-of-Fluid (fill fraction per cell)
    *
    * fill is never resampled. It's moved directly between adjacent cells as a face flux (AdvectFill),
    * whatever leaves one cell is exactly what enters its neighbor, so total
    * fluid volume is conserved by construction, not by a periodic correction pass bolted on afterward
    *
    * Only cells that actually contain fluid (fill > 0) are tracked, at all. A cell simply exists for as long
    * as it holds any fluid and is removed the moment it doesn't (PruneStaleCells), and a neighbor that isn't
    * tracked unambiguously means "no fluid there"
    */
    public class SparseFluidHandler
    {
        private WorldData worldData;
        private SparseFluidGrid grid;

        private uint fluidTypeRenderIndex;

        // This handler's own drawn-as-fluid state, keyed the same way as grid. Doubles as ApplyToWorld's dirty-diff
        // baseline and as IsSolid's "is this actually our own fluid, not real terrain" check
        private HashSet<long> wasFluidVoxel = new HashSet<long>();

        // Pressure is warm-started across ticks (not cleared each StepPhysics call) so each tick's Jacobi solve
        // starts from last tick's answer, same reasoning as DenseFluidHandler's pressure/pressureScratch buffers
        private Dictionary<long, float> pressure = new Dictionary<long, float>();

        private static readonly Point3[] NeighborOffsets =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        private static readonly Point3[] PositiveAxisOffsets =
        {
            new Point3(1, 0, 0), new Point3(0, 1, 0), new Point3(0, 0, 1),
        };

        private const int jacobiIterations = 30; // matches DenseFluidHandler's diffusionIterations, kept the same per project convention

        // A cell below this is treated as empty for rendering purposes
        // it can still hold a small amount of fluid and be tracked/simulated, just not drawn as a solid voxel yet.
        private const float visibilityThreshold = 0.5f;

        // Nothing below this magnitude is treated as real fluid, anywhere in this class. AdvectFill refuses to
        // create/grow a cell by less than it, ApplyExternalForces skips it
        // Without this, a cell sitting at some vanishingly small but strictly-positive fill
        // still counted as tracked fluid: still got full gravity every tick, still transferred an even-tinier nonzero amount to its own neighbors, 
        // which then did the same the tick after a "ghost" cloud with real per-cell computational cost but no visible presence at all
        private const float minFillEpsilon = 0.001f;

        // Hard speed cap, applied to every tracked cell's velocity once per tick (ClampVelocity): no
        // cell may imply moving more than this fraction of a cell's width per tick, in any direction
        private const float cflMaxFraction = 0.5f;

        // Populated by AdvectFill every tick (cleared and rebuilt each call, not cumulative) with every key whose
        // fill increased this tick - see PruneStaleCells' own comment for why this exists
        private readonly HashSet<long> gainedFillThisTick = new HashSet<long>();

        public bool enableGravity = true;
        public float gravityStrength = 20f;

        // Fraction of velocity retained per second (1 = no damping, lower = more friction/viscosity)
        // Pressure projection only removes divergence. Nothing else in this pipeline dissipates kinetic energy, so without this a
        // settled puddle can slide/slosh along the floor indefinitely instead of coming to rest 
        // A cheap uniform damping term, not a real viscous diffusion solve
        public float velocityDampingPerSecond = 0.3f;

        // How many CONSECUTIVE ticks a cell must sit at or below visibilityThreshold without gaining any fill
        // before PruneStaleCells judges it genuinely abandoned and destroys it
        public int stagnantTicksBeforeDestroy = 30;

        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;
        private const int maxPhysicsStepsPerFrame = 4; // same spiral-of-death guard DenseFluidHandler needed after hitting it live; built in from the start here

        // Ticks elapsed since the last seed, and whether we've already logged first solid contact this seed -
        // both drive LogTick's finer-grained diagnostic window
        private int ticksSinceSeed = 0;
        private bool loggedFirstContact = false;

        private int lastPrunedBySolid = 0;
        private int lastPrunedByEmpty = 0;
        private int? lastKnownFloorY = null;

        public SparseFluidHandler(WorldData worldData)
        {
            this.worldData = worldData;
            grid = new SparseFluidGrid();

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
        }

        public void Update(float gameTime)
        {
            HandlePlaceInput();

            if (grid.Count == 0)
                return;

            physicsAccumulatorMs += gameTime;
            bool stepped = false;
            int stepsThisFrame = 0;
            while (physicsAccumulatorMs >= physicsIntervalMs && stepsThisFrame < maxPhysicsStepsPerFrame)
            {
                physicsAccumulatorMs -= physicsIntervalMs;
                StepPhysics(physicsIntervalMs / 1000f);
                stepped = true;
                stepsThisFrame++;
                if (physicsIntervalMs <= 0f) break; // avoid an infinite loop if physicsIntervalMs is set to 0
            }
            if (stepsThisFrame >= maxPhysicsStepsPerFrame)
                physicsAccumulatorMs = 0f; // drop the backlog rather than let it compound into next frame's catch-up

            if (stepped)
                ApplyToWorld();
        }

        // P drops a seed sphere aimed from the camera, same idea as FluidHandler's G but sized like SeedDefaultScenario
        private void HandlePlaceInput()
        {
            if (!IO.KBStates.IsKeyToggleDown(Keys.P))
                return;

            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            SeedSphere(Point3.FromVector3(camPos + camDir * 20f), 4);
        }

        // Called once when this mode is selected from Settings
        // Finds the floor below the aim point and seeds a few voxels above it, so
        // gravity has to actually carry the fluid down and settle it
        // The point is testing rest stability after motion, not starting already at rest
        public void SeedDefaultScenario()
        {
            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();
            Vector3 aimPoint = camPos + camDir * 20f;

            uint hitType = worldData.RayTraversal(aimPoint + new Vector3(0, 200, 0), new Vector3(0, -1, 0), out float hitLength, out Point3 hitVoxel, out Point3 hitNormal);
            int floorY = hitType != 0 ? hitVoxel.Y + 1 : (int)aimPoint.Y - 40;
            lastKnownFloorY = floorY;

            Point3 seedCenter = new Point3((int)aimPoint.X, floorY + 6, (int)aimPoint.Z);
            SeedSphere(seedCenter, 4);
        }

        // Fills a sphere directly with fill-fraction data, no separate reconstruction pass needed
        // Any cell that lands on/inside solid ground is dropped immediately
        // AdvectFill's own IsSolid gate would never let a fresh cell exist there anyway, so this
        // just avoids seeding something that would vanish on the very first tick regardless
        private void SeedSphere(Point3 center, int radius)
        {
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

                        grid.Set(pos, new SparseFluidGrid.Cell { fill = fill, velocity = Vector3.Zero });
                    }

            ticksSinceSeed = 0;
            loggedFirstContact = false;
            Console.WriteLine($"SparseFluidHandler: seeded {grid.Count} fluid cells around ({center.X}, {center.Y}, {center.Z}).");
        }

        // Erases every cell this handler has drawn back to air and drops all tracked state, called by
        // WorldData.ApplyFluidSimulationMode before switching away from this mode
        public void ClearAll()
        {
            if (wasFluidVoxel.Count == 0 && grid.Count == 0)
                return;

            foreach (long key in wasFluidVoxel)
                WriteCell(SparseFluidGrid.Unpack(key), false);
            wasFluidVoxel.Clear();

            worldData.editingHandler.processChunks(false);

            grid.Clear();
            pressure.Clear();
            physicsAccumulatorMs = 0f;
        }

        private void StepPhysics(float dt)
        {
            if (enableGravity)
                ApplyExternalForces(dt);

            AdvectVelocity(dt);
            AdvectFill(dt);
            PruneStaleCells();

            ProjectPressure();
            ApplyDamping(dt);
            ClampVelocity(dt);

            ticksSinceSeed++;
            LogTick();
        }

        // Applied last, after damping, so it's a hard final bound on whatever velocity the tick actually ends with, 
        // what AdvectFill reads at the START of next tick is guaranteed to already respect the CFL limit
        private void ClampVelocity(float dt)
        {
            float maxSpeed = cflMaxFraction / dt;
            long[] keys = grid.SnapshotKeys();
            foreach (long key in keys)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                float speed = cell.velocity.Length();
                if (speed > maxSpeed)
                {
                    cell.velocity *= maxSpeed / speed;
                    grid.SetCell(key, cell);
                }
            }
        }

        // Applied after ProjectPressure so it acts on the divergence-corrected velocity, not a value the projection is about to
        // overwrite anyway. Exponential (Pow, not a flat per-tick multiply) so the retained fraction stays meaningful regardless of physicsIntervalMs
        private void ApplyDamping(float dt)
        {
            if (velocityDampingPerSecond >= 1f)
                return;

            float factor = MathF.Pow(MathHelper.Clamp(velocityDampingPerSecond, 0f, 1f), dt);
            long[] keys = grid.SnapshotKeys();
            foreach (long key in keys)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                cell.velocity *= factor;
                grid.SetCell(key, cell);
            }
        }

        // Drops any tracked cell that's stagnant (not gaining fill this tick) and at or below visibilityThreshold,
        // or that has become solid. No growth happens here, this is purely a cleanup pass
        //
        // DELIBERATE DESIGN CHOICE: a cell that's about to become invisible AND isn't actively filling up 
        // is destroyed outright rather than kept around as tracked-but-unrendered residue
        // it's the same practical tradeoff Minecraft/Terraria-style voxel fluids already make (their
        // water sources also delete themselves past a spread distance, not "conserve" a diminishing puddle)
        // On an open floor, a genuinely mass-conservative design has no equilibrium to settle into,
        // motion never fully decays to exact zero, so AdvectFill's diffusive donor-cell scheme keeps thinning 
        // and spreading the SAME finite volume forever, and every one of those
        // thinning cells still costs a full tick of simulation even once nothing is visible.
        //
        private void PruneStaleCells()
        {
            lastPrunedBySolid = 0;
            lastPrunedByEmpty = 0;

            long[] keys = grid.SnapshotKeys();
            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                if (IsSolid(pos))
                {
                    grid.RemoveKey(key);
                    lastPrunedBySolid++;
                    continue;
                }

                SparseFluidGrid.Cell cell = grid.GetCell(key);
                if (cell.fill <= minFillEpsilon)
                {
                    grid.RemoveKey(key);
                    lastPrunedByEmpty++;
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
                    continue; // visible, or still actively filling up - not a stagnation candidate at all
                }

                cell.ticksStagnant++;
                if (cell.ticksStagnant >= stagnantTicksBeforeDestroy)
                {
                    grid.RemoveKey(key);
                    lastPrunedByEmpty++;
                }
                else
                {
                    grid.SetCell(key, cell);
                }
            }
        }

        // Only genuinely filled (fill > minFillEpsilon), non-solid cells get a body force. Matches ProjectPressure's
        // own exclusion of empty/solid cells from incompressibility correction. The epsilon (not a literal > 0f)
        // matters here specifically: a "ghost" cell sitting at some vanishingly small fill left over from
        // AdvectFill would otherwise keep receiving full gravity forever, growing velocity with no real fluid
        // behind it
        private void ApplyExternalForces(float dt)
        {
            Vector3 gravity = new Vector3(0f, -gravityStrength, 0f);
            long[] keys = grid.SnapshotKeys();
            foreach (long key in keys)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                if (cell.fill <= minFillEpsilon || IsSolid(SparseFluidGrid.Unpack(key)))
                    continue;

                cell.velocity += gravity * dt;
                grid.SetCell(key, cell);
            }
        }

        // Semi-Lagrangian back-trace: each cell's new velocity is sampled from where its content came from this step
        // Written to a scratch dictionary first since a back-traced sample can land on any
        // cell in the grid, including one this same pass hasn't updated yet. Velocity is still fine to resample
        // this way, losing a little accuracy in velocity doesn't make matter vanish
        private void AdvectVelocity(float dt)
        {
            long[] keys = grid.SnapshotKeys();
            Dictionary<long, Vector3> scratch = new Dictionary<long, Vector3>(keys.Length);
            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                Vector3 velocity = grid.GetCell(key).velocity;
                Vector3 backPos = pos.ToVector3() - velocity * dt;
                scratch[key] = grid.SampleVelocity(backPos);
            }

            foreach (KeyValuePair<long, Vector3> kvp in scratch)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(kvp.Key);
                cell.velocity = kvp.Value;
                grid.SetCell(kvp.Key, cell);
            }
        }

        // Moves fill directly between adjacent cells via face fluxes (donor-cell/upwind)
        // Whatever amount leaves one side of a face is exactly what enters the other
        // Solid cells never donate or receive. All deltas for this tick are accumulated in a scratch dictionary first (never applied
        // in place) since a single cell can be touched by several different face computations within the same
        // tick, including cells not in the original snapshot at all
        private void AdvectFill(float dt)
        {
            long[] keys = grid.SnapshotKeys();
            Dictionary<long, float> currentFill = new Dictionary<long, float>(keys.Length);
            foreach (long key in keys)
                currentFill[key] = grid.GetCell(key).fill;

            HashSet<(long lowerKey, int axisIndex)> faces = new HashSet<(long, int)>();
            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                for (int axisIndex = 0; axisIndex < 3; axisIndex++)
                {
                    Point3 axis = PositiveAxisOffsets[axisIndex];
                    faces.Add((key, axisIndex));                                          // this cell is the lower side of its own +axis face
                    faces.Add((SparseFluidGrid.Key(pos - axis), axisIndex));               // this cell is the upper side of its -axis neighbor's face
                }
            }

            Dictionary<long, float> delta = new Dictionary<long, float>();
            void AddDelta(long key, float amount)
            {
                delta.TryGetValue(key, out float existing);
                delta[key] = existing + amount;
            }

            foreach ((long lowerKey, int axisIndex) in faces)
            {
                Point3 lowerPos = SparseFluidGrid.Unpack(lowerKey);
                if (IsSolid(lowerPos))
                    continue;

                Point3 axis = PositiveAxisOffsets[axisIndex];
                Point3 upperPos = lowerPos + axis;
                if (IsSolid(upperPos))
                    continue;

                float lowerFill = currentFill.TryGetValue(lowerKey, out float lf) ? lf : 0f;
                long upperKey = SparseFluidGrid.Key(upperPos);
                float upperFill = currentFill.TryGetValue(upperKey, out float uf) ? uf : 0f;
                if (lowerFill <= minFillEpsilon && upperFill <= minFillEpsilon)
                    continue; // no MEANINGFUL fluid on either side of this face - nothing worth moving

                Vector3 lowerVelocity = grid.GetVelocity(lowerPos);
                Vector3 upperVelocity = grid.GetVelocity(upperPos);
                float faceVelocity = (AxisComponent(lowerVelocity, axis) + AxisComponent(upperVelocity, axis)) * 0.5f;
                float flux = MathHelper.Clamp(faceVelocity * dt, -1f, 1f); // clamp: a face shouldn't move more than one full cell's worth in a tick

                float amount;
                if (flux > 0f)
                    amount = MathF.Min(flux * lowerFill, 1f - upperFill); // lower -> upper, capped so upper doesn't overfill
                else if (flux < 0f)
                    amount = -MathF.Min(-flux * upperFill, 1f - lowerFill); // upper -> lower, capped so lower doesn't overfill
                else
                    continue;

                // a transfer smaller than this is refused outright, so a fresh cell is never created (and an existing one's fill
                // never nudged) by a negligible amount in the first place, PruneStaleCells' epsilon alone would
                // eventually clean up a ghost cell that already exists, but doesn't stop new ones from being born
                // every tick at the diffuse edge of the advection front
                if (MathF.Abs(amount) < minFillEpsilon)
                    continue;

                AddDelta(lowerKey, -amount);
                AddDelta(upperKey, amount);
            }

            gainedFillThisTick.Clear();
            foreach (KeyValuePair<long, float> kvp in delta)
            {
                SparseFluidGrid.Cell cell = grid.GetCell(kvp.Key); // zeroed default for a brand-new key
                cell.fill = MathHelper.Clamp(cell.fill + kvp.Value, 0f, 1f); // clamp at the source, same lesson as the dense arm's negative-density clamp
                grid.SetCell(kvp.Key, cell);

                // A cell mid-accumulation (still gaining fill this tick, whether brand new or already existing) must not be judged "stagnant" and
                // destroyed for sitting below visibilityThreshold before it's had a real chance to fill up.
                if (kvp.Value > 0f)
                    gainedFillThisTick.Add(kvp.Key);
            }
        }

        private static float AxisComponent(Vector3 v, Point3 axis) => axis.X != 0 ? v.X : (axis.Y != 0 ? v.Y : v.Z);

        // Central-difference divergence/pressure solve: a neighbor outside the fluid
        // contributes zero pressure rather than mirroring back the cell's own value. 
        // Solid geometry still mirrors (zero-normal-velocity wall)
        private void ProjectPressure()
        {
            long[] keys = grid.SnapshotKeys();

            Dictionary<long, float> divergence = new Dictionary<long, float>(keys.Length);
            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                divergence[key] = (cell.fill <= 0f || IsSolid(pos)) ? 0f : ComputeDivergence(pos, cell.velocity);
            }

            for (int iteration = 0; iteration < jacobiIterations; iteration++)
            {
                Dictionary<long, float> next = new Dictionary<long, float>(keys.Length);
                foreach (long key in keys)
                {
                    Point3 pos = SparseFluidGrid.Unpack(key);
                    SparseFluidGrid.Cell cell = grid.GetCell(key);

                    if (cell.fill <= 0f || IsSolid(pos))
                    {
                        next[key] = 0f;
                        continue;
                    }

                    float ownPressure = pressure.TryGetValue(key, out float p) ? p : 0f;
                    float neighborSum =
                        NeighborPressure(ownPressure, pos + new Point3(1, 0, 0)) + NeighborPressure(ownPressure, pos + new Point3(-1, 0, 0)) +
                        NeighborPressure(ownPressure, pos + new Point3(0, 1, 0)) + NeighborPressure(ownPressure, pos + new Point3(0, -1, 0)) +
                        NeighborPressure(ownPressure, pos + new Point3(0, 0, 1)) + NeighborPressure(ownPressure, pos + new Point3(0, 0, -1));

                    next[key] = (neighborSum - divergence[key]) / 6f;
                }
                pressure = next;
            }

            foreach (long key in keys)
            {
                Point3 pos = SparseFluidGrid.Unpack(key);
                SparseFluidGrid.Cell cell = grid.GetCell(key);
                if (cell.fill <= 0f || IsSolid(pos))
                    continue;

                float ownPressure = pressure.TryGetValue(key, out float p) ? p : 0f;
                float right = NeighborPressure(ownPressure, pos + new Point3(1, 0, 0));
                float left = NeighborPressure(ownPressure, pos + new Point3(-1, 0, 0));
                float top = NeighborPressure(ownPressure, pos + new Point3(0, 1, 0));
                float bottom = NeighborPressure(ownPressure, pos + new Point3(0, -1, 0));
                float front = NeighborPressure(ownPressure, pos + new Point3(0, 0, 1));
                float back = NeighborPressure(ownPressure, pos + new Point3(0, 0, -1));

                Vector3 gradient = new Vector3(right - left, top - bottom, front - back) / 2f;
                cell.velocity -= gradient;
                grid.SetCell(key, cell);
            }
        }

        private float ComputeDivergence(Point3 pos, Vector3 velocity)
        {
            Vector3 right = NeighborVelocity(velocity, pos + new Point3(1, 0, 0));
            Vector3 left = NeighborVelocity(velocity, pos + new Point3(-1, 0, 0));
            Vector3 top = NeighborVelocity(velocity, pos + new Point3(0, 1, 0));
            Vector3 bottom = NeighborVelocity(velocity, pos + new Point3(0, -1, 0));
            Vector3 front = NeighborVelocity(velocity, pos + new Point3(0, 0, 1));
            Vector3 back = NeighborVelocity(velocity, pos + new Point3(0, 0, -1));

            return ((right.X - left.X) + (top.Y - bottom.Y) + (front.Z - back.Z)) / 2f;
        }

        // A solid neighbor mirrors the cell's own velocity negated
        // A neighbor that's tracked and actually filled reads its real value
        // An empty/untracked neighbor, unambiguously atmosphere now extrapolates the cell's own velocity rather than defaulting to zero, so divergence
        // isn't artificially inflated right at the free surface (the standard treatment, e.g. Bridson's "Fluid
        // Simulation for Computer Graphics")
        private Vector3 NeighborVelocity(Vector3 ownVelocity, Point3 neighborPos)
        {
            if (IsSolid(neighborPos))
                return -ownVelocity;

            long nKey = SparseFluidGrid.Key(neighborPos);
            if (grid.TryGetCell(nKey, out SparseFluidGrid.Cell nCell) && nCell.fill > 0f)
                return nCell.velocity;

            return ownVelocity;
        }

        // A solid neighbor mirrors the cell's own pressure. A neighbor that's empty or simply untracked is unambiguously atmosphere in this
        // design. It gets the real free-surface treatment, zero pressure
        private float NeighborPressure(float ownPressure, Point3 neighborPos)
        {
            if (IsSolid(neighborPos))
                return ownPressure;

            long nKey = SparseFluidGrid.Key(neighborPos);
            if (grid.TryGetCell(nKey, out SparseFluidGrid.Cell nCell) && nCell.fill > 0f)
                return pressure.TryGetValue(nKey, out float p) ? p : 0f;

            return 0f;
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

        // Diffs fill > visibilityThreshold against wasFluidVoxel and only re-writes cells whose visible state
        // actually changed. Also erases any cell that was drawn as fluid but has since left tracking entirely (via PruneStaleCells)
        private void ApplyToWorld()
        {
            long[] keys = grid.SnapshotKeys();
            HashSet<long> nowFluid = new HashSet<long>();
            foreach (long key in keys)
                if (grid.GetCell(key).fill > visibilityThreshold)
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

            // Stage the edited chunks the same way the editing tools use. ChangeHandler.Update() then uploads everything to the GPU
            worldData.editingHandler.processChunks(false);
        }

        // Same address translation + solid-guard + return-bool pattern as FluidHandler/DenseFluidHandler.WriteCell
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
        private void LogTick()
        {
            int visibleCellCount = 0;
            float totalSpeed = 0f;
            bool contactThisTick = false;
            int minY = int.MaxValue, maxY = int.MinValue;

            foreach (long key in grid.SnapshotKeys())
            {
                Point3 trackedPos = SparseFluidGrid.Unpack(key);
                if (trackedPos.Y < minY) minY = trackedPos.Y;
                if (trackedPos.Y > maxY) maxY = trackedPos.Y;

                SparseFluidGrid.Cell cell = grid.GetCell(key);
                totalSpeed += cell.velocity.Length();
                if (cell.fill <= visibilityThreshold)
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
            Console.WriteLine($"SparseFluidHandler: tick {ticksSinceSeed} ({ticksSinceSeed * physicsIntervalMs / 1000f:F2}s), visible cells = {visibleCellCount}, tracked cells = {grid.Count}, total speed = {totalSpeed:F3}, pruned this tick (solid/empty) = {lastPrunedBySolid}/{lastPrunedByEmpty}{bandYInfo}{floorInfo}");
        }
    }
}
