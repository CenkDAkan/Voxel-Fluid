using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Render;
using System;

namespace NAADF.World.Data
{
    /*
    * Reproduces the reference master's thesis's dense Eulerian solver as a comparison baseline against FluidHandler's
    * sparse particle system. A fixed-size velocity+density field is advanced by external forces, advection, diffusion, and pressure projection
    * every tick, over every cell in a bounded domain, regardless of how much of that domain is actually fluid
    * Diffusion is applied to both velocity and density, matching the masters thesis's implementation
    */
    public class DenseFluidHandler
    {
        private WorldData worldData;

        // Both share size/origin once placed. grid is the current field, scratch is the buffer whichever step is
        // running writes its result into before swapping, which is reused by both Advect and Diffuse rather than reallocated
        private DenseFluidGrid grid;
        private DenseFluidGrid scratch;

        private uint fluidTypeRenderIndex;

        // Cached per-cell "was this drawn as fluid last apply" state, indexed the same way as the grid itself, 
        // so ApplyToWorld only re-writes cells whose visible state actually flipped instead of the whole domain
        private bool[] wasVisible;

        // Cached per-cell IsBlockedWorld result, recomputed once per Update call rather than re-derived on every lookup
        private bool[] blocked;

        // Pressure field for projection. Kept as its own pair of flat arrays rather than living on DenseFluidGrid
        // Unlike density/velocity, pressure is never advected or sampled at continuous positions, only read/written 
        // by integer-indexed neighbor lookups within Project, so it doesn't need DenseFluidGrid's trilinear-sampling machinery
        // Divergence is fully recomputed each tick before the solve
        private float[] pressure;
        private float[] pressureScratch;
        private float[] divergence;

        // Fixed right-hand side ("b" in thesis Eq 3.13) for Diffuse's Jacobi solve
        private float[] diffuseSourceDensity;
        private Vector3[] diffuseSourceVelocity;

        // The 6 face-neighbor offsets, shared by every full-domain sweep that needs them (Diffuse's neighbor sums)
        // A single static array instead of allocating a fresh one per cell per iteration
        private static readonly Point3[] neighborOffsets =
        {
            new Point3(1, 0, 0), new Point3(-1, 0, 0),
            new Point3(0, 1, 0), new Point3(0, -1, 0),
            new Point3(0, 0, 1), new Point3(0, 0, -1),
        };

        // Same magnitude as FluidHandler's gravity so the two are physically comparable, not just algorithmically
        private Vector3 gravity = new Vector3(0f, -20f, 0f);

        public bool enableGravity = false;

        // How fast density/velocity spread into neighboring cells per second
        private float diffusionRate = 0.01f;

        private const int diffusionIterations = 30;

        // Captured once at seed time
        private float expectedTotalDensityMass;

        private float physicsIntervalMs = 16f;
        private float physicsAccumulatorMs = 0f;

        private float massLogAccumulatorMs = 0f;

        // If a single StepPhysics call ever takes longer than physicsIntervalMs, the accumulator falls further behind every frame, 
        // so the catch-up while loop below runs more and more iterations per frame
        private const int maxPhysicsStepsPerFrame = 4;

        // Density at or above this is drawn as solid fluid, below is empty/air
        private float visibilityThreshold = 0.5f;

        public DenseFluidHandler(WorldData worldData)
        {
            this.worldData = worldData;

            VoxelType denseFluidType = new VoxelType
            {
                ID = "fluid_dense_demo",
                colorBase = new Vector3(0.2f, 1.0f, 0.6f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Emissive,
                materialLayer = MaterialTypeLayer.None,
                roughness = 1.0f,
            };
            fluidTypeRenderIndex = App.worldHandler.voxelTypeHandler.ApplyVoxelType(denseFluidType).renderIndex;
        }

        public void Update(float gameTime)
        {
            HandlePlaceInput();

            if (grid == null)
                return;

            RecomputeBlockedCache();

            LogTotalMass(gameTime);

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
        // K places the domain manually, anchored in front of the camera the same way the sprinkler (J) places its marker
        private void HandlePlaceInput()
        {
            if (grid != null || !IO.KBStates.IsKeyToggleDown(Keys.K))
                return;

            PlaceDomain();
        }

        // Places the domain once, seeded with a small blob of density near the top so there's something for gravity+advection to visibly carry down
        // Unlike G/H/J this only ever fires once, the thesis's grid is set up once, not respawned
        // Called either by HandlePlaceInput (manual K press) or SeedDefaultScenario
        private void PlaceDomain()
        {
            Vector3 camPos = WorldRender.camera.GetPos().toVector3();
            Vector3 camDir = WorldRender.camera.GetDir();

            // Tried 64^3 first, even that was enough to make the engine struggle. The thesis's 120^3 seems out of reach for now
            Point3 domainSize = new Point3(24, 24, 24);

            Vector3 aimPoint = camPos + camDir * 20f;
            Point3 domainXZCenter = Point3.FromVector3(aimPoint);

            uint hitType = worldData.RayTraversal(aimPoint + new Vector3(0, 200, 0), new Vector3(0, -1, 0), out float hitLength, out Point3 hitVoxel, out Point3 hitNormal);
            int domainBottomY = hitType != 0 ? hitVoxel.Y + 1 : (int)aimPoint.Y - 40; // fallback if the ray finds nothing

            Point3 origin = new Point3(domainXZCenter.X - domainSize.X / 2, domainBottomY, domainXZCenter.Z - domainSize.Z / 2);

            grid = new DenseFluidGrid(origin, domainSize);
            scratch = new DenseFluidGrid(origin, domainSize);
            wasVisible = new bool[domainSize.X * domainSize.Y * domainSize.Z];
            blocked = new bool[domainSize.X * domainSize.Y * domainSize.Z];
            pressure = new float[domainSize.X * domainSize.Y * domainSize.Z];
            pressureScratch = new float[domainSize.X * domainSize.Y * domainSize.Z];
            divergence = new float[domainSize.X * domainSize.Y * domainSize.Z];
            diffuseSourceDensity = new float[domainSize.X * domainSize.Y * domainSize.Z];
            diffuseSourceVelocity = new Vector3[domainSize.X * domainSize.Y * domainSize.Z];
            massLogAccumulatorMs = 0f;

            // Same cube radius as FluidHandler.SeedDefaultScenario's SeedBlock, so both modes start from a directly comparable amount of fluid
            SeedBlob(new Point3(domainSize.X / 2, domainSize.Y * 3 / 4, domainSize.Z / 2), 4);
            expectedTotalDensityMass = TotalDensityMass();
            Console.WriteLine($"DenseFluidHandler: placed {domainSize.X}x{domainSize.Y}x{domainSize.Z} domain at world ({origin.X}, {origin.Y}, {origin.Z}).");
        }

        // Called once when this mode is selected from Settings (WorldData.ApplyFluidSimulationMode) instead of
        // waiting for a manual K press, so switching modes always starts from the same reproducible scenario
        // Manual K still works afterward too, though it's a no-op once a domain already exists
        public void SeedDefaultScenario()
        {
            PlaceDomain();
        }

        // Erases every visible cell this handler has drawn back to air and drops the domain entirely, called by
        // WorldData.ApplyFluidSimulationMode before switching away from this mode, so leftover fluid voxels don't
        // linger as ordinary solid terrain once this handler stops being updated
        public void ClearAll()
        {
            if (grid == null)
                return;

            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 local = new Point3(x, y, z);
                        int idx = grid.Index(local);
                        if (wasVisible[idx])
                            WriteCell(grid.LocalToWorld(local), false);
                    }

            worldData.editingHandler.processChunks(false);

            grid = null;
            scratch = null;
            wasVisible = null;
            blocked = null;
            pressure = null;
            pressureScratch = null;
            divergence = null;
            diffuseSourceDensity = null;
            diffuseSourceVelocity = null;
            physicsAccumulatorMs = 0f;
        }

        // Fills the blocked cache for the whole domain, see the field's comment for why this only needs to run
        // once per Update call rather than once per lookup.
        private void RecomputeBlockedCache()
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);
                        blocked[idx] = IsBlockedWorld(grid.LocalToWorld(cell), idx);
                    }
        }

        // Prints total density mass, total velocity magnitude, and the single highest per-cell density once a second
        private void LogTotalMass(float gameTime)
        {
            massLogAccumulatorMs += gameTime;
            if (massLogAccumulatorMs < 1000f)
                return;
            massLogAccumulatorMs -= 1000f;

            float totalSpeed = 0f;
            float peakDensity = 0f;
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        totalSpeed += grid.GetVelocity(cell).Length();
                        peakDensity = Math.Max(peakDensity, grid.GetDensity(cell));
                    }

            Console.WriteLine($"DenseFluidHandler: total density mass = {TotalDensityMass():F3}, total speed = {totalSpeed:F3}, peak density = {peakDensity:F3}");
        }

        private float TotalDensityMass()
        {
            float total = 0f;
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                        total += grid.GetDensity(new Point3(x, y, z));
            return total;
        }

        // Rescales every cell's density so the domain's total mass matches expectedTotalDensityMass which is captured once at seed time
        private void RenormalizeDensity()
        {
            float currentTotal = TotalDensityMass();
            if (currentTotal < 0.001f)
                return; // nothing to rescale, and dividing by 0 would blow up

            float scale = expectedTotalDensityMass / currentTotal;
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        grid.SetDensity(cell, grid.GetDensity(cell) * scale);
                    }
        }

        // Fills a small cube of local-space cells around center with density, giving advection an initial blob to
        // carry instead of starting from an empty field
        private void SeedBlob(Point3 center, int radius)
        {
            for (int x = -radius; x <= radius; x++)
                for (int y = -radius; y <= radius; y++)
                    for (int z = -radius; z <= radius; z++)
                        grid.SetDensity(center + new Point3(x, y, z), 1f);
        }

        // Per-tick pipeline, in the thesis's own chapter order: external forces -> advection -> diffusion -> projection, 
        // plus RenormalizeDensity at the end
        private void StepPhysics(float dt)
        {
            if (enableGravity)
                ApplyExternalForces(dt);
            Advect(dt);
            Diffuse(dt);
            Project();
            RenormalizeDensity();
        }

        // u(t+dt) = u(t) + F*dt (thesis 3.10, same external forces term FluidHandler uses for gravity), applied
        // to every cell in the domain regardless of whether it currently holds any fluid
        // This full-domain sweep, vs FluidHandler's per-particle loop, is the dense side of the comparison
        private void ApplyExternalForces(float dt)
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);

                        // Solid cells never accumulate velocity, because there's no fluid there to move
                        if (blocked[idx])
                            continue;

                        grid.SetVelocity(cell, grid.GetVelocity(cell) + gravity * dt);
                    }
        }

        // For each cell, trace its center backward along the current velocity field to find where its contents came from this step, 
        // then trilinearly sample that source position from the current (pre-advection) field
        private void Advect(float dt)
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);

                        Vector3 sourcePos = ClampToDomain(cell.ToVector3() - grid.GetVelocity(cell) * dt);

                        Vector3 sampledVelocity = grid.SampleVelocity(sourcePos);
                        float sampledDensity = grid.SampleDensity(sourcePos);

                        if (blocked[idx])
                        {
                            sampledVelocity = Vector3.Zero;
                            sampledDensity = 0f;
                        }

                        scratch.SetVelocity(cell, sampledVelocity);
                        scratch.SetDensity(cell, sampledDensity);
                    }

            (grid, scratch) = (scratch, grid); // this tick's result becomes current, old current becomes next tick's scratch buffer
        }

        // Diffusion (thesis 4.1.5): spreads velocity and density into each cell's real face neighbors over time
        // The effect that softens the seeded cube's sharp edges into a rounder blob, distinct from advection's pure
        // transport (advection moves the field, diffusion blurs it)
        //
        // Boundary handling: only real (in-domain, unblocked) neighbors are summed AND counted, and that count,
        // not a fixed 6, is what both the sum and the divisor use
        //
        // The Jacobi right-hand side (diffuseSourceDensity/diffuseSourceVelocity, thesis Eq 3.13's "b") is
        // snapshotted once per call, before the iteration loop, and held fixed across all 30 iterations
        //
        // Known problem: the thesis does not have any cohesion term, so the diffusion step alone will 
        // eventually spread the seeded cube into a uniform low-density haze across the whole domain
        // Thesis' own testing involves continuously seeding new fluid to counteract this, but this implementation does not do that, 
        // so the seeded cube will eventually vanish
        private void Diffuse(float dt)
        {
            float a = diffusionRate * dt;

            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);
                        diffuseSourceDensity[idx] = grid.GetDensity(cell);
                        diffuseSourceVelocity[idx] = grid.GetVelocity(cell);
                    }

            for (int iteration = 0; iteration < diffusionIterations; iteration++)
            {
                for (int x = 0; x < grid.size.X; x++)
                    for (int y = 0; y < grid.size.Y; y++)
                        for (int z = 0; z < grid.size.Z; z++)
                        {
                            Point3 cell = new Point3(x, y, z);
                            int idx = grid.Index(cell);

                            if (blocked[idx])
                            {
                                scratch.SetDensity(cell, 0f);
                                scratch.SetVelocity(cell, Vector3.Zero);
                                continue;
                            }

                            int realNeighborCount = RealNeighborSums(cell, out float densitySum, out Vector3 velocitySum);

                            float newDensity = (diffuseSourceDensity[idx] + a * densitySum) / (1f + realNeighborCount * a);
                            scratch.SetDensity(cell, newDensity);

                            Vector3 newVelocity = (diffuseSourceVelocity[idx] + a * velocitySum) / (1f + realNeighborCount * a);
                            scratch.SetVelocity(cell, newVelocity);
                        }

                (grid, scratch) = (scratch, grid); // this iteration's result becomes current for the next iteration
            }
        }

        // Sums density/velocity over only the neighbors that actually exist (in-domain and not solid) and returns
        // how many that was, for Diffuse's boundary-correct divisor
        private int RealNeighborSums(Point3 cell, out float densitySum, out Vector3 velocitySum)
        {
            densitySum = 0f;
            velocitySum = Vector3.Zero;
            int count = 0;

            foreach (Point3 offset in neighborOffsets)
            {
                Point3 neighbor = cell + offset;
                if (!grid.IsInsideLocal(neighbor))
                    continue;

                int neighborIdx = grid.Index(neighbor);
                if (blocked[neighborIdx])
                    continue;

                densitySum += grid.GetDensity(neighbor);
                velocitySum += grid.GetVelocity(neighbor);
                count++;
            }

            return count;
        }

        // Pressure projection: the velocity field coming out of Diffuse has nonzero divergence in general
        // Projection finds the pressure field whose gradient, subtracted from velocity, cancels that divergence out
        // the step that makes this an incompressible fluid simulation rather than just an advected, diffusing blob of density 
        // with no physical constraint holding its volume together
        private void Project()
        {
            ComputeDivergence();
            SolvePressure();
            SubtractPressureGradient();
        }

        // Divergence of the velocity field: how much each cell's velocity spreads outward vs. converges inward
        // This becomes the source term for the pressure Poisson solve below
        private void ComputeDivergence()
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);

                        if (blocked[idx])
                        {
                            divergence[idx] = 0f;
                            continue;
                        }

                        Vector3 right = BoundaryVelocity(cell, cell + new Point3(1, 0, 0));
                        Vector3 left = BoundaryVelocity(cell, cell + new Point3(-1, 0, 0));
                        Vector3 top = BoundaryVelocity(cell, cell + new Point3(0, 1, 0));
                        Vector3 bottom = BoundaryVelocity(cell, cell + new Point3(0, -1, 0));
                        Vector3 front = BoundaryVelocity(cell, cell + new Point3(0, 0, 1));
                        Vector3 back = BoundaryVelocity(cell, cell + new Point3(0, 0, -1));

                        divergence[idx] = ((right.X - left.X) + (top.Y - bottom.Y) + (front.Z - back.Z)) / 2f;
                    }
        }

        // Solves the Poisson equation for pressure (thesis Eq. 3.6/3.13, with alpha=-1, beta=6 for the pressure
        // case): p_new = (sum of 6 neighboring p values - divergence) / 6. Same Jacobi ping-pong as Diffuse and
        // same iteration count, but no "old self" term the way diffusion has "old velocity"
        // Pressure isn't being transported or decayed, this is pure relaxation toward satisfying the equation
        // pressure/pressureScratch are warm-started (not zeroed here) so each tick's solve starts from last tick's answer
        private void SolvePressure()
        {
            for (int iteration = 0; iteration < diffusionIterations; iteration++)
            {
                for (int x = 0; x < grid.size.X; x++)
                    for (int y = 0; y < grid.size.Y; y++)
                        for (int z = 0; z < grid.size.Z; z++)
                        {
                            Point3 cell = new Point3(x, y, z);
                            int idx = grid.Index(cell);

                            if (blocked[idx])
                            {
                                pressureScratch[idx] = 0f;
                                continue;
                            }

                            float neighborSum =
                                BoundaryPressure(cell, cell + new Point3(1, 0, 0), pressure) + BoundaryPressure(cell, cell + new Point3(-1, 0, 0), pressure) +
                                BoundaryPressure(cell, cell + new Point3(0, 1, 0), pressure) + BoundaryPressure(cell, cell + new Point3(0, -1, 0), pressure) +
                                BoundaryPressure(cell, cell + new Point3(0, 0, 1), pressure) + BoundaryPressure(cell, cell + new Point3(0, 0, -1), pressure);

                            pressureScratch[idx] = (neighborSum - divergence[idx]) / 6f;
                        }

                (pressure, pressureScratch) = (pressureScratch, pressure); // this iteration's result becomes current for the next iteration
            }
        }

        // Subtracts the gradient of the solved pressure field from velocity, the step that actually produces the divergence-free 
        // (incompressible) result projection exists to compute
        private void SubtractPressureGradient()
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);

                        if (blocked[idx])
                            continue;

                        float right = BoundaryPressure(cell, cell + new Point3(1, 0, 0), pressure);
                        float left = BoundaryPressure(cell, cell + new Point3(-1, 0, 0), pressure);
                        float top = BoundaryPressure(cell, cell + new Point3(0, 1, 0), pressure);
                        float bottom = BoundaryPressure(cell, cell + new Point3(0, -1, 0), pressure);
                        float front = BoundaryPressure(cell, cell + new Point3(0, 0, 1), pressure);
                        float back = BoundaryPressure(cell, cell + new Point3(0, 0, -1), pressure);

                        Vector3 gradient = new Vector3(right - left, top - bottom, front - back) / 2f;
                        grid.SetVelocity(cell, grid.GetVelocity(cell) - gradient);
                    }
        }

        // Velocity at "neighbor" as seen from "cell": the real value if the neighbor exists and isn't solid,
        // otherwise a sign-flipped reflection of cell's own velocity (thesis Eq. 4.3: u_N = -u_(N-1))
        // This is what forces velocity to exactly zero AT the wall, rather than merely stopping it from changing across the wall the way Diffuse's boundary does
        private Vector3 BoundaryVelocity(Point3 cell, Point3 neighbor)
        {
            if (!grid.IsInsideLocal(neighbor))
                return -grid.GetVelocity(cell);

            int neighborIdx = grid.Index(neighbor);
            if (blocked[neighborIdx])
                return -grid.GetVelocity(cell);

            return grid.GetVelocity(neighbor);
        }

        // Pressure at "neighbor" as seen from "cell", read from the given field (pressure or pressureScratch): 
        // the real value if the neighbor exists and isn't solid, otherwise a plain (unsigned) mirror of cell's own pressure
        // Same zero-gradient idea as Diffuse's boundary, just expressed as an explicit mirror here, 
        // since pressure's fixed 6-neighbor formula (unlike Diffuse's per-cell neighbor count) expects a value for all 6 directions every time
        private float BoundaryPressure(Point3 cell, Point3 neighbor, float[] pressureField)
        {
            if (!grid.IsInsideLocal(neighbor))
                return pressureField[grid.Index(cell)];

            int neighborIdx = grid.Index(neighbor);
            if (blocked[neighborIdx])
                return pressureField[grid.Index(cell)];

            return pressureField[neighborIdx];
        }

        // Keeps a back-traced sample position inside the range SampleDensity/SampleVelocity can safely read
        private Vector3 ClampToDomain(Vector3 localPos)
        {
            return new Vector3(
                MathHelper.Clamp(localPos.X, 0.001f, grid.size.X - 1.001f),
                MathHelper.Clamp(localPos.Y, 0.001f, grid.size.Y - 1.001f),
                MathHelper.Clamp(localPos.Z, 0.001f, grid.size.Z - 1.001f));
        }

        // A cell blocks fluid motion if its world position is outside the addressable world, 
        // or solid there and not this handler's own fluid. wasVisible (this handler's own live state) is checked before
        // worldData.IsVoxelSolid for the same reason FluidHandler checks fluidGrid.IsFluid first
        // Our own rendered fluid voxels share the same solid render bit as real terrain, and worldData's copy only catches up once
        // ApplyToWorld flushes, so without this a cell would read its own just-flushed fluid as a wall
        private bool IsBlockedWorld(Point3 worldVoxel, int localIndex)
        {
            if (worldVoxel.X < 0 || worldVoxel.Y < 0 || worldVoxel.Z < 0
                || worldVoxel.X >= worldData.sizeInVoxels.X || worldVoxel.Y >= worldData.sizeInVoxels.Y || worldVoxel.Z >= worldData.sizeInVoxels.Z)
                return true;

            if (wasVisible[localIndex])
                return false;

            return worldData.IsVoxelSolid(worldVoxel);
        }

        // Diffs the domain's density-above-threshold state against wasVisible and only re-writes cells whose
        // visible state actually changed. Same dirty-driven write FluidHandler does via its dirtyCells list, just
        // computed by a full-domain scan here since there's no separate list of what moved
        private void ApplyToWorld()
        {
            for (int x = 0; x < grid.size.X; x++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int z = 0; z < grid.size.Z; z++)
                    {
                        Point3 local = new Point3(x, y, z);
                        int idx = grid.Index(local);
                        bool visible = grid.GetDensity(local) >= visibilityThreshold;

                        if (visible != wasVisible[idx])
                        {
                            if (WriteCell(grid.LocalToWorld(local), visible))
                                wasVisible[idx] = visible;
                        }
                    }

            // Stage the edited chunks the same way the editing tools use. ChangeHandler.Update() then uploads everything to the GPU
            worldData.editingHandler.processChunks(false);
        }

        // Translates a world-voxel position into the engine's chunk + in-chunk coordinates and writes one voxel,
        // same address translation as FluidHandler.WriteCell. Returns whether the write actually happened, so
        // ApplyToWorld can avoid recording wasVisible for a write that was silently refused
        private bool WriteCell(Point3 worldVoxel, bool visible)
        {
            if (worldVoxel.X < 0 || worldVoxel.Y < 0 || worldVoxel.Z < 0
                || worldVoxel.X >= worldData.sizeInVoxels.X || worldVoxel.Y >= worldData.sizeInVoxels.Y || worldVoxel.Z >= worldData.sizeInVoxels.Z)
                return false;

            if (visible && worldData.IsVoxelSolid(worldVoxel))
                return false;

            Point3 chunkPos = worldVoxel / 16;         // which chunk contains this voxel, in chunk-grid coordinate
            Point3 voxelPosInChunk = worldVoxel % 16;  // 0..15 on each axis, the voxel's position within the chunk

            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            // Voxel encoding: bit 15 = "solid" flag, low 15 bits = material render index. 0 means empty/air.
            uint type = visible ? (1u << 15) | fluidTypeRenderIndex : 0u;
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, type);
            return true;
        }
    }
}
