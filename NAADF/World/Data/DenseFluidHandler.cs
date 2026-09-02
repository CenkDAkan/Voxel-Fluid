using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Render;
using System;
using System.Threading.Tasks;

namespace NAADF.World.Data
{
    /*
    * Reproduces the reference master's thesis's dense Eulerian solver as a comparison baseline against FluidHandler's
    * sparse particle system. A fixed-size velocity+density field is advanced by external forces, advection, diffusion, and pressure projection
    * every tick, over every cell in a bounded domain, regardless of how much of that domain is actually fluid
    * Diffusion is applied to both velocity and density, matching the masters thesis's implementation
    *
    * Cohesion is a Continuum Surface Force style attraction added on top of the thesis's own pipeline,
    * the thesis has no cohesion term at all, so without it diffusion's spreading is a one-way process with nothing opposing it
    * Cohesion is curvature-driven (pulls concave regions inward, same mechanism that makes a real liquid body relax toward a sphere), 
    * not just "pull toward denser neighbors", see denseFluidSim.fx's Cohesion section for the full derivation and formula
    *
    * The physics pipeline (forces/advect/diffuse/project/renormalize) runs entirely on GPU via denseFluidSim.fx, 
    * mirroring the CPU version's own ping-pong swap points buffer-for-buffer
    * `grid` is kept only as a CPU-side mirror, its density array is refreshed from the GPU once per Update call (via GetData) 
    * so ApplyToWorld/WriteCell can keep working completely unchanged
    */
    public class DenseFluidHandler
    {
        private WorldData worldData;
        private Effect fluidEffect;

        // CPU-side mirror of density, synced from the GPU buffers below after each Update call that stepped physics
        private DenseFluidGrid grid;

        private uint fluidTypeRenderIndex;

        // Cached per-cell "was this drawn as fluid last apply" state, indexed the same way as the grid itself,
        // so ApplyToWorld only re-writes cells whose visible state actually flipped instead of the whole domain
        private bool[] wasVisible;

        // Cached per-cell IsBlockedWorld result, recomputed once per Update call rather than re-derived on every
        // lookup, and uploaded to the GPU (blockedBuffer) each time it changes
        // uint (0/1) rather than bool[] because that's what maps onto a StructuredBuffer<uint> on the shader side
        private uint[] blocked;

        private int cellCount;

        // GPU buffers. Ping-pong pairs (velocity/velocityScratch, density/densityScratch, pressure/pressureScratch)
        // are swapped by reassigning these C# references after each dispatch that writes to the scratch side
        private StructuredBuffer velocityBuffer;
        private StructuredBuffer velocityScratchBuffer;
        private StructuredBuffer densityBuffer;
        private StructuredBuffer densityScratchBuffer;
        private StructuredBuffer pressureBuffer;
        private StructuredBuffer pressureScratchBuffer;
        private StructuredBuffer divergenceBuffer;
        private StructuredBuffer diffuseSourceDensityBuffer;
        private StructuredBuffer diffuseSourceVelocityBuffer;
        private StructuredBuffer blockedBuffer;
        private StructuredBuffer densityPartialSumsBuffer; // one slot per Z slice, for RenormalizeDensity's mass sum

        // Cohesion buffers - densitySmooth/Scratch are a throwaway denoised copy of density, never written back
        private StructuredBuffer densitySmoothBuffer;
        private StructuredBuffer densitySmoothScratchBuffer;
        private StructuredBuffer cohesionGradientBuffer;

        // Curvature, frozen once per tick by ApplyCohesion, reused by every one of Diffuse's 30 Jacobi iterations
        private StructuredBuffer cohesionCurvatureBuffer;

        private float[] densityPartialSumsCpu;
        private Vector4[] velocityLogScratch;

        public float gravityStrength = 20f;

        public bool enableGravity = false;

        // How fast density/velocity spread into neighboring cells per second
        private float diffusionRate = 0.01f;

        private const int diffusionIterations = 30;

        public bool enableCohesion = false;

        // Force strength (sigma in the CSF formula)
        public float cohesionCoefficient = 20f;

        // Density-space term applied inside every one of Diffuse's 30 Jacobi iterations, 
        // gives cohesion the same direct, repeated access to density that diffusion itself has,
        // instead of routing everything through one indirect velocity-mediated Advect sample per tick
        public float cohesionDensityCoefficient = 2f;

        // Only apply the cohesion force where |grad(density)| exceeds this - keeps deep-interior/deep-empty
        // cells (gradient ~0, direction is just noise) from getting a spurious force
        private float cohesionGradientThreshold = 0.05f;

        // How much of smoothDensityIteration's neighbor average to blend in per pass
        public float curvatureSmoothBlend = 0.5f;
        public int curvatureSmoothIterations = 8;

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
        public int domainSize = 24;

        public DenseFluidHandler(WorldData worldData)
        {
            this.worldData = worldData;

            fluidEffect = App.contentManager.Load<Effect>("shaders/fluid/denseFluidSim");

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
            blockedBuffer.SetData(blocked);

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
            {
                // Only density needs to come back. ApplyToWorld only ever reads GetDensity, never velocity
                // GetData writes directly into grid's backing array and DensityArray returns the live reference
                densityBuffer.GetData(grid.DensityArray);
                ApplyToWorld();
            }
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

            Point3 domainDims = new Point3(domainSize, domainSize, domainSize);

            Vector3 aimPoint = camPos + camDir * 20f;
            Point3 domainXZCenter = Point3.FromVector3(aimPoint);

            uint hitType = worldData.RayTraversal(aimPoint + new Vector3(0, 200, 0), new Vector3(0, -1, 0), out float hitLength, out Point3 hitVoxel, out Point3 hitNormal);
            int domainBottomY = hitType != 0 ? hitVoxel.Y + 1 : (int)aimPoint.Y - 40; // fallback if the ray finds nothing

            Point3 origin = new Point3(domainXZCenter.X - domainDims.X / 2, domainBottomY, domainXZCenter.Z - domainDims.Z / 2);

            grid = new DenseFluidGrid(origin, domainDims);
            cellCount = domainDims.X * domainDims.Y * domainDims.Z;

            wasVisible = new bool[cellCount];
            blocked = new uint[cellCount];
            densityPartialSumsCpu = new float[domainDims.Z];
            velocityLogScratch = new Vector4[cellCount];
            massLogAccumulatorMs = 0f;

            SeedBlob(new Point3(domainDims.X / 2, domainDims.Y * 3 / 4, domainDims.Z / 2), 4);

            CreateGpuResources();

            float initialMass = 0f;
            foreach (float d in grid.DensityArray)
                initialMass += d;
            expectedTotalDensityMass = initialMass;

            Console.WriteLine($"DenseFluidHandler: placed {domainDims.X}x{domainDims.Y}x{domainDims.Z} domain at world ({origin.X}, {origin.Y}, {origin.Z}).");
        }

        // Allocates every GPU buffer for the current domain, uploads the seeded density, 
        // zero-initializes everything else, and sets the domain-size uniforms once
        private void CreateGpuResources()
        {
            velocityBuffer = new StructuredBuffer(App.graphicsDevice, typeof(Vector4), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            velocityScratchBuffer = new StructuredBuffer(App.graphicsDevice, typeof(Vector4), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            densityBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            densityScratchBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            pressureBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            pressureScratchBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            divergenceBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            diffuseSourceDensityBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            diffuseSourceVelocityBuffer = new StructuredBuffer(App.graphicsDevice, typeof(Vector4), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            blockedBuffer = new StructuredBuffer(App.graphicsDevice, typeof(uint), cellCount, BufferUsage.None, ShaderAccess.Read);
            densityPartialSumsBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), grid.size.Z, BufferUsage.None, ShaderAccess.ReadWrite);

            densitySmoothBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            densitySmoothScratchBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            cohesionGradientBuffer = new StructuredBuffer(App.graphicsDevice, typeof(Vector4), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);
            cohesionCurvatureBuffer = new StructuredBuffer(App.graphicsDevice, typeof(float), cellCount, BufferUsage.None, ShaderAccess.ReadWrite);

            velocityBuffer.SetData(new Vector4[cellCount]);
            velocityScratchBuffer.SetData(new Vector4[cellCount]);
            densityBuffer.SetData(grid.DensityArray); // seeded by SeedBlob just before this call
            densityScratchBuffer.SetData(new float[cellCount]);
            pressureBuffer.SetData(new float[cellCount]);
            pressureScratchBuffer.SetData(new float[cellCount]);
            densitySmoothBuffer.SetData(new float[cellCount]);
            densitySmoothScratchBuffer.SetData(new float[cellCount]);
            cohesionGradientBuffer.SetData(new Vector4[cellCount]);
            cohesionCurvatureBuffer.SetData(new float[cellCount]);

            fluidEffect.Parameters["sizeX"].SetValue((uint)grid.size.X);
            fluidEffect.Parameters["sizeY"].SetValue((uint)grid.size.Y);
            fluidEffect.Parameters["sizeZ"].SetValue((uint)grid.size.Z);
            fluidEffect.Parameters["cellCount"].SetValue((uint)cellCount);
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

            for (int z = 0; z < grid.size.Z; z++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int x = 0; x < grid.size.X; x++)
                    {
                        Point3 local = new Point3(x, y, z);
                        int idx = grid.Index(local);
                        if (wasVisible[idx])
                            WriteCell(grid.LocalToWorld(local), false);
                    }

            worldData.editingHandler.processChunks(false);

            DisposeGpuResources();

            grid = null;
            wasVisible = null;
            blocked = null;
            densityPartialSumsCpu = null;
            velocityLogScratch = null;
            physicsAccumulatorMs = 0f;
        }

        private void DisposeGpuResources()
        {
            velocityBuffer?.Dispose();
            velocityScratchBuffer?.Dispose();
            densityBuffer?.Dispose();
            densityScratchBuffer?.Dispose();
            pressureBuffer?.Dispose();
            pressureScratchBuffer?.Dispose();
            divergenceBuffer?.Dispose();
            diffuseSourceDensityBuffer?.Dispose();
            diffuseSourceVelocityBuffer?.Dispose();
            blockedBuffer?.Dispose();
            densityPartialSumsBuffer?.Dispose();
            densitySmoothBuffer?.Dispose();
            densitySmoothScratchBuffer?.Dispose();
            cohesionGradientBuffer?.Dispose();
            cohesionCurvatureBuffer?.Dispose();
        }

        // Fills the blocked cache for the whole domain, see the field's comment for why this only needs to run
        // once per Update call rather than once per lookup.
        private void RecomputeBlockedCache()
        {
            Parallel.For(0, grid.size.Z, z =>
            {
                for (int y = 0; y < grid.size.Y; y++)
                    for (int x = 0; x < grid.size.X; x++)
                    {
                        Point3 cell = new Point3(x, y, z);
                        int idx = grid.Index(cell);
                        blocked[idx] = IsBlockedWorld(grid.LocalToWorld(cell), idx) ? 1u : 0u;
                    }
            });
        }

        // Prints total density mass, total velocity magnitude, and the single highest per-cell density once a second
        // Needs a GPU readback of both fields to do this, but only at this 1/sec throttle, so the cost is negligible
        // Can be removed later, as it is not needed for simulation itself, but useful for debugging and future benchmarking
        private void LogTotalMass(float gameTime)
        {
            massLogAccumulatorMs += gameTime;
            if (massLogAccumulatorMs < 1000f)
                return;
            massLogAccumulatorMs -= 1000f;

            densityBuffer.GetData(grid.DensityArray);
            velocityBuffer.GetData(velocityLogScratch);

            float totalSpeed = 0f;
            float peakDensity = 0f;
            float totalMass = 0f;
            for (int i = 0; i < cellCount; i++)
            {
                Vector4 v = velocityLogScratch[i];
                totalSpeed += new Vector3(v.X, v.Y, v.Z).Length();
                float density = grid.DensityArray[i];
                peakDensity = Math.Max(peakDensity, density);
                totalMass += density;
            }

            Console.WriteLine($"DenseFluidHandler: total density mass = {totalMass:F3}, total speed = {totalSpeed:F3}, peak density = {peakDensity:F3}");
        }

        // Fills a small cube of local-space cells around center with density, giving advection an initial blob to
        // carry instead of starting from an empty field. Writes into grid's CPU array directly - this only ever
        // runs once, before CreateGpuResources uploads the result, so there's no GPU buffer to write through yet
        private void SeedBlob(Point3 center, int radius)
        {
            for (int x = -radius; x <= radius; x++)
                for (int y = -radius; y <= radius; y++)
                    for (int z = -radius; z <= radius; z++)
                        grid.SetDensity(center + new Point3(x, y, z), 1f);
        }

        // Per-tick pipeline, in the thesis's own chapter order: external forces -> advection -> diffusion -> projection,
        // plus RenormalizeDensity at the end. Every step below is a GPU dispatch; nothing here reads back to CPU
        // that only happens once, in Update, after this whole accumulator loop finishes for the frame
        private void StepPhysics(float dt)
        {
            int groups = (cellCount + 63) / 64;

            fluidEffect.Parameters["blocked"].SetValue(blockedBuffer);

            if (enableGravity)
                ApplyExternalForces(dt, groups);
            if (enableCohesion)
                ApplyCohesion(dt, groups);
            Advect(dt, groups);
            Diffuse(dt, groups);
            Project(groups);
            RenormalizeDensity(groups);
        }

        // u(t+dt) = u(t) + F*dt (thesis 3.10, same external forces term FluidHandler uses for gravity), applied
        // to every cell in the domain regardless of whether it currently holds any fluid, in place on velocity
        // safe because gravity has no neighbor dependency (every thread only touches its own cell)
        private void ApplyExternalForces(float dt, int groups)
        {
            fluidEffect.Parameters["dt"].SetValue(dt);
            fluidEffect.Parameters["gravity"].SetValue(new Vector3(0f, -gravityStrength, 0f));
            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);

            fluidEffect.Techniques[0].Passes["ApplyExternalForces"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);
        }

        // Curvature-driven cohesion force, added to velocity from this tick's starting density shape, same timing as gravity above
        // Three GPU passes: smooth a throwaway density copy, compute its gradient/normal, then compute curvature and apply the force
        private void ApplyCohesion(float dt, int groups)
        {
            fluidEffect.Parameters["density"].SetValue(densityBuffer);
            fluidEffect.Parameters["densitySmooth"].SetValue(densitySmoothBuffer);
            fluidEffect.Techniques[0].Passes["SnapshotDensityForSmoothing"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);

            fluidEffect.Parameters["curvatureSmoothBlend"].SetValue(curvatureSmoothBlend);
            for (int iteration = 0; iteration < curvatureSmoothIterations; iteration++)
            {
                fluidEffect.Parameters["densitySmooth"].SetValue(densitySmoothBuffer);
                fluidEffect.Parameters["densitySmoothScratch"].SetValue(densitySmoothScratchBuffer);
                fluidEffect.Techniques[0].Passes["SmoothDensityIteration"].ApplyCompute();
                App.graphicsDevice.DispatchCompute(groups, 1, 1);

                (densitySmoothBuffer, densitySmoothScratchBuffer) = (densitySmoothScratchBuffer, densitySmoothBuffer);
            }

            fluidEffect.Parameters["densitySmooth"].SetValue(densitySmoothBuffer);
            fluidEffect.Parameters["cohesionGradient"].SetValue(cohesionGradientBuffer);
            fluidEffect.Techniques[0].Passes["ComputeGradientNormal"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);

            fluidEffect.Parameters["dt"].SetValue(dt);
            fluidEffect.Parameters["cohesionCoefficient"].SetValue(cohesionCoefficient);
            fluidEffect.Parameters["cohesionGradientThreshold"].SetValue(cohesionGradientThreshold);
            fluidEffect.Parameters["cohesionGradient"].SetValue(cohesionGradientBuffer);
            fluidEffect.Parameters["cohesionCurvature"].SetValue(cohesionCurvatureBuffer);
            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
            fluidEffect.Techniques[0].Passes["ApplyCohesionForce"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);
        }

        // For each cell, trace its center backward along the current velocity field to find where its contents came from this step,
        // then trilinearly sample that source position from the current (pre-advection) field
        // Writes to *Scratch (never in place) since a backtraced sample can land on any cell in the domain,
        // including one another thread hasn't updated yet this dispatch, then swaps the C# buffer references
        private void Advect(float dt, int groups)
        {
            fluidEffect.Parameters["dt"].SetValue(dt);
            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
            fluidEffect.Parameters["density"].SetValue(densityBuffer);
            fluidEffect.Parameters["velocityScratch"].SetValue(velocityScratchBuffer);
            fluidEffect.Parameters["densityScratch"].SetValue(densityScratchBuffer);

            fluidEffect.Techniques[0].Passes["Advect"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);

            (velocityBuffer, velocityScratchBuffer) = (velocityScratchBuffer, velocityBuffer);
            (densityBuffer, densityScratchBuffer) = (densityScratchBuffer, densityBuffer);
        }

        // Diffusion: spreads velocity and density into each cell's real face neighbors over time.
        // Snapshots the pre-iteration field as the Jacobi right-hand side once, then runs diffusionIterations relaxation passes
        //
        // The thesis has no cohesion term, so this step alone eventually spreads a resting blob into a uniform low-density haze
        private void Diffuse(float dt, int groups)
        {
            fluidEffect.Parameters["dt"].SetValue(dt);
            fluidEffect.Parameters["cohesionDensityCoefficient"].SetValue(enableCohesion ? cohesionDensityCoefficient : 0f);
            fluidEffect.Parameters["cohesionCurvature"].SetValue(cohesionCurvatureBuffer);

            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
            fluidEffect.Parameters["density"].SetValue(densityBuffer);
            fluidEffect.Parameters["diffuseSourceVelocity"].SetValue(diffuseSourceVelocityBuffer);
            fluidEffect.Parameters["diffuseSourceDensity"].SetValue(diffuseSourceDensityBuffer);
            fluidEffect.Techniques[0].Passes["SnapshotDiffuseSource"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);

            float a = diffusionRate * dt;
            fluidEffect.Parameters["diffusionA"].SetValue(a);

            for (int iteration = 0; iteration < diffusionIterations; iteration++)
            {
                fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
                fluidEffect.Parameters["density"].SetValue(densityBuffer);
                fluidEffect.Parameters["velocityScratch"].SetValue(velocityScratchBuffer);
                fluidEffect.Parameters["densityScratch"].SetValue(densityScratchBuffer);

                fluidEffect.Techniques[0].Passes["DiffuseIteration"].ApplyCompute();
                App.graphicsDevice.DispatchCompute(groups, 1, 1);

                (velocityBuffer, velocityScratchBuffer) = (velocityScratchBuffer, velocityBuffer);
                (densityBuffer, densityScratchBuffer) = (densityScratchBuffer, densityBuffer);
            }
        }

        // Pressure projection: the velocity field coming out of Diffuse has nonzero divergence in general.
        // Projection finds the pressure field whose gradient, subtracted from velocity, cancels that divergence out
        private void Project(int groups)
        {
            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
            fluidEffect.Parameters["divergence"].SetValue(divergenceBuffer);
            fluidEffect.Techniques[0].Passes["ComputeDivergence"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);

            // pressure/pressureScratch are warm-started (never cleared) so each tick's solve starts from last tick's answer, same as the CPU version
            // Outer loop stays sequential, each iteration reads the previous iteration's swapped-in pressure field
            for (int iteration = 0; iteration < diffusionIterations; iteration++)
            {
                fluidEffect.Parameters["pressure"].SetValue(pressureBuffer);
                fluidEffect.Parameters["pressureScratch"].SetValue(pressureScratchBuffer);
                fluidEffect.Parameters["divergence"].SetValue(divergenceBuffer);

                fluidEffect.Techniques[0].Passes["SolvePressureIteration"].ApplyCompute();
                App.graphicsDevice.DispatchCompute(groups, 1, 1);

                (pressureBuffer, pressureScratchBuffer) = (pressureScratchBuffer, pressureBuffer);
            }

            fluidEffect.Parameters["velocity"].SetValue(velocityBuffer);
            fluidEffect.Parameters["pressure"].SetValue(pressureBuffer);
            fluidEffect.Techniques[0].Passes["SubtractPressureGradient"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);
        }

        // Rescales every cell's density so the domain's total mass matches expectedTotalDensityMass, captured once at seed time
        private void RenormalizeDensity(int groups)
        {
            int zGroups = (grid.size.Z + 63) / 64;

            fluidEffect.Parameters["density"].SetValue(densityBuffer);
            fluidEffect.Parameters["densityPartialSums"].SetValue(densityPartialSumsBuffer);
            fluidEffect.Techniques[0].Passes["SumDensitySlice"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(zGroups, 1, 1);

            densityPartialSumsBuffer.GetData(densityPartialSumsCpu);
            float currentTotal = 0f;
            for (int i = 0; i < densityPartialSumsCpu.Length; i++)
                currentTotal += densityPartialSumsCpu[i];

            if (currentTotal < 0.001f)
                return; // nothing to rescale, and dividing by 0 would blow up

            float scale = expectedTotalDensityMass / currentTotal;
            fluidEffect.Parameters["renormalizeScale"].SetValue(scale);
            fluidEffect.Parameters["density"].SetValue(densityBuffer);
            fluidEffect.Techniques[0].Passes["RescaleDensity"].ApplyCompute();
            App.graphicsDevice.DispatchCompute(groups, 1, 1);
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
        // computed by a full-domain scan here since there's no separate list of what moved.
        // Reads grid.GetDensity, which Update() refreshes from the GPU via GetData right before calling this
        private void ApplyToWorld()
        {
            for (int z = 0; z < grid.size.Z; z++)
                for (int y = 0; y < grid.size.Y; y++)
                    for (int x = 0; x < grid.size.X; x++)
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
