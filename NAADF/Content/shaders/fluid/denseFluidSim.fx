// Dense Eulerian fluid simulation: GPU port of DenseFluidHandler.cs's per-tick pipeline
// (external forces -> advection -> diffusion -> projection -> density renormalization).
//
// Buffer design mirrors the CPU version's ping-pong swap points exactly. only Advect, DiffuseIteration and
// SolvePressureIteration read a *neighboring* cell that another thread could be writing this same dispatch, so
// only those write to a *Scratch buffer (double-buffered, swapped from C# between calls)
// ApplyExternalForces, ComputeDivergence and SubtractPressureGradient only ever touch their own cell (or a different, already-stable buffer),
// so they write in place

uint sizeX, sizeY, sizeZ;
uint cellCount; // sizeX * sizeY * sizeZ

float dt;
float3 gravity;
float diffusionA; // diffusionRate * dt
float renormalizeScale;

RWStructuredBuffer<float4> velocity;         // .xyz used, .w unused - no float3 structured buffer precedent in this engine
RWStructuredBuffer<float4> velocityScratch;
RWStructuredBuffer<float> density;
RWStructuredBuffer<float> densityScratch;

RWStructuredBuffer<float> pressure;
RWStructuredBuffer<float> pressureScratch;
RWStructuredBuffer<float> divergence;

RWStructuredBuffer<float> diffuseSourceDensity;
RWStructuredBuffer<float4> diffuseSourceVelocity;

StructuredBuffer<uint> blocked; // CPU-uploaded each Update via RecomputeBlockedCache, read-only from every kernel

RWStructuredBuffer<float> densityPartialSums; // one slot per Z slice, for the renormalization mass sum

// Cohesion (see the Cohesion section below for the full pipeline)
RWStructuredBuffer<float> densitySmooth;         // throwaway smoothed copy of density, used only for curvature estimation
RWStructuredBuffer<float> densitySmoothScratch;
RWStructuredBuffer<float4> cohesionGradient;     // xyz = unit normal n-hat, w = |grad(density)| magnitude
RWStructuredBuffer<float> cohesionCurvature;     // frozen once per tick, reused by every diffuseIteration pass below

float cohesionCoefficient;   
float cohesionDensityCoefficient; 
float cohesionGradientThreshold;  
float curvatureSmoothBlend;       

uint FlatIndex(int3 c)
{
    return (uint) (c.x + c.y * (int) sizeX + c.z * (int) sizeX * (int) sizeY);
}

bool IsInside(int3 c)
{
    return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < (int) sizeX && c.y < (int) sizeY && c.z < (int) sizeZ;
}

int3 CoordFromIndex(uint idx)
{
    int x = (int) (idx % sizeX);
    int y = (int) ((idx / sizeX) % sizeY);
    int z = (int) (idx / (sizeX * sizeY));
    return int3(x, y, z);
}

// ---------------------------------------------------------------------------------------------------------
// External forces: in place on velocity, safe because gravity has no neighbor dependency
[numthreads(64, 1, 1)]
void applyExternalForces(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount || blocked[idx] != 0)
        return;

    velocity[idx].xyz += gravity * dt;
}

// ---------------------------------------------------------------------------------------------------------
// Advection: trilinear sample from the current (pre-advect) density/velocity buffers
float3 ClampToDomain(float3 localPos)
{
    return clamp(localPos, float3(0.001f, 0.001f, 0.001f), float3(sizeX - 1.001f, sizeY - 1.001f, sizeZ - 1.001f));
}

float SampleDensityAt(float3 localPos)
{
    int3 c000 = int3(floor(localPos));
    float3 f = localPos - c000;

    float x00 = lerp(density[FlatIndex(c000 + int3(0, 0, 0))], density[FlatIndex(c000 + int3(1, 0, 0))], f.x);
    float x10 = lerp(density[FlatIndex(c000 + int3(0, 1, 0))], density[FlatIndex(c000 + int3(1, 1, 0))], f.x);
    float x01 = lerp(density[FlatIndex(c000 + int3(0, 0, 1))], density[FlatIndex(c000 + int3(1, 0, 1))], f.x);
    float x11 = lerp(density[FlatIndex(c000 + int3(0, 1, 1))], density[FlatIndex(c000 + int3(1, 1, 1))], f.x);

    float y0 = lerp(x00, x10, f.y);
    float y1 = lerp(x01, x11, f.y);
    return lerp(y0, y1, f.z);
}

float3 SampleVelocityAt(float3 localPos)
{
    int3 c000 = int3(floor(localPos));
    float3 f = localPos - c000;

    float3 x00 = lerp(velocity[FlatIndex(c000 + int3(0, 0, 0))].xyz, velocity[FlatIndex(c000 + int3(1, 0, 0))].xyz, f.x);
    float3 x10 = lerp(velocity[FlatIndex(c000 + int3(0, 1, 0))].xyz, velocity[FlatIndex(c000 + int3(1, 1, 0))].xyz, f.x);
    float3 x01 = lerp(velocity[FlatIndex(c000 + int3(0, 0, 1))].xyz, velocity[FlatIndex(c000 + int3(1, 0, 1))].xyz, f.x);
    float3 x11 = lerp(velocity[FlatIndex(c000 + int3(0, 1, 1))].xyz, velocity[FlatIndex(c000 + int3(1, 1, 1))].xyz, f.x);

    float3 y0 = lerp(x00, x10, f.y);
    float3 y1 = lerp(x01, x11, f.y);
    return lerp(y0, y1, f.z);
}

// Writes to *Scratch only, never in place. A back-traced sample can land on any cell in the domain, including
// one another thread hasn't updated yet this dispatch, so this must double-buffer
[numthreads(64, 1, 1)]
void advect(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    int3 cell = CoordFromIndex(idx);
    float3 sourcePos = ClampToDomain((float3) cell - velocity[idx].xyz * dt);

    float3 sampledVelocity = SampleVelocityAt(sourcePos);
    float sampledDensity = SampleDensityAt(sourcePos);

    if (blocked[idx] != 0)
    {
        sampledVelocity = float3(0, 0, 0);
        sampledDensity = 0;
    }

    velocityScratch[idx] = float4(sampledVelocity, 0);
    densityScratch[idx] = sampledDensity;
}

// ---------------------------------------------------------------------------------------------------------
// Diffusion: snapshot the pre-iteration field as the Jacobi right-hand side.
//
// The cohesion density term went through two earlier, broken designs before this one - worth keeping the
// history so the reasoning isn't re-litigated:
//   1. Added directly to diffuseIteration's output, outside that kernel's stabilizing "/(1+count*diffusionA)" division
//   2. Moved here into diffuseSourceDensity as a RAW per-cell addition of cohesionCurvature[idx]
//      stable, but not conservative: nothing ties one cell's addition to a matching subtraction anywhere else, so it could
//      inflate the domain's total mass every tick, which RenormalizeDensity then "fixed" by rescaling
//      the ENTIRE field including cells that never received any addition, influencing peak density in cells the correction was supposed to help, 
//      and eventually driving the total negative once locally negative source values propagated far enough
// This version instead injects the DISCRETE LAPLACIAN of curvature (the difference between each cell's
// curvature and its neighbors', summed over the same 6 faces RealNeighborSums walks) rather than curvature itself
[numthreads(64, 1, 1)]
void snapshotDiffuseSource(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    float cohesionFlux = 0.0f;
    if (cohesionDensityCoefficient != 0.0f)
    {
        int3 cell = CoordFromIndex(idx);
        int3 offsets[6] =
        {
            int3(1, 0, 0), int3(-1, 0, 0),
            int3(0, 1, 0), int3(0, -1, 0),
            int3(0, 0, 1), int3(0, 0, -1)
        };

        float ownCurvature = cohesionCurvature[idx];
        [unroll]
        for (int i = 0; i < 6; i++)
        {
            int3 neighbor = cell + offsets[i];
            float neighborCurvature = (IsInside(neighbor) && blocked[FlatIndex(neighbor)] == 0)
                ? cohesionCurvature[FlatIndex(neighbor)]
                : ownCurvature; // zero-flux mirror: contributes nothing, matches BoundaryNormalAt/BoundaryPressureAt
            cohesionFlux += neighborCurvature - ownCurvature;
        }
    }

    diffuseSourceDensity[idx] = density[idx] + cohesionDensityCoefficient * dt * cohesionFlux;
    diffuseSourceVelocity[idx] = velocity[idx];
}

// Sums density/velocity over only the neighbors that exist and aren't blocked
void RealNeighborSums(int3 cell, out float densitySum, out float3 velocitySum, out int count)
{
    densitySum = 0;
    velocitySum = float3(0, 0, 0);
    count = 0;

    int3 offsets[6] =
    {
        int3(1, 0, 0), int3(-1, 0, 0),
        int3(0, 1, 0), int3(0, -1, 0),
        int3(0, 0, 1), int3(0, 0, -1)
    };

    [unroll]
    for (int i = 0; i < 6; i++)
    {
        int3 neighbor = cell + offsets[i];
        if (!IsInside(neighbor))
            continue;

        uint neighborIdx = FlatIndex(neighbor);
        if (blocked[neighborIdx] != 0)
            continue;

        densitySum += density[neighborIdx];
        velocitySum += velocity[neighborIdx].xyz;
        count++;
    }
}

// One Jacobi relaxation step, dispatched diffusionIterations times from C# with a buffer swap between calls.
//
// Gives cohesion the same direct, repeated access to density that diffusion itself has
// diffusion gets diffusionIterations (30) direct passes spreading density outward every tick, while the velocity-space
// cohesion force (applyCohesionForce) only gets one nudge through a single Advect sample
// That 30-against-1 mismatch is why velocity-space cohesion alone only slowed dilution rather than reaching a stable balance
[numthreads(64, 1, 1)]
void diffuseIteration(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        densityScratch[idx] = 0;
        velocityScratch[idx] = float4(0, 0, 0, 0);
        return;
    }

    int3 cell = CoordFromIndex(idx);
    float densitySum;
    float3 velocitySum;
    int count;
    RealNeighborSums(cell, densitySum, velocitySum, count);

    densityScratch[idx] = max(0.0f, (diffuseSourceDensity[idx] + diffusionA * densitySum) / (1.0f + count * diffusionA));
    velocityScratch[idx] = float4((diffuseSourceVelocity[idx].xyz + diffusionA * velocitySum) / (1.0f + count * diffusionA), 0);
}

// ---------------------------------------------------------------------------------------------------------
// Cohesion: a Continuum Surface Force (Brackbill/Kothe/Zemach 1992) style attraction. Real surface tension is
// curvature-driven, it's what makes a free liquid body relax toward a sphere (minimizing surface area), 
// and is the mechanism meant to counteract diffusion/advection's spreading
//
// Three passes: (1) smooth a THROWAWAY copy of density, raw density is too noisy at this grid resolution
// for a stable curvature estimate, this smoothed copy is never written back to the real mass-bearing density buffer; 
// (2) compute each cell's density gradient and unit normal from the smoothed copy; 
// (3) compute curvature (divergence of the normal field) and add sigma * curvature * grad(density) to velocity, 
// masked to only the interface region (|grad| above threshold) so deep-interior/deep-empty cells don't get a force
float BoundaryDensitySmoothAt(int3 cell, int3 neighbor)
{
    if (!IsInside(neighbor) || blocked[FlatIndex(neighbor)] != 0)
        return densitySmooth[FlatIndex(cell)];
    return densitySmooth[FlatIndex(neighbor)];
}

float3 BoundaryNormalAt(int3 cell, int3 neighbor)
{
    if (!IsInside(neighbor) || blocked[FlatIndex(neighbor)] != 0)
        return cohesionGradient[FlatIndex(cell)].xyz;
    return cohesionGradient[FlatIndex(neighbor)].xyz;
}

[numthreads(64, 1, 1)]
void snapshotDensityForSmoothing(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    densitySmooth[idx] = density[idx];
}

// One explicit-blend smoothing pass, dispatched a small fixed number of times with a buffer swap between calls
// deliberately not a solved Jacobi system like Diffuse, this is pure denoising for curvature estimation, 
// not a physically meaningful field in its own right
[numthreads(64, 1, 1)]
void smoothDensityIteration(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        densitySmoothScratch[idx] = 0;
        return;
    }

    int3 cell = CoordFromIndex(idx);
    int3 offsets[6] =
    {
        int3(1, 0, 0), int3(-1, 0, 0),
        int3(0, 1, 0), int3(0, -1, 0),
        int3(0, 0, 1), int3(0, 0, -1)
    };

    float neighborSum = 0;
    int count = 0;
    [unroll]
    for (int i = 0; i < 6; i++)
    {
        int3 neighbor = cell + offsets[i];
        if (!IsInside(neighbor))
            continue;
        uint neighborIdx = FlatIndex(neighbor);
        if (blocked[neighborIdx] != 0)
            continue;
        neighborSum += densitySmooth[neighborIdx];
        count++;
    }

    float neighborAvg = count > 0 ? neighborSum / count : densitySmooth[idx];
    densitySmoothScratch[idx] = lerp(densitySmooth[idx], neighborAvg, curvatureSmoothBlend);
}

// Central-difference gradient of the smoothed density field, normalized to a unit normal. Magnitude is
// stashed in .w so applyCohesionForce doesn't need a second buffer
[numthreads(64, 1, 1)]
void computeGradientNormal(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        cohesionGradient[idx] = float4(0, 0, 0, 0);
        return;
    }

    int3 cell = CoordFromIndex(idx);
    float right = BoundaryDensitySmoothAt(cell, cell + int3(1, 0, 0));
    float left = BoundaryDensitySmoothAt(cell, cell + int3(-1, 0, 0));
    float top = BoundaryDensitySmoothAt(cell, cell + int3(0, 1, 0));
    float bottom = BoundaryDensitySmoothAt(cell, cell + int3(0, -1, 0));
    float front = BoundaryDensitySmoothAt(cell, cell + int3(0, 0, 1));
    float back = BoundaryDensitySmoothAt(cell, cell + int3(0, 0, -1));

    float3 gradRho = float3(right - left, top - bottom, front - back) / 2.0f;
    float gradMag = length(gradRho);
    float3 normal = gradMag > 1e-5f ? gradRho / gradMag : float3(0, 0, 0);

    cohesionGradient[idx] = float4(normal, gradMag);
}

// Curvature = -div(normal field). Force = sigma * curvature * grad(density),
// added straight into velocity in place, safe because it only ever reads cohesionGradient 
// and its own cell's blocked/threshold state, never a velocity neighbor
//
// Also stashes curvature itself into cohesionCurvature, frozen for the rest of this tick and reused by every
// diffuseIteration pass in Diffuse, computed once here rather than 30 times there
[numthreads(64, 1, 1)]
void applyCohesionForce(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        cohesionCurvature[idx] = 0;
        return;
    }

    float gradMag = cohesionGradient[idx].w;
    if (gradMag < cohesionGradientThreshold)
    {
        cohesionCurvature[idx] = 0;
        return;
    }

    int3 cell = CoordFromIndex(idx);
    float3 right = BoundaryNormalAt(cell, cell + int3(1, 0, 0));
    float3 left = BoundaryNormalAt(cell, cell + int3(-1, 0, 0));
    float3 top = BoundaryNormalAt(cell, cell + int3(0, 1, 0));
    float3 bottom = BoundaryNormalAt(cell, cell + int3(0, -1, 0));
    float3 front = BoundaryNormalAt(cell, cell + int3(0, 0, 1));
    float3 back = BoundaryNormalAt(cell, cell + int3(0, 0, -1));

    float curvature = -((right.x - left.x) + (top.y - bottom.y) + (front.z - back.z)) / 2.0f;
    cohesionCurvature[idx] = curvature;

    float3 gradRho = cohesionGradient[idx].xyz * gradMag;
    velocity[idx].xyz += cohesionCoefficient * curvature * gradRho * dt;
}

// ---------------------------------------------------------------------------------------------------------
// Projection: divergence, pressure solve, gradient subtraction. Boundary handling mirrors BoundaryVelocity/BoundaryPressure exactly
float3 BoundaryVelocityAt(int3 cell, int3 neighbor)
{
    if (!IsInside(neighbor) || blocked[FlatIndex(neighbor)] != 0)
        return -velocity[FlatIndex(cell)].xyz;
    return velocity[FlatIndex(neighbor)].xyz;
}

[numthreads(64, 1, 1)]
void computeDivergence(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        divergence[idx] = 0;
        return;
    }

    int3 cell = CoordFromIndex(idx);
    float3 right = BoundaryVelocityAt(cell, cell + int3(1, 0, 0));
    float3 left = BoundaryVelocityAt(cell, cell + int3(-1, 0, 0));
    float3 top = BoundaryVelocityAt(cell, cell + int3(0, 1, 0));
    float3 bottom = BoundaryVelocityAt(cell, cell + int3(0, -1, 0));
    float3 front = BoundaryVelocityAt(cell, cell + int3(0, 0, 1));
    float3 back = BoundaryVelocityAt(cell, cell + int3(0, 0, -1));

    divergence[idx] = ((right.x - left.x) + (top.y - bottom.y) + (front.z - back.z)) / 2.0f;
}

float BoundaryPressureAt(int3 cell, int3 neighbor)
{
    if (!IsInside(neighbor) || blocked[FlatIndex(neighbor)] != 0)
        return pressure[FlatIndex(cell)];
    return pressure[FlatIndex(neighbor)];
}

// One Jacobi relaxation step for pressure, dispatched diffusionIterations times from C# with a buffer swap
// between calls. pressure/pressureScratch are warm-started
[numthreads(64, 1, 1)]
void solvePressureIteration(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    if (blocked[idx] != 0)
    {
        pressureScratch[idx] = 0;
        return;
    }

    int3 cell = CoordFromIndex(idx);
    float neighborSum =
        BoundaryPressureAt(cell, cell + int3(1, 0, 0)) + BoundaryPressureAt(cell, cell + int3(-1, 0, 0)) +
        BoundaryPressureAt(cell, cell + int3(0, 1, 0)) + BoundaryPressureAt(cell, cell + int3(0, -1, 0)) +
        BoundaryPressureAt(cell, cell + int3(0, 0, 1)) + BoundaryPressureAt(cell, cell + int3(0, 0, -1));

    pressureScratch[idx] = (neighborSum - divergence[idx]) / 6.0f;
}

[numthreads(64, 1, 1)]
void subtractPressureGradient(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount || blocked[idx] != 0)
        return;

    int3 cell = CoordFromIndex(idx);
    float right = BoundaryPressureAt(cell, cell + int3(1, 0, 0));
    float left = BoundaryPressureAt(cell, cell + int3(-1, 0, 0));
    float top = BoundaryPressureAt(cell, cell + int3(0, 1, 0));
    float bottom = BoundaryPressureAt(cell, cell + int3(0, -1, 0));
    float front = BoundaryPressureAt(cell, cell + int3(0, 0, 1));
    float back = BoundaryPressureAt(cell, cell + int3(0, 0, -1));

    float3 gradient = float3(right - left, top - bottom, front - back) / 2.0f;
    velocity[idx].xyz -= gradient;
}

// ---------------------------------------------------------------------------------------------------------
// Density renormalization: same lock-free per-slice partial-sum trick that was used on CPU for TotalDensityMass,
// just running on GPU instead. one thread per Z slice, serial within the slice, tiny CPU readback+reduce after
// Might not be needed later on. Will have a look
[numthreads(64, 1, 1)]
void sumDensitySlice(uint3 globalID : SV_DispatchThreadID)
{
    uint z = globalID.x;
    if (z >= sizeZ)
        return;

    float total = 0;
    for (uint y = 0; y < sizeY; y++)
        for (uint x = 0; x < sizeX; x++)
            total += density[FlatIndex(int3((int) x, (int) y, (int) z))];

    densityPartialSums[z] = total;
}

// renormalizeScale is computed CPU-side from the tiny densityPartialSums readback
[numthreads(64, 1, 1)]
void rescaleDensity(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    density[idx] *= renormalizeScale;
}

technique Tech0
{
    pass ApplyExternalForces { ComputeShader = compile cs_5_0 applyExternalForces(); }
    pass SnapshotDensityForSmoothing { ComputeShader = compile cs_5_0 snapshotDensityForSmoothing(); }
    pass SmoothDensityIteration { ComputeShader = compile cs_5_0 smoothDensityIteration(); }
    pass ComputeGradientNormal { ComputeShader = compile cs_5_0 computeGradientNormal(); }
    pass ApplyCohesionForce { ComputeShader = compile cs_5_0 applyCohesionForce(); }
    pass Advect { ComputeShader = compile cs_5_0 advect(); }
    pass SnapshotDiffuseSource { ComputeShader = compile cs_5_0 snapshotDiffuseSource(); }
    pass DiffuseIteration { ComputeShader = compile cs_5_0 diffuseIteration(); }
    pass ComputeDivergence { ComputeShader = compile cs_5_0 computeDivergence(); }
    pass SolvePressureIteration { ComputeShader = compile cs_5_0 solvePressureIteration(); }
    pass SubtractPressureGradient { ComputeShader = compile cs_5_0 subtractPressureGradient(); }
    pass SumDensitySlice { ComputeShader = compile cs_5_0 sumDensitySlice(); }
    pass RescaleDensity { ComputeShader = compile cs_5_0 rescaleDensity(); }
}
