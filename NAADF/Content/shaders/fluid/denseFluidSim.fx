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
// Diffusion: snapshot the pre-iteration field as the Jacobi right-hand side
[numthreads(64, 1, 1)]
void snapshotDiffuseSource(uint3 globalID : SV_DispatchThreadID)
{
    uint idx = globalID.x;
    if (idx >= cellCount)
        return;

    diffuseSourceDensity[idx] = density[idx];
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

// One Jacobi relaxation step, dispatched diffusionIterations times from C# with a buffer swap between calls
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

    densityScratch[idx] = (diffuseSourceDensity[idx] + diffusionA * densitySum) / (1.0f + count * diffusionA);
    velocityScratch[idx] = float4((diffuseSourceVelocity[idx].xyz + diffusionA * velocitySum) / (1.0f + count * diffusionA), 0);
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
    pass Advect { ComputeShader = compile cs_5_0 advect(); }
    pass SnapshotDiffuseSource { ComputeShader = compile cs_5_0 snapshotDiffuseSource(); }
    pass DiffuseIteration { ComputeShader = compile cs_5_0 diffuseIteration(); }
    pass ComputeDivergence { ComputeShader = compile cs_5_0 computeDivergence(); }
    pass SolvePressureIteration { ComputeShader = compile cs_5_0 solvePressureIteration(); }
    pass SubtractPressureGradient { ComputeShader = compile cs_5_0 subtractPressureGradient(); }
    pass SumDensitySlice { ComputeShader = compile cs_5_0 sumDensitySlice(); }
    pass RescaleDensity { ComputeShader = compile cs_5_0 rescaleDensity(); }
}
