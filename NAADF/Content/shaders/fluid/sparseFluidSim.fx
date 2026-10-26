// Sparse Eulerian fluid simulation: GPU versions of SparseFluidSolver's stages
//
// Cell data is laid out exactly like SparseFluidGrid: index = brick slot * 512 + local index, local index = x + 8y + 64z
// Per-cell kernels use one compute group per brick (8x8x8 threads), so SV_GroupID.x is the brick slot and SV_GroupIndex the local index
// Per-brick kernels use one thread per brick, the Finalize* kernels one group of threads that share out the bricks, and the source and ledger kernels a single thread

#define BRICK_SHIFT 3
#define BRICK_MASK 7
#define BRICK_CELLS 512
#define CELL_SHIFT 9

// Bits of a cell's state word, laid out as SparseFluidGrid.PackState writes it
#define STATE_TRACKED 1
#define STATE_LIQUID 2
#define STATE_SOURCE 4
#define STATE_GAINED 8
#define STATE_PRESENT 16      // face present, one bit per axis from here on
#define STATE_PRESENT_ALL 112
#define STATE_LOW_HALF 0xFFFF // everything but the stagnation counter in the upper half
#define STATE_STAGNANT_SHIFT 16

// Bits of a solid mask byte, same values as SparseFluidGrid.SolidBlocked / SolidDrain
#define SOLID_BLOCKED 2
#define SOLID_DRAIN 4

// Slots of the ledger, and of each brick's part of brickLedger
#define LEDGER_INJECTED 0
#define LEDGER_CLAMPED_AWAY 1
#define LEDGER_CLAMP_CREATED 2
#define LEDGER_PRUNED_SOLID 3
#define LEDGER_PRUNED_EMPTY 4
#define LEDGER_PRUNED_STAGNANT 5
#define LEDGER_DRAINED 6
#define LEDGER_COUNT_SOLID 7
#define LEDGER_COUNT_EMPTY 8
#define LEDGER_COUNT_DRAINED 9
#define LEDGER_SIZE 10

// Slots of the scalars buffer
#define SCALAR_RHO 0
#define SCALAR_ALPHA 1
#define SCALAR_BETA 2
#define SCALAR_RESIDUAL 3
#define SCALAR_DONE 4
#define SCALAR_ITERATIONS 5

// Outcome of fill transport for a cell, in fillStatus
#define FILL_TOUCHED 1
#define FILL_GAINED 2

float dt;
int slotCount;
float compactionRate;
float liquidThreshold;
float minInterfaceTheta;
float pressureTolerance;
float minFillEpsilon;
float gravityStep;   // downward velocity added per tick
float velocityScale; // damping factor per tick
float maxSpeed;
float visibilityThreshold;
int stagnantLimit; // ticks a dim cell may go without gaining fill
int keepSurfaceFilm; // SparseFluidSolver.keepSurfaceFilm, 0 or 1
int sourceCount;
int previousSourceCount;
int ledgerFirst;   // range of ledger slots sumBrickLedger adds up
int ledgerCount;

StructuredBuffer<int> brickInfo;     // per slot: brick x, y, z, allocated (0/1)
StructuredBuffer<int> brickNeighbor; // per slot: slot of each of the 27 surrounding bricks (self included), -1 where there is none

StructuredBuffer<uint> solidMask;    // one byte per cell, four cells per element

// Cell state words and fill fractions: each exists under a read-only and a read-write name bound to the same buffer,
// and a kernel uses only one of the two
StructuredBuffer<uint> cellState;
RWStructuredBuffer<uint> cellStateRw;
StructuredBuffer<float> fill;
RWStructuredBuffer<float> fillRw;

// Face velocities as read by velocity advection and fill transport
StructuredBuffer<float> faceX;
StructuredBuffer<float> faceY;
StructuredBuffer<float> faceZ;

// Face velocities written by velocity advection, and changed in place by the other stages that touch them
RWStructuredBuffer<float> faceRwX;
RWStructuredBuffer<float> faceRwY;
RWStructuredBuffer<float> faceRwZ;

// Pressure solve: conjugate gradient vectors, one entry per cell (zero for cells that aren't liquid)
RWStructuredBuffer<float> pressure;
RWStructuredBuffer<float> pcgDiag;
RWStructuredBuffer<float> pcgR;
RWStructuredBuffer<float> pcgS;
RWStructuredBuffer<float> pcgQ;
RWStructuredBuffer<float> brickSum; // per slot
RWStructuredBuffer<float> brickMax; // per slot
RWStructuredBuffer<float> scalars;

// Fill transport: per-cell totals of the transfers out of and into a cell, then the result
RWStructuredBuffer<float> fillOutTotal;  // outflow as the face velocities ask for it
RWStructuredBuffer<float> fillOutScaled; // outflow once limited to what the cell holds
RWStructuredBuffer<float> fillInTotal;   // inflow, from donors already limited
RWStructuredBuffer<float> fillRaw;       // fill after transport, not yet clamped to [0,1]
RWStructuredBuffer<uint> fillStatus;     // FILL_TOUCHED / FILL_GAINED

// Sources: eight ints per source cell (its index, the indices of its lower neighbors along x, y, z, then the velocity's three floats as bits)
StructuredBuffer<int> sourceCells;
StructuredBuffer<int> previousSourceCells; // one index per cell held as a source by the previous tick

// Volume bookkeeping of the tick, and per-brick figures for the layout
RWStructuredBuffer<float> ledger;
RWStructuredBuffer<float> brickLedger;  // LEDGER_SIZE entries per slot
RWStructuredBuffer<uint> brickSummary;  // per slot, as SparseFluidGrid.SetBrickSummary reads it, plus the number of liquid cells from bit 18 on
RWStructuredBuffer<uint> visibleMask;   // one bit per cell, 32 cells per element: tracked and above the visibility threshold

groupshared float sharedA[BRICK_CELLS];
groupshared float sharedB[BRICK_CELLS];
groupshared uint sharedKind[BRICK_CELLS];
groupshared float sliceA[8];
groupshared float sliceB[8];

#define REDUCE_THREADS 1024
#define REDUCE_FOLD 32
groupshared float reduceSum[REDUCE_THREADS];
groupshared float reduceMax[REDUCE_THREADS];
groupshared float foldSum[REDUCE_FOLD];
groupshared float foldMax[REDUCE_FOLD];

// ---------------------------------------------------------------------------------------------------------
// Shared helpers

// The solid mask byte of a cell: SOLID_BLOCKED and SOLID_DRAIN bits
uint SolidBits(int index)
{
    return (solidMask[index >> 2] >> ((index & 3) * 8)) & 0xFF;
}

// Whether a cell blocks fluid; a cell no brick covers (-1) counts as not solid
bool IsSolidCell(int index)
{
    return index >= 0 && (SolidBits(index) & SOLID_BLOCKED) != 0;
}

// Whether a cell is an unknown of this tick's pressure solve
bool IsLiquidCell(int index)
{
    return index >= 0 && (cellState[index] & STATE_LIQUID) != 0;
}

// Whether a cell is held by a source this tick
bool IsSourceCell(int index)
{
    return index >= 0 && (cellState[index] & STATE_SOURCE) != 0;
}

// Unit step along an axis
int3 AxisStep(int axis)
{
    return int3(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0);
}

// Slot of the brick at the given brick coordinates, found through the neighbor table of homeSlot, -1 when it isn't allocated or isn't adjacent
int BrickNear(int homeSlot, int3 brick)
{
    int3 d = brick - int3(brickInfo[homeSlot * 4], brickInfo[homeSlot * 4 + 1], brickInfo[homeSlot * 4 + 2]);
    if (any(d < -1) || any(d > 1))
        return -1;
    return brickNeighbor[homeSlot * 27 + 13 + d.x + d.y * 3 + d.z * 9];
}

// Index of the cell at the given world cell coordinates, or -1 when no brick covers it
int CellNear(int homeSlot, int3 cell)
{
    int slot = BrickNear(homeSlot, cell >> BRICK_SHIFT);
    if (slot < 0)
        return -1;
    int3 local = cell & BRICK_MASK;
    return (slot << CELL_SHIFT) | local.x | (local.y << BRICK_SHIFT) | (local.z << (BRICK_SHIFT * 2));
}

// Index of the cell one step along an axis (direction +1 or -1), or -1 when no brick covers it
int NeighborCell(int index, int axis, int direction)
{
    int shift = BRICK_SHIFT * axis;
    int coordinate = (index >> shift) & BRICK_MASK;
    if (direction > 0 ? coordinate < BRICK_MASK : coordinate > 0)
        return index + direction * (1 << shift);

    int stepInTable = axis == 0 ? 1 : (axis == 1 ? 3 : 9);
    int other = brickNeighbor[(index >> CELL_SHIFT) * 27 + 13 + direction * stepInTable];
    if (other < 0)
        return -1;
    return (other << CELL_SHIFT) | ((index & (BRICK_CELLS - 1)) - direction * (BRICK_MASK << shift));
}

// ---------------------------------------------------------------------------------------------------------
// Gravity on every present Y face; damping and the speed limit on every face

// Accelerates every present vertical face downwards by one tick of gravity
[numthreads(8, 8, 8)]
void applyGravity(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    if ((cellState[index] & (STATE_PRESENT << 1)) != 0)
        faceRwY[index] -= gravityStep;
}

// Scales every face velocity by the tick's damping factor and limits it to the maximum speed
[numthreads(8, 8, 8)]
void dampAndClampVelocity(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    faceRwX[index] = clamp(faceRwX[index] * velocityScale, -maxSpeed, maxSpeed);
    faceRwY[index] = clamp(faceRwY[index] * velocityScale, -maxSpeed, maxSpeed);
    faceRwZ[index] = clamp(faceRwZ[index] * velocityScale, -maxSpeed, maxSpeed);
}

// ---------------------------------------------------------------------------------------------------------
// Velocity advection: reads face*, writes faceRw* and which faces are present
// The arithmetic is marked precise so it rounds exactly like the CPU version

// Velocity of a face through its read-only name, 0 for a cell no brick covers
float ReadFace(int axis, int index)
{
    if (index < 0)
        return 0.0f;
    if (axis == 0)
        return faceX[index];
    if (axis == 1)
        return faceY[index];
    return faceZ[index];
}

// Linear interpolation written exactly like MathHelper.Lerp, so it rounds the same
precise float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

// Trilinear sample of one velocity component at a world position (cell centers sit on integer coordinates)
float SampleComponent(int axis, float3 pos, int homeSlot)
{
    precise float3 q = pos - 0.5f * (float3) AxisStep(axis);
    float3 base = floor(q);
    int3 c = (int3) base;
    precise float3 f = q - base;

    float v000 = ReadFace(axis, CellNear(homeSlot, c));
    float v100 = ReadFace(axis, CellNear(homeSlot, c + int3(1, 0, 0)));
    float v010 = ReadFace(axis, CellNear(homeSlot, c + int3(0, 1, 0)));
    float v110 = ReadFace(axis, CellNear(homeSlot, c + int3(1, 1, 0)));
    float v001 = ReadFace(axis, CellNear(homeSlot, c + int3(0, 0, 1)));
    float v101 = ReadFace(axis, CellNear(homeSlot, c + int3(1, 0, 1)));
    float v011 = ReadFace(axis, CellNear(homeSlot, c + int3(0, 1, 1)));
    float v111 = ReadFace(axis, CellNear(homeSlot, c + int3(1, 1, 1)));

    float c00 = Lerp(v000, v100, f.x);
    float c10 = Lerp(v010, v110, f.x);
    float c01 = Lerp(v001, v101, f.x);
    float c11 = Lerp(v011, v111, f.x);

    return Lerp(Lerp(c00, c10, f.y), Lerp(c01, c11, f.y), f.z);
}

// Trilinear sample of the full velocity at a world position
float3 SampleVelocity(float3 pos, int homeSlot)
{
    return float3(SampleComponent(0, pos, homeSlot), SampleComponent(1, pos, homeSlot), SampleComponent(2, pos, homeSlot));
}

// Velocity of one face after semi-Lagrangian advection, a face that borders no tracked cell is not present and has zero velocity
float AdvectFace(int axis, int slot, int3 cell, bool ownTracked, out bool active)
{
    active = ownTracked;
    if (!active)
    {
        int upper = CellNear(slot, cell + AxisStep(axis));
        active = upper >= 0 && (cellStateRw[upper] & STATE_TRACKED) != 0;
    }
    if (!active)
        return 0.0f;

    precise float3 facePos = (float3) cell + (float3) AxisStep(axis) * 0.5f;
    precise float3 velocity = SampleVelocity(facePos, slot);
    precise float3 backPos = facePos - velocity * dt;
    return SampleComponent(axis, backPos, slot);
}

// Semi-Lagrangian advection of the three faces each cell owns; clears the faces of unallocated slots
[numthreads(8, 8, 8)]
void advectVelocity(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;

    uint state = cellStateRw[index];
    if (brickInfo[slot * 4 + 3] == 0)
    {
        faceRwX[index] = 0.0f;
        faceRwY[index] = 0.0f;
        faceRwZ[index] = 0.0f;
        cellStateRw[index] = state & ~STATE_PRESENT_ALL;
        return;
    }

    int3 cell = int3(brickInfo[slot * 4], brickInfo[slot * 4 + 1], brickInfo[slot * 4 + 2]) * 8 + (int3) localID;
    bool ownTracked = (state & STATE_TRACKED) != 0;

    bool presentX, presentY, presentZ;
    faceRwX[index] = AdvectFace(0, slot, cell, ownTracked, presentX);
    faceRwY[index] = AdvectFace(1, slot, cell, ownTracked, presentY);
    faceRwZ[index] = AdvectFace(2, slot, cell, ownTracked, presentZ);
    cellStateRw[index] = (state & ~STATE_PRESENT_ALL)
        | (presentX ? STATE_PRESENT : 0) | (presentY ? STATE_PRESENT << 1 : 0) | (presentZ ? STATE_PRESENT << 2 : 0);
}

// ---------------------------------------------------------------------------------------------------------
// Sources: cells held full, with all their faces at the source velocity
// One thread goes through the list in order, so overlapping sources end up as on the CPU

// Releases last tick's source cells, then holds this tick's full with all their faces at the source velocity
[numthreads(1, 1, 1)]
void applySources()
{
    int i;
    for (i = 0; i < previousSourceCount; i++)
        cellStateRw[previousSourceCells[i]] &= ~STATE_SOURCE;

    float injected = 0.0f;
    for (i = 0; i < sourceCount; i++)
    {
        int cell = sourceCells[i * 8];
        injected += 1.0f - fillRw[cell];
        fillRw[cell] = 1.0f;
        cellStateRw[cell] = (cellStateRw[cell] & STATE_LOW_HALF) | STATE_TRACKED | STATE_SOURCE | STATE_PRESENT_ALL;

        for (int axis = 0; axis < 3; axis++)
        {
            float component = asfloat(sourceCells[i * 8 + 4 + axis]);
            int lower = sourceCells[i * 8 + 1 + axis];
            if (axis == 0)
            {
                faceRwX[cell] = component;
                faceRwX[lower] = component;
            }
            else if (axis == 1)
            {
                faceRwY[cell] = component;
                faceRwY[lower] = component;
            }
            else
            {
                faceRwZ[cell] = component;
                faceRwZ[lower] = component;
            }
            cellStateRw[lower] |= STATE_PRESENT << axis;
        }
    }
    ledger[LEDGER_INJECTED] += injected;
}

// ---------------------------------------------------------------------------------------------------------
// Pressure projection: changes faceRw* in place, using fill, cellFlags, solidMask and the previous tick's pressure as the starting guess
// Same system as SparseFluidSolver.SolveProjectionCpu: Jacobi-preconditioned conjugate gradient over the liquid cells
// Iterations are dispatched in batches, once the residual is within tolerance the remaining ones of a batch do nothing
// The sums an iteration needs are folded per brick inside the per-cell kernels, and over the bricks by the Finalize* kernels

// Velocity of a face through its read-write name, 0 for a cell no brick covers
float ReadFaceRw(int axis, int index)
{
    if (index < 0)
        return 0.0f;
    if (axis == 0)
        return faceRwX[index];
    if (axis == 1)
        return faceRwY[index];
    return faceRwZ[index];
}

// Sets the velocity of a face through its read-write name
void WriteFaceRw(int axis, int index, float value)
{
    if (axis == 0)
        faceRwX[index] = value;
    else if (axis == 1)
        faceRwY[index] = value;
    else
        faceRwZ[index] = value;
}

// How a liquid cell sees one of its neighbors: 0 = liquid, 1 = air, 2 = solid or source (no flow through the face)
int NeighborKind(int neighbor)
{
    if (neighbor < 0)
        return 1;
    if (IsSolidCell(neighbor) || IsSourceCell(neighbor))
        return 2;
    return IsLiquidCell(neighbor) ? 0 : 1;
}

// 1/theta for the face between a liquid cell and an air neighbor, theta being where fill crosses the liquid threshold
float AirWeight(float ownFill, int neighbor)
{
    float neighborFill = neighbor >= 0 ? fill[neighbor] : 0.0f;
    float theta = clamp((ownFill - liquidThreshold) / max(ownFill - neighborFill, 1e-6f), minInterfaceTheta, 1.0f);
    return 1.0f / theta;
}

// Which cells are liquid for this tick's pressure solve
[numthreads(8, 8, 8)]
void classifyLiquid(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    uint state = cellStateRw[index];
    bool liquid = (state & STATE_TRACKED) != 0 && (state & STATE_SOURCE) == 0 && fill[index] >= liquidThreshold && !IsSolidCell(index);
    cellStateRw[index] = liquid ? (state | STATE_LIQUID) : (state & ~STATE_LIQUID);
}

// Faces against solid carry no flow
[numthreads(8, 8, 8)]
void zeroSolidFaces(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    bool ownSolid = IsSolidCell(index);
    for (int axis = 0; axis < 3; axis++)
    {
        if (ownSolid || IsSolidCell(NeighborCell(index, axis, 1)))
            WriteFaceRw(axis, index, 0.0f);
    }
}

// The sum of sharedA and the maximum of sharedB over a brick, once every thread of the group has written its entry
// Eight threads each fold one 64-cell slice, then the first thread folds the eight results, only that thread gets the totals
void ReduceBrick(uint localIndex, out float sum, out float largest)
{
    GroupMemoryBarrierWithGroupSync();
    if (localIndex < 8)
    {
        float sliceSum = 0.0f;
        float sliceMax = 0.0f;
        for (int k = 0; k < 64; k++)
        {
            sliceSum += sharedA[localIndex * 64 + k];
            sliceMax = max(sliceMax, sharedB[localIndex * 64 + k]);
        }
        sliceA[localIndex] = sliceSum;
        sliceB[localIndex] = sliceMax;
    }
    GroupMemoryBarrierWithGroupSync();

    sum = 0.0f;
    largest = 0.0f;
    if (localIndex == 0)
    {
        for (int slice = 0; slice < 8; slice++)
        {
            sum += sliceA[slice];
            largest = max(largest, sliceB[slice]);
        }
    }
}

// Diagonal and right-hand side per liquid cell, then the starting residual r = b - A p and search direction s = r / diag
// Also leaves, per brick, the sum of r * (r / diag) and the largest |r|; reads the faces through their read-only names to stay within eight writable buffers
[numthreads(8, 8, 8)]
void buildSystem(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    float diag = 0.0f;
    float r = 0.0f;
    if (!IsLiquidCell(index))
    {
        pressure[index] = 0.0f;
        pcgDiag[index] = 0.0f;
        pcgR[index] = 0.0f;
        pcgS[index] = 0.0f;
        pcgQ[index] = 0.0f;
    }
    else
    {
        float ownFill = fill[index];
        float diagonal = 0.0f;
        float divergence = 0.0f;
        float neighborPressure = 0.0f;
        for (int axis = 0; axis < 3; axis++)
        {
            int lower = NeighborCell(index, axis, -1);
            int upper = NeighborCell(index, axis, 1);
            divergence += ReadFace(axis, index) - ReadFace(axis, lower);

            for (int side = 0; side < 2; side++)
            {
                int neighbor = side == 0 ? upper : lower;
                int kind = NeighborKind(neighbor);
                if (kind == 0)
                {
                    diagonal += 1.0f;
                    neighborPressure += pressure[neighbor];
                }
                else if (kind == 1)
                    diagonal += AirWeight(ownFill, neighbor);
            }
        }

        diag = diagonal > 0.0f ? diagonal : 1.0f;
        float b = -compactionRate * (1.0f - ownFill) / dt - divergence;
        r = b - (diag * pressure[index] - neighborPressure);

        pcgDiag[index] = diag;
        pcgR[index] = r;
        pcgS[index] = r / diag;
        pcgQ[index] = 0.0f;
    }

    sharedA[localIndex] = diag > 0.0f ? r * r / diag : 0.0f;
    sharedB[localIndex] = abs(r);
    float sum, largest;
    ReduceBrick(localIndex, sum, largest);
    if (localIndex == 0)
    {
        brickSum[slot] = sum;
        brickMax[slot] = largest;
    }
}

// q = A s, and per brick the sum of s * q
[numthreads(8, 8, 8)]
void applyA(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    float product = 0.0f;
    if (scalars[SCALAR_DONE] == 0.0f && IsLiquidCell(index))
    {
        float ownS = pcgS[index];
        float sum = pcgDiag[index] * ownS;
        for (int axis = 0; axis < 3; axis++)
        {
            int upper = NeighborCell(index, axis, 1);
            int lower = NeighborCell(index, axis, -1);
            if (NeighborKind(upper) == 0)
                sum -= pcgS[upper];
            if (NeighborKind(lower) == 0)
                sum -= pcgS[lower];
        }
        pcgQ[index] = sum;
        product = ownS * sum;
    }

    sharedA[localIndex] = product;
    sharedB[localIndex] = 0.0f;
    float total, unused;
    ReduceBrick(localIndex, total, unused);
    if (localIndex == 0)
        brickSum[slot] = total;
}

// The sum of brickSum and the maximum of brickMax over all bricks, by one group of threads:
// each takes every REDUCE_THREADS-th brick, then the results are folded in two rounds; only the first thread gets the totals
void ReduceBricks(uint thread, out float sum, out float largest)
{
    float partSum = 0.0f;
    float partMax = 0.0f;
    for (int slot = (int) thread; slot < slotCount; slot += REDUCE_THREADS)
    {
        partSum += brickSum[slot];
        partMax = max(partMax, brickMax[slot]);
    }
    reduceSum[thread] = partSum;
    reduceMax[thread] = partMax;
    GroupMemoryBarrierWithGroupSync();

    if (thread < REDUCE_FOLD)
    {
        float folded = 0.0f;
        float foldedMax = 0.0f;
        for (int k = 0; k < REDUCE_THREADS / REDUCE_FOLD; k++)
        {
            folded += reduceSum[thread * (REDUCE_THREADS / REDUCE_FOLD) + k];
            foldedMax = max(foldedMax, reduceMax[thread * (REDUCE_THREADS / REDUCE_FOLD) + k]);
        }
        foldSum[thread] = folded;
        foldMax[thread] = foldedMax;
    }
    GroupMemoryBarrierWithGroupSync();

    sum = 0.0f;
    largest = 0.0f;
    if (thread == 0)
    {
        for (int fold = 0; fold < REDUCE_FOLD; fold++)
        {
            sum += foldSum[fold];
            largest = max(largest, foldMax[fold]);
        }
    }
}

// After buildSystem: starting rho and residual
[numthreads(REDUCE_THREADS, 1, 1)]
void finalizeStart(uint3 threadID : SV_GroupThreadID)
{
    float sum, residual;
    ReduceBricks(threadID.x, sum, residual);
    if (threadID.x != 0)
        return;

    scalars[SCALAR_RHO] = sum;
    scalars[SCALAR_ALPHA] = 0.0f;
    scalars[SCALAR_BETA] = 0.0f;
    scalars[SCALAR_RESIDUAL] = residual;
    scalars[SCALAR_DONE] = residual <= pressureTolerance ? 1.0f : 0.0f;
    scalars[SCALAR_ITERATIONS] = 0.0f;
}

// After applyA: step length alpha = rho / (s . q)
[numthreads(REDUCE_THREADS, 1, 1)]
void finalizeAlpha(uint3 threadID : SV_GroupThreadID)
{
    float sq, unused;
    ReduceBricks(threadID.x, sq, unused);
    if (threadID.x != 0)
        return;

    if (scalars[SCALAR_DONE] != 0.0f)
    {
        scalars[SCALAR_ALPHA] = 0.0f;
        return;
    }

    if (abs(sq) < 1e-30f)
    {
        scalars[SCALAR_ALPHA] = 0.0f;
        scalars[SCALAR_DONE] = 1.0f;
        return;
    }
    scalars[SCALAR_ALPHA] = scalars[SCALAR_RHO] / sq;
    scalars[SCALAR_ITERATIONS] += 1.0f;
}

// p += alpha s, r -= alpha q, and per brick the sum of r * (r / diag) and the largest |r| of the new residual
[numthreads(8, 8, 8)]
void updateSolution(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    float alpha = scalars[SCALAR_ALPHA];
    float diag = pcgDiag[index];
    float r = 0.0f;
    if (diag > 0.0f)
    {
        r = pcgR[index];
        if (alpha != 0.0f)
        {
            pressure[index] += alpha * pcgS[index];
            r -= alpha * pcgQ[index];
            pcgR[index] = r;
        }
    }

    sharedA[localIndex] = diag > 0.0f ? r * r / diag : 0.0f;
    sharedB[localIndex] = abs(r);
    float sum, largest;
    ReduceBrick(localIndex, sum, largest);
    if (localIndex == 0)
    {
        brickSum[slot] = sum;
        brickMax[slot] = largest;
    }
}

// After updateSolution: new residual, and beta = rhoNew / rho unless the solve has converged
[numthreads(REDUCE_THREADS, 1, 1)]
void finalizeBeta(uint3 threadID : SV_GroupThreadID)
{
    float rhoNew, residual;
    ReduceBricks(threadID.x, rhoNew, residual);
    if (threadID.x != 0 || scalars[SCALAR_DONE] != 0.0f)
        return;

    scalars[SCALAR_RESIDUAL] = residual;
    if (residual <= pressureTolerance)
    {
        scalars[SCALAR_DONE] = 1.0f;
        return;
    }

    scalars[SCALAR_BETA] = rhoNew / scalars[SCALAR_RHO];
    scalars[SCALAR_RHO] = rhoNew;
}

// s = r / diag + beta s
[numthreads(8, 8, 8)]
void updateDirection(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    if (scalars[SCALAR_DONE] != 0.0f)
        return;

    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    float diag = pcgDiag[index];
    if (diag <= 0.0f)
        return;

    pcgS[index] = pcgR[index] / diag + scalars[SCALAR_BETA] * pcgS[index];
}

// u -= p(upper) - p(lower) on every face bordering a liquid cell that isn't against solid or a source
[numthreads(8, 8, 8)]
void subtractPressureGradient(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    if (brickInfo[(int) groupID.x * 4 + 3] == 0)
        return;

    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    bool ownLiquid = IsLiquidCell(index);
    bool ownWall = IsSolidCell(index) || IsSourceCell(index);

    for (int axis = 0; axis < 3; axis++)
    {
        int upper = NeighborCell(index, axis, 1);
        if (ownLiquid)
        {
            int kind = NeighborKind(upper);
            if (kind == 2)
                continue;

            float upperPressure = kind == 0 ? pressure[upper] : 0.0f;
            float weight = kind == 0 ? 1.0f : AirWeight(fill[index], upper);
            WriteFaceRw(axis, index, ReadFaceRw(axis, index) - (upperPressure - pressure[index]) * weight);
        }
        else if (!ownWall && IsLiquidCell(upper))
        {
            WriteFaceRw(axis, index, ReadFaceRw(axis, index) - pressure[upper] * AirWeight(fill[upper], index));
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Velocities for below-threshold cells resting on liquid or solid: their faces the pressure solve didn't cover take
// the average of the neighboring faces it did cover (zero when there are none), in place

// Whether a cell is liquid or a source, read through the read-write state name
bool IsFluidCellRw(int index)
{
    return index >= 0 && (cellStateRw[index] & (STATE_LIQUID | STATE_SOURCE)) != 0;
}

// Whether a cell is a below-threshold cell resting directly on liquid or solid
bool IsSupportedAir(int index)
{
    if (index < 0)
        return false;

    uint state = cellStateRw[index];
    if ((state & STATE_TRACKED) == 0 || (state & (STATE_LIQUID | STATE_SOURCE)) != 0 || fill[index] <= 0.0f)
        return false;

    int below = NeighborCell(index, 1, -1);
    return (below >= 0 && (cellStateRw[below] & STATE_LIQUID) != 0) || IsSolidCell(below);
}

// A face the pressure solve has made consistent: not against solid, and bordering a liquid or source cell
bool IsSolvedFace(int lower, int axis)
{
    if (lower < 0)
        return false;

    int upper = NeighborCell(lower, axis, 1);
    if (IsSolidCell(lower) || IsSolidCell(upper))
        return false;
    return IsFluidCellRw(lower) || IsFluidCellRw(upper);
}

// Gives the unsolved faces of supported air cells the average of the solved faces around them
[numthreads(8, 8, 8)]
void extrapolateIntoSupportedAir(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    bool ownSupported = IsSupportedAir(index);
    bool ownSolid = IsSolidCell(index);

    uint present = 0;
    for (int axis = 0; axis < 3; axis++)
    {
        int upper = NeighborCell(index, axis, 1);
        if (!ownSupported && !IsSupportedAir(upper))
            continue;
        if (ownSolid || IsSolidCell(upper) || IsSolvedFace(index, axis))
            continue;

        precise float sum = 0.0f;
        int count = 0;
        for (int d = 0; d < 6; d++)
        {
            int neighbor = NeighborCell(index, d >> 1, (d & 1) == 0 ? 1 : -1);
            if (IsSolvedFace(neighbor, axis))
            {
                sum += ReadFaceRw(axis, neighbor);
                count++;
            }
        }

        // Divided in double so the result rounds like the CPU version
        WriteFaceRw(axis, index, count > 0 ? (float) ((double) sum / (double) count) : 0.0f);
        present |= STATE_PRESENT << axis;
    }

    if (present != 0)
        cellStateRw[index] |= present;
}

// ---------------------------------------------------------------------------------------------------------
// Fill transport: reads face*, fill and solidMask, writes fillRaw and fillStatus
// Same donor-cell scheme as SparseFluidSolver.AdvectFillCpu. Instead of a list of transfers, three passes each gather over the six faces of a cell:
// outflow totals per donor, inflow totals per receiver, then the change in fill
// A cell goes through its faces in the order the CPU version meets them (by index of the face's lower cell, then by axis) and the arithmetic
// is marked precise, so every sum rounds exactly like the CPU version

struct CellFaces
{
    int lower[6]; // the cell a face is stored on, -1 when no brick covers it
    int upper[6];
    int axis[6];
    int order[6]; // face numbers sorted by (lower, axis)
};

// The six faces of a cell, with the order the CPU version meets them in
CellFaces FacesOf(int index)
{
    CellFaces faces;
    int key[6];
    int k;
    for (k = 0; k < 3; k++)
    {
        int below = NeighborCell(index, k, -1);

        faces.lower[k * 2] = index;
        faces.upper[k * 2] = NeighborCell(index, k, 1);
        faces.axis[k * 2] = k;
        key[k * 2] = index * 3 + k;

        faces.lower[k * 2 + 1] = below;
        faces.upper[k * 2 + 1] = index;
        faces.axis[k * 2 + 1] = k;
        key[k * 2 + 1] = below < 0 ? 0x7FFFFFFF - k : below * 3 + k;
    }

    // order[n] = the face that has exactly n faces before it
    for (int n = 0; n < 6; n++)
    {
        faces.order[n] = 0;
        for (k = 0; k < 6; k++)
        {
            int rank = 0;
            for (int other = 0; other < 6; other++)
                rank += key[other] < key[k] ? 1 : 0;
            if (rank == n)
                faces.order[n] = k;
        }
    }
    return faces;
}

// Fill moved through a face before any limiting: positive from the lower to the upper cell, negative the other way, zero when nothing moves
float FaceTransfer(int lower, int upper, int axis)
{
    if (lower < 0 || upper < 0)
        return 0.0f;

    precise float flux = clamp(ReadFace(axis, lower) * dt, -1.0f, 1.0f);
    if (flux == 0.0f)
        return 0.0f;

    float donorFill = flux > 0.0f ? fill[lower] : fill[upper];
    if (donorFill <= minFillEpsilon)
        return 0.0f;
    if (IsSolidCell(lower) || IsSolidCell(upper))
        return 0.0f;

    precise float amount = flux * donorFill;
    return amount;
}

// a * b / c rounded like the CPU version, the division goes through double because a float division on the GPU may be off by one unit in the last place
float ScaleExact(float a, float b, float c)
{
    precise float product = a * b;
    return (float) ((double) product / (double) c);
}

// A transfer once its donor's total outflow has been limited to what the donor holds
float DonorLimited(float amount, int donor)
{
    float total = fillOutTotal[donor];
    float available = fill[donor];
    if (total > available)
    {
        return ScaleExact(amount, available, total);
    }
    return amount;
}

// Per cell: total outflow as the faces ask for it, and the outflow once limited to what the cell holds
[numthreads(8, 8, 8)]
void fillOutflow(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    CellFaces faces = FacesOf(index);

    float outgoing[6];
    precise float total = 0.0f;
    int n;
    for (n = 0; n < 6; n++)
    {
        int k = faces.order[n];
        float transfer = FaceTransfer(faces.lower[k], faces.upper[k], faces.axis[k]);
        outgoing[n] = max(faces.lower[k] == index ? transfer : -transfer, 0.0f);
        if (outgoing[n] > 0.0f)
            total += outgoing[n];
    }

    float available = fill[index];
    precise float scaled = 0.0f;
    for (n = 0; n < 6; n++)
    {
        if (outgoing[n] <= 0.0f)
            continue;

        precise float amount = outgoing[n];
        if (total > available)
            amount = ScaleExact(amount, available, total);
        scaled += amount;
    }

    fillOutTotal[index] = total;
    fillOutScaled[index] = scaled;
}

// Per cell: total inflow from its donors, each already limited
[numthreads(8, 8, 8)]
void fillInflow(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    CellFaces faces = FacesOf(index);

    precise float total = 0.0f;
    for (int n = 0; n < 6; n++)
    {
        int k = faces.order[n];
        bool ownLower = faces.lower[k] == index;
        float transfer = FaceTransfer(faces.lower[k], faces.upper[k], faces.axis[k]);
        float incoming = ownLower ? -transfer : transfer;
        if (incoming > 0.0f)
            total += DonorLimited(incoming, ownLower ? faces.upper[k] : faces.lower[k]);
    }
    fillInTotal[index] = total;
}

// Per cell: the change in fill from all its transfers once both limits apply, into fillRaw and fillStatus
[numthreads(8, 8, 8)]
void fillApply(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    CellFaces faces = FacesOf(index);

    precise float delta = 0.0f;
    bool touched = false;
    for (int n = 0; n < 6; n++)
    {
        int k = faces.order[n];
        float transfer = FaceTransfer(faces.lower[k], faces.upper[k], faces.axis[k]);
        if (transfer == 0.0f)
            continue;

        int donor = transfer > 0.0f ? faces.lower[k] : faces.upper[k];
        int receiver = transfer > 0.0f ? faces.upper[k] : faces.lower[k];
        precise float amount = DonorLimited(abs(transfer), donor);

        // The receiver's total inflow is limited to its free space, counting what it gives away itself
        float incoming = fillInTotal[receiver];
        precise float space = 1.0f - fill[receiver] + fillOutScaled[receiver];
        if (incoming > space)
            amount = ScaleExact(amount, max(space, 0.0f), incoming);

        // Negligible amounts don't start new cells
        if (amount < minFillEpsilon && fill[receiver] <= minFillEpsilon && !(keepSurfaceFilm != 0 && IsLiquidCell(donor)))
            continue;

        touched = true;
        if (donor == index)
            delta -= amount;
        else
            delta += amount;
    }

    precise float raw = fill[index] + delta;
    fillRaw[index] = touched ? raw : fill[index];
    fillStatus[index] = touched ? (FILL_TOUCHED | (delta > 0.0f ? FILL_GAINED : 0)) : 0;
}

// Takes the outcome of fill transport over into fill and the state words: clamped to [0,1], cells that received fluid become tracked
[numthreads(8, 8, 8)]
void finishFill(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    uint status = fillStatus[index];
    uint state = cellStateRw[index] & ~STATE_GAINED;

    float clampedAway = 0.0f;
    float clampCreated = 0.0f;
    if ((status & FILL_TOUCHED) != 0)
    {
        float raw = fillRaw[index];
        if (raw > 1.0f)
            clampedAway = raw - 1.0f;
        else if (raw < 0.0f)
            clampCreated = -raw;

        fillRw[index] = clamp(raw, 0.0f, 1.0f);
        state |= STATE_TRACKED;
        if ((status & FILL_GAINED) != 0)
            state |= STATE_GAINED;
    }
    cellStateRw[index] = state;

    sharedA[localIndex] = clampedAway;
    sharedB[localIndex] = clampCreated;
    GroupMemoryBarrierWithGroupSync();
    if (localIndex == 0)
    {
        float sumAway = 0.0f;
        float sumCreated = 0.0f;
        for (int local = 0; local < BRICK_CELLS; local++)
        {
            sumAway += sharedA[local];
            sumCreated += sharedB[local];
        }
        brickLedger[slot * LEDGER_SIZE + LEDGER_CLAMPED_AWAY] = sumAway;
        brickLedger[slot * LEDGER_SIZE + LEDGER_CLAMP_CREATED] = sumCreated;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Pruning: removes cells in a drain or in solid, cells holding (almost) nothing, and dim cells that went too long without gaining fill

// Whether the cell below is a liquid cell that this pass keeps (neither solid nor a drain), whatever order the cells run in
bool RestsOnLiquid(int index)
{
    int below = NeighborCell(index, 1, -1);
    return below >= 0 && (cellStateRw[below] & STATE_LIQUID) != 0 && (SolidBits(below) & (SOLID_BLOCKED | SOLID_DRAIN)) == 0;
}

// Removes drained, solid, empty and stagnant cells, and leaves the brick's removed volumes and counts in brickLedger
[numthreads(8, 8, 8)]
void pruneStaleCells(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    uint state = cellStateRw[index];

    // 0 = kept, otherwise the ledger slot the removed cell's volume goes to
    uint kind = 0;
    float volume = 0.0f;
    if ((state & STATE_TRACKED) != 0)
    {
        float ownFill = fillRw[index];
        uint solidBits = SolidBits(index);
        if ((solidBits & SOLID_DRAIN) != 0)
            kind = LEDGER_DRAINED;
        else if ((solidBits & SOLID_BLOCKED) != 0)
            kind = LEDGER_PRUNED_SOLID;
        else if (keepSurfaceFilm != 0 && ownFill > 0.0f && ownFill <= visibilityThreshold && RestsOnLiquid(index))
            state &= STATE_LOW_HALF;
        else if (ownFill <= minFillEpsilon)
            kind = LEDGER_PRUNED_EMPTY;
        else if ((state & STATE_GAINED) != 0)
            state &= STATE_LOW_HALF;
        else if (ownFill <= visibilityThreshold)
        {
            uint ticks = min((state >> STATE_STAGNANT_SHIFT) + 1, 0xFFFF);
            state = (state & STATE_LOW_HALF) | (ticks << STATE_STAGNANT_SHIFT);
            if ((int) ticks >= stagnantLimit)
                kind = LEDGER_PRUNED_STAGNANT;
        }

        if (kind != 0)
        {
            volume = ownFill;
            fillRw[index] = 0.0f;
            state &= STATE_PRESENT_ALL;
        }
        cellStateRw[index] = state;
    }

    sharedA[localIndex] = volume;
    sharedKind[localIndex] = kind;
    GroupMemoryBarrierWithGroupSync();
    if (localIndex == 0)
    {
        // x = solid, y = empty, z = stagnant, w = drained
        float4 volumes = 0.0f;
        float4 counts = 0.0f;
        for (int local = 0; local < BRICK_CELLS; local++)
        {
            uint localKind = sharedKind[local];
            float4 select = float4(localKind == LEDGER_PRUNED_SOLID, localKind == LEDGER_PRUNED_EMPTY, localKind == LEDGER_PRUNED_STAGNANT, localKind == LEDGER_DRAINED);
            volumes += select * sharedA[local];
            counts += select;
        }

        int start = slot * LEDGER_SIZE;
        brickLedger[start + LEDGER_PRUNED_SOLID] = volumes.x;
        brickLedger[start + LEDGER_PRUNED_EMPTY] = volumes.y;
        brickLedger[start + LEDGER_PRUNED_STAGNANT] = volumes.z;
        brickLedger[start + LEDGER_DRAINED] = volumes.w;
        brickLedger[start + LEDGER_COUNT_SOLID] = counts.x;
        brickLedger[start + LEDGER_COUNT_EMPTY] = counts.y + counts.z;
        brickLedger[start + LEDGER_COUNT_DRAINED] = counts.w;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Bookkeeping

// Adds the per-brick parts of a range of ledger slots to the ledger
[numthreads(1, 1, 1)]
void sumBrickLedger()
{
    for (int k = ledgerFirst; k < ledgerFirst + ledgerCount; k++)
    {
        float sum = 0.0f;
        for (int slot = 0; slot < slotCount; slot++)
            sum += brickLedger[slot * LEDGER_SIZE + k];
        ledger[k] += sum;
    }
}

// Per brick: along which faces tracked cells lie, whether any face is present, and how many cells are tracked and liquid
[numthreads(64, 1, 1)]
void summarizeBricks(uint3 globalID : SV_DispatchThreadID)
{
    int slot = (int) globalID.x;
    if (slot >= slotCount)
        return;

    int start = slot << CELL_SHIFT;
    uint bits = 0;
    uint trackedCells = 0;
    uint liquidCells = 0;
    for (int local = 0; local < BRICK_CELLS; local++)
    {
        uint state = cellState[start + local];
        if ((state & STATE_PRESENT_ALL) != 0)
            bits |= 64;
        if ((state & STATE_LIQUID) != 0)
            liquidCells++;
        if ((state & STATE_TRACKED) == 0)
            continue;

        trackedCells++;
        int x = local & BRICK_MASK;
        int y = (local >> BRICK_SHIFT) & BRICK_MASK;
        int z = local >> (BRICK_SHIFT * 2);
        bits |= (x == 0 ? 1 : 0) | (x == BRICK_MASK ? 2 : 0) | (y == 0 ? 4 : 0) | (y == BRICK_MASK ? 8 : 0) | (z == 0 ? 16 : 0) | (z == BRICK_MASK ? 32 : 0);
    }
    brickSummary[slot] = bits | (trackedCells << 8) | (liquidCells << 18);
}

// One thread per 32 cells
[numthreads(64, 1, 1)]
void buildVisibleMask(uint3 globalID : SV_DispatchThreadID)
{
    int element = (int) globalID.x;
    if (element >= slotCount * (BRICK_CELLS / 32))
        return;

    int start = element * 32;
    uint bits = 0;
    for (int k = 0; k < 32; k++)
    {
        if ((cellState[start + k] & STATE_TRACKED) != 0 && fill[start + k] > visibilityThreshold)
            bits |= 1u << k;
    }
    visibleMask[element] = bits;
}

technique Tech0
{
    pass ApplyGravity
    {
        ComputeShader = compile cs_5_0 applyGravity();
    }

    pass DampAndClampVelocity
    {
        ComputeShader = compile cs_5_0 dampAndClampVelocity();
    }

    pass AdvectVelocity
    {
        ComputeShader = compile cs_5_0 advectVelocity();
    }

    pass ZeroSolidFaces
    {
        ComputeShader = compile cs_5_0 zeroSolidFaces();
    }

    pass BuildSystem
    {
        ComputeShader = compile cs_5_0 buildSystem();
    }

    pass ApplyA
    {
        ComputeShader = compile cs_5_0 applyA();
    }

    pass FinalizeStart
    {
        ComputeShader = compile cs_5_0 finalizeStart();
    }

    pass FinalizeAlpha
    {
        ComputeShader = compile cs_5_0 finalizeAlpha();
    }

    pass UpdateSolution
    {
        ComputeShader = compile cs_5_0 updateSolution();
    }

    pass FinalizeBeta
    {
        ComputeShader = compile cs_5_0 finalizeBeta();
    }

    pass UpdateDirection
    {
        ComputeShader = compile cs_5_0 updateDirection();
    }

    pass SubtractPressureGradient
    {
        ComputeShader = compile cs_5_0 subtractPressureGradient();
    }

    pass FillOutflow
    {
        ComputeShader = compile cs_5_0 fillOutflow();
    }

    pass FillInflow
    {
        ComputeShader = compile cs_5_0 fillInflow();
    }

    pass FillApply
    {
        ComputeShader = compile cs_5_0 fillApply();
    }

    pass ApplySources
    {
        ComputeShader = compile cs_5_0 applySources();
    }

    pass ClassifyLiquid
    {
        ComputeShader = compile cs_5_0 classifyLiquid();
    }

    pass ExtrapolateIntoSupportedAir
    {
        ComputeShader = compile cs_5_0 extrapolateIntoSupportedAir();
    }

    pass FinishFill
    {
        ComputeShader = compile cs_5_0 finishFill();
    }

    pass PruneStaleCells
    {
        ComputeShader = compile cs_5_0 pruneStaleCells();
    }

    pass SumBrickLedger
    {
        ComputeShader = compile cs_5_0 sumBrickLedger();
    }

    pass SummarizeBricks
    {
        ComputeShader = compile cs_5_0 summarizeBricks();
    }

    pass BuildVisibleMask
    {
        ComputeShader = compile cs_5_0 buildVisibleMask();
    }
}
