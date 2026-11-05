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
#define STATE_LIQUID_NEIGHBORS_SHIFT 7 // six bits set by markLiquidNeighbors for the pressure solve: bit axis * 2 for the + neighbor, axis * 2 + 1 for the -
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
#define KIND_EVAPORATED 11 // prune outcome: a stagnant cell lost some fill but stays

// Slots of the scalars buffer
#define SCALAR_RHO 0
#define SCALAR_ALPHA 1
#define SCALAR_BETA 2
#define SCALAR_RESIDUAL 3
#define SCALAR_DONE 4
#define SCALAR_ITERATIONS 5
#define SCALAR_SEALED_CELLS 6    // copied from sealedStatus by finalizeStart, so one read of scalars tells the CPU everything
#define SCALAR_LABELS_SETTLED 7

// Byte offsets of the group counts in dispatchArgs, for the kernels dispatched indirectly
#define ARGS_LABELS 0   // propagateSealedLabels: every slot until a round changes nothing
#define ARGS_SEALED 12  // assignSealedGroups, sumSealedResiduals, removeSealedMeans: every slot once settled with sealed cells
#define ARGS_PRESSURE 24 // applyA, updateSolution, updateDirection: every slot until the solve is done

// Sealed liquid labels: 0 = not liquid, LABEL_OPEN = connected to air, otherwise the lowest cell index of the group + 2;
// once the labels have settled, the lowest cell's own label is LABEL_ROOT | its group number
#define LABEL_OPEN 1
#define LABEL_ROOT 0x80000000
#define LABEL_STEPS 8 // propagation steps inside a brick per round
#define SEALED_STATUS_CHANGED 0
#define SEALED_STATUS_CELLS 1
#define SEALED_STATUS_GROUPS 2
#define SEALED_STATUS_SETTLED 3
#define SEALED_STATUS_ROUND_CELLS 4 // sealed cells counted by the last round that changed labels
#define SEALED_MAX_GROUPS 4096
#define SEALED_FIXED_SCALE 65536.0

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
float fastSpeedSquared; // buildFastMask: squared speed above which a cell counts as fast
int stagnantLimit; // ticks a dim cell may go without gaining fill
int keepSurfaceFilm; // SparseFluidSolver.keepSurfaceFilm, 0 or 1
int donorAcceptorFill; // SparseFluidSolver.donorAcceptorFill, 0 or 1
float thinWaterRate;      // SparseFluidSolver.ThinWaterRate
float thinWaterMinSpread; // SparseFluidSolver.thinWaterMinSpreadFill
float evaporationStep;    // SparseFluidSolver.EvaporationStep, 0 removes stagnant cells at once
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
StructuredBuffer<float> pcgDiagRead; // read-only names of pcgDiag, pcgS, pcgQ and scalars, so a kernel that only reads them binds no UAV for them
StructuredBuffer<float> pcgSRead;
StructuredBuffer<float> pcgQRead;
StructuredBuffer<float> scalarsRead;
RWStructuredBuffer<uint> sealedLabel;  // per cell, see LABEL_OPEN
RWStructuredBuffer<uint> sealedStatus; // SEALED_STATUS_* counters
RWStructuredBuffer<uint> sealedSums;   // four per group: low byte, second byte and the signed rest of the fixed-point residual sum, then the cell count
RWByteAddressBuffer dispatchArgs;      // group counts at the ARGS_* offsets

// Fill transport: per-cell totals of the transfers out of and into a cell, then the result
RWStructuredBuffer<float> fillOutTotal;  // outflow as the face velocities ask for it
RWStructuredBuffer<float> fillOutScaled; // outflow once limited to what the cell holds
RWStructuredBuffer<float> fillInTotal;   // inflow, from donors already limited
RWStructuredBuffer<float> fillRaw;       // fill after transport, not yet clamped to [0,1]
RWStructuredBuffer<uint> fillStatus;     // FILL_TOUCHED / FILL_GAINED

// Donor-acceptor corrections (see fillCorrectionTotals), held in the pressure solve's vectors, which buildSystem overwrites every tick
StructuredBuffer<float> fillLowDelta;    // change in fill from the donor-cell transfers alone
StructuredBuffer<float> fillGains;       // sum of the corrections that would raise the cell
StructuredBuffer<float> fillLosses;      // sum of the corrections that would lower the cell
RWStructuredBuffer<float> fillLowDeltaRw;
RWStructuredBuffer<float> fillGainsRw;
RWStructuredBuffer<float> fillLossesRw;
StructuredBuffer<uint> fillSurfaceAxes;  // per cell, bit a set when its surface faces axis a (fillSurfaceOrientation), held in sealedLabel
RWStructuredBuffer<uint> fillSurfaceAxesRw;

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
groupshared float sliceA[8];
groupshared float sliceB[8];
groupshared uint sharedLabel[BRICK_CELLS];
groupshared float4 sharedV[BRICK_CELLS];
groupshared float4 sharedC[BRICK_CELLS];
groupshared uint sharedBits;
groupshared uint sharedTracked;
groupshared uint sharedLiquid;
groupshared uint sharedChanged;
groupshared uint sharedSealed;
groupshared uint sharedAnyLabel;

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

// a / c rounded to the nearest float (ties to even), as an IEEE division and so the CPU version rounds it, a float division on the GPU
// may be off by up to 2.5 units in the last place. Long division of the 24-bit significands with a round and a sticky bit
// zero, subnormal, infinite and out-of-range values fall back to the plain division
float DivideRounded(float a, float c)
{
    float result = a / c;
    uint ua = asuint(a);
    uint uc = asuint(c);
    int ea = (int) ((ua >> 23) & 0xFF);
    int ec = (int) ((uc >> 23) & 0xFF);
    if (ea != 0 && ec != 0 && ea != 255 && ec != 255)
    {
        uint ma = (ua & 0x7FFFFFu) | 0x800000u;
        uint mc = (uc & 0x7FFFFFu) | 0x800000u;
        int e = ea - ec + 127;
        if (ma < mc)
        {
            ma <<= 1;
            e -= 1;
        }

        // q gets the leading 1, 23 more significand bits and the round bit, r is what remains
        uint r = ma - mc;
        uint q = 1;
        [unroll]
        for (int i = 0; i < 24; i++)
        {
            r <<= 1;
            q <<= 1;
            if (r >= mc)
            {
                r -= mc;
                q |= 1;
            }
        }

        uint m = q >> 1;
        if ((q & 1) != 0 && (r != 0 || (m & 1) != 0))
            m++;
        if (m == 0x1000000u)
        {
            m = 0x800000u;
            e++;
        }
        if (e > 0 && e < 255)
            result = asfloat(((ua ^ uc) & 0x80000000u) | ((uint) e << 23) | (m & 0x7FFFFFu));
    }
    return result;
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
// Pressure projection: changes faceRw* in place, using fill, cellState, solidMask and the previous tick's pressure as the starting guess
// Same system as SparseFluidSolver.SolveProjection: Jacobi-preconditioned conjugate gradient over the liquid cells
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

// Velocity extrapolation into the air (Bridson), one layer deep, before advection: every face that isn't present and isn't against solid
// takes the average of the present faces next to it along the same axis. Only faces that aren't present are written and only present
// ones are read, so the order the threads run in doesn't matter, advection sets these faces back to zero
[numthreads(8, 8, 8)]
void extrapolateIntoAir(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    if (brickInfo[slot * 4 + 3] == 0)
        return;

    int index = (slot << CELL_SHIFT) | (int) localIndex;
    uint state = cellStateRw[index];
    for (int axis = 0; axis < 3; axis++)
    {
        uint presentBit = STATE_PRESENT << axis;
        if ((state & presentBit) != 0)
            continue;

        precise float sum = 0.0f;
        int count = 0;
        for (int d = 0; d < 6; d++)
        {
            int neighbor = NeighborCell(index, d >> 1, (d & 1) == 0 ? 1 : -1);
            if (neighbor >= 0 && (cellStateRw[neighbor] & presentBit) != 0)
            {
                sum += ReadFaceRw(axis, neighbor);
                count++;
            }
        }
        if (count == 0 || IsSolidCell(index) || IsSolidCell(NeighborCell(index, axis, 1)))
            continue;
        WriteFaceRw(axis, index, DivideRounded(sum, (float) count));
    }
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

// The sums of sharedA and sharedB over a brick, halving the range each step, once every thread of the group has written its entry, every thread gets the totals
void SumBrickPair(uint localIndex, out float sumA, out float sumB)
{
    for (uint stride = BRICK_CELLS / 2; stride > 0; stride >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (localIndex < stride)
        {
            sharedA[localIndex] += sharedA[localIndex + stride];
            sharedB[localIndex] += sharedB[localIndex + stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    sumA = sharedA[0];
    sumB = sharedB[0];
}

// Per liquid cell, which of its six neighbors are liquid cells too, in the state word's STATE_LIQUID_NEIGHBORS bits
// (a liquid cell is never solid or a source, so these are the neighbors NeighborKind calls liquid). Only those bits change, 
// so reading a neighbor's word while it is written is safe
[numthreads(8, 8, 8)]
void markLiquidNeighbors(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    uint state = cellStateRw[index];
    uint mask = 0;
    if ((state & STATE_LIQUID) != 0)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            int upper = NeighborCell(index, axis, 1);
            int lower = NeighborCell(index, axis, -1);
            if (upper >= 0 && (cellStateRw[upper] & STATE_LIQUID) != 0)
                mask |= 1u << (axis * 2);
            if (lower >= 0 && (cellStateRw[lower] & STATE_LIQUID) != 0)
                mask |= 1u << (axis * 2 + 1);
        }
    }
    cellStateRw[index] = (state & ~(63u << STATE_LIQUID_NEIGHBORS_SHIFT)) | (mask << STATE_LIQUID_NEIGHBORS_SHIFT);
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

// ---------------------------------------------------------------------------------------------------------
// Sealed liquid (Bridson): a connected group of liquid cells with no air neighbor has its pressure fixed only up to a constant,
// and A p = b only has a solution when b sums to zero over the group, so the group's mean is taken out of the starting residual
// A's columns sum to zero over such a group, so the residual r = b - A p sums to the same as b

// Liquid cells next to air (or with no liquid neighbor) start out open, the others with their own index; liquid neighbors then share the lowest label
[numthreads(8, 8, 8)]
void initSealedLabels(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    uint label = 0;
    if (IsLiquidCell(index))
    {
        // A cell with no liquid neighbor is a group of its own whose diagonal is forced to 1, so it is not singular either
        bool anyAir = false, anyLiquid = false;
        for (int axis = 0; axis < 3; axis++)
        {
            int upperKind = NeighborKind(NeighborCell(index, axis, 1));
            int lowerKind = NeighborKind(NeighborCell(index, axis, -1));
            anyAir = anyAir || upperKind == 1 || lowerKind == 1;
            anyLiquid = anyLiquid || upperKind == 0 || lowerKind == 0;
        }
        label = anyAir || !anyLiquid ? LABEL_OPEN : (uint) index + 2;
    }
    sealedLabel[index] = label;
}

// Clears the counters before the first propagation round, which then runs over every slot, and holds back the sealed-mean kernels
[numthreads(1, 1, 1)]
void startSealedLabels()
{
    sealedStatus[SEALED_STATUS_CHANGED] = 0;
    sealedStatus[SEALED_STATUS_CELLS] = 0;
    sealedStatus[SEALED_STATUS_GROUPS] = 0;
    sealedStatus[SEALED_STATUS_SETTLED] = 0;
    sealedStatus[SEALED_STATUS_ROUND_CELLS] = 0;
    dispatchArgs.Store3(ARGS_LABELS, uint3((uint) slotCount, 1, 1));
    dispatchArgs.Store3(ARGS_SEALED, uint3(0, 1, 1));
}

// Marks the labels final with the given number of sealed cells: no further rounds, and the sealed-mean kernels run if there are any
void SettleSealedLabels(uint sealedCells)
{
    sealedStatus[SEALED_STATUS_CELLS] = sealedCells;
    sealedStatus[SEALED_STATUS_SETTLED] = 1;
    dispatchArgs.Store(ARGS_LABELS, 0);
    dispatchArgs.Store3(ARGS_SEALED, uint3(sealedCells > 0 ? (uint) slotCount : 0, 1, 1));
}

// After each propagation round: settles the labels if the round changed nothing, otherwise clears its counters for the next one
[numthreads(1, 1, 1)]
void finishLabelRound()
{
    if (sealedStatus[SEALED_STATUS_SETTLED] != 0)
        return;
    if (sealedStatus[SEALED_STATUS_CHANGED] == 0)
    {
        SettleSealedLabels(sealedStatus[SEALED_STATUS_CELLS]);
        return;
    }
    sealedStatus[SEALED_STATUS_CHANGED] = 0;
    sealedStatus[SEALED_STATUS_ROUND_CELLS] = sealedStatus[SEALED_STATUS_CELLS];
    sealedStatus[SEALED_STATUS_CELLS] = 0;
}

// When the round limit is reached: takes the labels as they are, with the cell count of the last round
[numthreads(1, 1, 1)]
void forceSettleSealedLabels()
{
    if (sealedStatus[SEALED_STATUS_SETTLED] == 0)
        SettleSealedLabels(sealedStatus[SEALED_STATUS_ROUND_CELLS]);
}

// One round: each liquid cell takes the lowest label of its liquid neighbors in other bricks once, then of those in its own brick
// for LABEL_STEPS steps. Counts whether any label changed and how many liquid cells are still sealed, labels only ever decrease,
// so a round in which nothing changed means every group carries its lowest label, or LABEL_OPEN if any of its cells touches air.
// A brick none of whose cells is still sealed can neither change nor count anything, and skips the round
[numthreads(8, 8, 8)]
void propagateSealedLabels(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    uint own = sealedLabel[index];
    uint best = own;
    uint inBrick = 0; // bit d set when neighbor d (axis d / 2, + for even d) is a liquid cell of this brick
    if (localIndex == 0)
    {
        sharedChanged = 0;
        sharedSealed = 0;
        sharedAnyLabel = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (own > LABEL_OPEN)
        sharedAnyLabel = 1;
    GroupMemoryBarrierWithGroupSync();
    if (sharedAnyLabel == 0)
        return;

    if (own > LABEL_OPEN)
    {
        uint liquidNeighbors = cellState[index] >> STATE_LIQUID_NEIGHBORS_SHIFT;
        for (int d = 0; d < 6; d++)
        {
            if ((liquidNeighbors & (1u << d)) == 0)
                continue;
            int neighbor = NeighborCell(index, d >> 1, (d & 1) == 0 ? 1 : -1);
            if ((neighbor >> CELL_SHIFT) == slot)
                inBrick |= 1u << d;
            else
                best = min(best, sealedLabel[neighbor]);
        }
    }

    sharedLabel[localIndex] = best;
    for (int step = 0; step < LABEL_STEPS; step++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint next = best;
        for (int d = 0; d < 6; d++)
        {
            if ((inBrick & (1u << d)) != 0)
                next = min(next, sharedLabel[(int) localIndex + ((d & 1) == 0 ? 1 : -1) * (1 << (BRICK_SHIFT * (d >> 1)))]);
        }
        GroupMemoryBarrierWithGroupSync();
        best = next;
        sharedLabel[localIndex] = best;
    }

    if (best < own)
    {
        sealedLabel[index] = best;
        sharedChanged = 1;
    }
    if (best > LABEL_OPEN)
        InterlockedAdd(sharedSealed, 1);
    GroupMemoryBarrierWithGroupSync();
    if (localIndex == 0)
    {
        if (sharedChanged != 0)
            sealedStatus[SEALED_STATUS_CHANGED] = 1;
        if (sharedSealed != 0)
            InterlockedAdd(sealedStatus[SEALED_STATUS_CELLS], sharedSealed);
    }
}

// The group number of a sealed cell, from its own label or its group's lowest cell. SEALED_MAX_GROUPS or more when it has none
uint SealedGroup(uint label)
{
    return (label & LABEL_ROOT) != 0 ? (label & ~LABEL_ROOT) : (sealedLabel[label - 2] & ~LABEL_ROOT);
}

// Once the labels have settled: the lowest cell of each sealed group numbers its group and clears the group's sums
[numthreads(8, 8, 8)]
void assignSealedGroups(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    if (sealedLabel[index] != (uint) index + 2)
        return;

    uint group;
    InterlockedAdd(sealedStatus[SEALED_STATUS_GROUPS], 1, group);
    sealedLabel[index] = LABEL_ROOT | group;
    if (group < SEALED_MAX_GROUPS)
    {
        for (int k = 0; k < 4; k++)
            sealedSums[group * 4 + k] = 0;
    }
}

// Adds each sealed cell's residual to its group's sum, in fixed point so the sum doesn't depend on the order of the additions
[numthreads(8, 8, 8)]
void sumSealedResiduals(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    uint label = sealedLabel[index];
    if (label <= LABEL_OPEN)
        return;
    uint group = SealedGroup(label);
    if (group >= SEALED_MAX_GROUPS)
        return;

    int fixedPoint = (int) round(clamp(pcgR[index], -30000.0f, 30000.0f) * SEALED_FIXED_SCALE);
    InterlockedAdd(sealedSums[group * 4], (uint) (fixedPoint & 0xFF));
    InterlockedAdd(sealedSums[group * 4 + 1], (uint) ((fixedPoint >> 8) & 0xFF));
    InterlockedAdd(sealedSums[group * 4 + 2], (uint) (fixedPoint >> 16));
    InterlockedAdd(sealedSums[group * 4 + 3], 1);
}

// Takes each sealed group's mean out of the residual of its cells, sets s = r / diag again, and redoes the per-brick sums as buildSystem leaves them
[numthreads(8, 8, 8)]
void removeSealedMeans(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    float diag = pcgDiag[index];
    float r = pcgR[index];
    uint label = sealedLabel[index];
    if (label > LABEL_OPEN)
    {
        uint group = SealedGroup(label);
        if (group < SEALED_MAX_GROUPS)
        {
            // The fixed-point sum is exact; its float conversion and the mean are good to about 1e-7 relative, far inside the solver's tolerance
            float count = (float) sealedSums[group * 4 + 3];
            float mean = (float) asint(sealedSums[group * 4 + 2]) / count + ((float) sealedSums[group * 4 + 1] * 256.0f + (float) sealedSums[group * 4]) / (SEALED_FIXED_SCALE * count);
            r -= mean;
            pcgR[index] = r;
            pcgS[index] = r / diag;
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

// q = A s, and per brick the sum of s * q; the liquid neighbors come from markLiquidNeighbors
[numthreads(8, 8, 8)]
void applyA(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    int index = (slot << CELL_SHIFT) | (int) localIndex;
    float product = 0.0f;
    uint state = cellState[index];
    if (scalars[SCALAR_DONE] == 0.0f && (state & STATE_LIQUID) != 0)
    {
        uint liquidNeighbors = state >> STATE_LIQUID_NEIGHBORS_SHIFT;
        float ownS = pcgS[index];
        float sum = pcgDiag[index] * ownS;
        for (int axis = 0; axis < 3; axis++)
        {
            if ((liquidNeighbors & (1u << (axis * 2))) != 0)
                sum -= pcgS[NeighborCell(index, axis, 1)];
            if ((liquidNeighbors & (2u << (axis * 2))) != 0)
                sum -= pcgS[NeighborCell(index, axis, -1)];
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
// With skip set (the same for every thread) the bricks aren't read and the totals are zero
void ReduceBricks(uint thread, bool skip, out float sum, out float largest)
{
    float partSum = 0.0f;
    float partMax = 0.0f;
    if (!skip)
    {
        for (int slot = (int) thread; slot < slotCount; slot += REDUCE_THREADS)
        {
            partSum += brickSum[slot];
            partMax = max(partMax, brickMax[slot]);
        }
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

// Stops the iterations dispatched after this point
void FinishSolve()
{
    scalars[SCALAR_DONE] = 1.0f;
    dispatchArgs.Store(ARGS_PRESSURE, 0);
}

// After buildSystem and the sealed means: starting rho and residual, and the sealed-label outcome for the CPU
// Labels that haven't settled yet leave the solve marked done before it starts, so the CPU can finish them and start the solve again
[numthreads(REDUCE_THREADS, 1, 1)]
void finalizeStart(uint3 threadID : SV_GroupThreadID)
{
    float sum, residual;
    ReduceBricks(threadID.x, false, sum, residual);
    if (threadID.x != 0)
        return;

    bool settled = sealedStatus[SEALED_STATUS_SETTLED] != 0;
    scalars[SCALAR_RHO] = sum;
    scalars[SCALAR_ALPHA] = 0.0f;
    scalars[SCALAR_BETA] = 0.0f;
    scalars[SCALAR_RESIDUAL] = residual;
    scalars[SCALAR_ITERATIONS] = 0.0f;
    scalars[SCALAR_SEALED_CELLS] = (float) sealedStatus[SEALED_STATUS_CELLS];
    scalars[SCALAR_LABELS_SETTLED] = settled ? 1.0f : 0.0f;
    scalars[SCALAR_DONE] = 0.0f;
    dispatchArgs.Store3(ARGS_PRESSURE, uint3((uint) slotCount, 1, 1));
    if (!settled || residual <= pressureTolerance)
        FinishSolve();
}

// After applyA: step length alpha = rho / (s . q)
[numthreads(REDUCE_THREADS, 1, 1)]
void finalizeAlpha(uint3 threadID : SV_GroupThreadID)
{
    float sq, unused;
    ReduceBricks(threadID.x, scalars[SCALAR_DONE] != 0.0f, sq, unused);
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
        FinishSolve();
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
    float alpha = scalarsRead[SCALAR_ALPHA];
    float diag = pcgDiagRead[index];
    float r = 0.0f;
    if (diag > 0.0f)
    {
        r = pcgR[index];
        if (alpha != 0.0f)
        {
            pressure[index] += alpha * pcgSRead[index];
            r -= alpha * pcgQRead[index];
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
    ReduceBricks(threadID.x, scalars[SCALAR_DONE] != 0.0f, rhoNew, residual);
    if (threadID.x != 0 || scalars[SCALAR_DONE] != 0.0f)
        return;

    scalars[SCALAR_RESIDUAL] = residual;
    if (residual <= pressureTolerance)
    {
        FinishSolve();
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

// Whether a cell holds fluid that nothing holds up: tracked with fill, neither liquid, source nor supported air
bool IsUnsupportedFluid(int index)
{
    if (index < 0)
        return false;
    uint state = cellStateRw[index];
    if ((state & STATE_TRACKED) == 0 || (state & (STATE_LIQUID | STATE_SOURCE)) != 0 || fill[index] <= 0.0f)
        return false;
    return !IsSupportedAir(index);
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

// Gives the unsolved faces of supported air cells the average of the solved faces around them, except faces shared with
// unsupported fluid, which keep that fluid's own velocity so it can still fall onto the supported cell
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
        if (IsUnsupportedFluid(ownSupported ? upper : index))
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

        WriteFaceRw(axis, index, count > 0 ? DivideRounded(sum, (float) count) : 0.0f);
        present |= STATE_PRESENT << axis;
    }

    if (present != 0)
        cellStateRw[index] |= present;
}

// ---------------------------------------------------------------------------------------------------------
// Fill transport: reads face*, fill and solidMask, writes fillRaw and fillStatus
// Same scheme as SparseFluidSolver.AdvectFill. Instead of a list of transfers, three passes each gather over the six faces of a cell:
// outflow totals per donor, inflow totals per receiver, then the change in fill
// With donorAcceptorFill, a fourth pass before the last one gathers the corrections towards the donor-acceptor fluxes
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
    [unroll]
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
    [unroll]
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

// a * b / c rounded like the CPU version
float ScaleExact(float a, float b, float c)
{
    precise float product = a * b;
    return DivideRounded(product, c);
}

// A transfer once its donor's total outflow has been limited to what the donor holds
float DonorLimited(float amount, int donor)
{
    float total = fillOutTotal[donor];
    float available = fill[donor];
    float limited = amount;
    if (total > available)
        limited = ScaleExact(amount, available, total);
    return limited;
}

// A donor-limited transfer once its receiver's total inflow has been limited to its free space, counting what it gives away itself
float ReceiverLimited(float amount, int receiver)
{
    float incoming = fillInTotal[receiver];
    precise float space = 1.0f - fill[receiver] + fillOutScaled[receiver];
    precise float limited = amount;
    if (incoming > space)
        limited = ScaleExact(amount, max(space, 0.0f), incoming);
    return limited;
}

// The fill of the neighbor one step along an axis, the given fill when that neighbor is solid, 0 when no brick covers it
float NeighborFillOrOwn(int index, int axis, int direction, float ownFill)
{
    int neighbor = NeighborCell(index, axis, direction);
    if (IsSolidCell(neighbor))
        return ownFill;
    return neighbor >= 0 ? fill[neighbor] : 0.0f;
}

// Whether the neighbor one step along an axis holds no fluid and isn't solid
bool IsEmptyNeighbor(int index, int axis, int direction)
{
    int neighbor = NeighborCell(index, axis, direction);
    if (neighbor >= 0 && fill[neighbor] > minFillEpsilon)
        return false;
    return !IsSolidCell(neighbor);
}

// The axes a cell's surface faces, one bit per axis: those along which the fill gradient is largest (and not zero)
uint SurfaceAxes(int index, float ownFill)
{
    float gradient[3];
    for (int a = 0; a < 3; a++)
    {
        precise float g = abs(NeighborFillOrOwn(index, a, -1, ownFill) - NeighborFillOrOwn(index, a, 1, ownFill));
        gradient[a] = g;
    }
    uint axes = 0;
    for (uint axis = 0; axis < 3; axis++)
    {
        float largestOther = max(gradient[(axis + 1) % 3], gradient[(axis + 2) % 3]);
        if (gradient[axis] > 0.0f && gradient[axis] >= largestOther)
            axes |= 1u << axis;
    }
    return axes;
}

// With donor-acceptor fill: per cell, the axes its surface faces, so the fill transport kernels look them up instead of working them out per face
[numthreads(8, 8, 8)]
void fillSurfaceOrientation(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    fillSurfaceAxesRw[index] = SurfaceAxes(index, fill[index]);
}

// Same as SparseFluidSolver.DonorAcceptorAmount: Hirt and Nichols' flux of fill from donor to acceptor for a face moving volume (0, 1]
float DonorAcceptorAmount(int axis, int donor, float acceptorFill, float volume, int upstreamDirection)
{
    float donorFill = fill[donor];
    bool acceptorEmpty = acceptorFill <= minFillEpsilon;
    bool upstreamEmpty = IsEmptyNeighbor(donor, axis, upstreamDirection);
    bool useAcceptor = !(acceptorEmpty && upstreamEmpty) && (acceptorEmpty || upstreamEmpty || ((fillSurfaceAxes[donor] >> axis) & 1u) != 0);
    float flowingFill = useAcceptor ? acceptorFill : donorFill;
    precise float extra = max((1.0f - flowingFill) * volume - (1.0f - donorFill), 0.0f);
    precise float amount = min(flowingFill * volume + extra, donorFill);
    return amount;
}

// The donor-acceptor amount minus the donor-cell amount of a face with the given FaceTransfer (not zero), towards the transfer's receiver
float TransferCorrection(int lower, int upper, int axis, float transfer)
{
    precise float flux = clamp(ReadFace(axis, lower) * dt, -1.0f, 1.0f);
    precise float correction;
    if (transfer > 0.0f)
        correction = DonorAcceptorAmount(axis, lower, fill[upper], flux, -1) - transfer;
    else
        correction = DonorAcceptorAmount(axis, upper, fill[lower], -flux, 1) - (-transfer);
    return correction;
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

// With donor-acceptor fill: per cell, the change in fill from the donor-cell transfers once both limits apply, and the sums of the
// corrections that would raise and lower it, as SparseFluidSolver.ApplyLimitedCorrections gathers them
[numthreads(8, 8, 8)]
void fillCorrectionTotals(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    CellFaces faces = FacesOf(index);

    precise float delta = 0.0f;
    precise float gains = 0.0f;
    precise float losses = 0.0f;
    for (int n = 0; n < 6; n++)
    {
        int k = faces.order[n];
        float transfer = FaceTransfer(faces.lower[k], faces.upper[k], faces.axis[k]);
        if (transfer == 0.0f)
            continue;

        int donor = transfer > 0.0f ? faces.lower[k] : faces.upper[k];
        int receiver = transfer > 0.0f ? faces.upper[k] : faces.lower[k];
        float amount = ReceiverLimited(DonorLimited(abs(transfer), donor), receiver);
        if (donor == index)
            delta -= amount;
        else
            delta += amount;

        float correction = TransferCorrection(faces.lower[k], faces.upper[k], faces.axis[k], transfer);
        if (correction > 0.0f)
        {
            if (receiver == index)
                gains += correction;
            else
                losses += correction;
        }
        else if (correction < 0.0f)
        {
            if (donor == index)
                gains -= correction;
            else
                losses -= correction;
        }
    }
    fillLowDeltaRw[index] = delta;
    fillGainsRw[index] = gains;
    fillLossesRw[index] = losses;
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
        precise float amount = ReceiverLimited(DonorLimited(abs(transfer), donor), receiver);

        // The correction towards the donor-acceptor amount, scaled so that neither the cell it raises nor the one it lowers leaves [0,1]
        if (donorAcceptorFill != 0)
        {
            float correction = TransferCorrection(faces.lower[k], faces.upper[k], faces.axis[k], transfer);
            if (correction != 0.0f)
            {
                int gainer = correction > 0.0f ? receiver : donor;
                int loser = correction > 0.0f ? donor : receiver;
                precise float gainerFill = fill[gainer] + fillLowDelta[gainer];
                precise float loserFill = fill[loser] + fillLowDelta[loser];
                float raise = min(1.0f, DivideRounded(max(1.0f - gainerFill, 0.0f), fillGains[gainer]));
                float lower = min(1.0f, DivideRounded(max(loserFill, 0.0f), fillLosses[loser]));
                precise float scaledCorrection = min(raise, lower) * correction;
                amount = amount + scaledCorrection;
            }
        }

        // Negligible amounts don't start new cells
        if (amount >= 0.0f && amount < minFillEpsilon && fill[receiver] <= minFillEpsilon && !(keepSurfaceFilm != 0 && IsLiquidCell(donor)))
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
    float sumAway, sumCreated;
    SumBrickPair(localIndex, sumAway, sumCreated);
    if (localIndex == 0)
    {
        brickLedger[slot * LEDGER_SIZE + LEDGER_CLAMPED_AWAY] = sumAway;
        brickLedger[slot * LEDGER_SIZE + LEDGER_CLAMP_CREATED] = sumCreated;
    }
}

// ---------------------------------------------------------------------------------------------------------
// Thin water: dim cells resting on solid ground, away from liquid, pass fill to horizontal neighbors holding less and over edges
// Same rules and arithmetic order as SparseFluidSolver.FlowThinWater, reads fill and cellState, writes fillRaw, then takes it over

// Neighbor in the horizontal direction d: 0 = +X, 1 = -X, 2 = +Z, 3 = -Z
int HorizontalNeighbor(int index, int d)
{
    return NeighborCell(index, d < 2 ? 0 : 2, (d & 1) == 0 ? 1 : -1);
}

// Whether the cell below is solid; whenNoBrick is the answer when no brick covers it
bool SolidBelow(int index, bool whenNoBrick)
{
    int below = NeighborCell(index, 1, -1);
    return below < 0 ? whenNoBrick : IsSolidCell(below);
}

// A dim cell resting on solid with no liquid or source cell beside or above it
bool IsThinWater(int index)
{
    if (index < 0)
        return false;
    uint state = cellState[index];
    if ((state & STATE_TRACKED) == 0 || (state & (STATE_LIQUID | STATE_SOURCE)) != 0)
        return false;
    float f = fill[index];
    if (f <= minFillEpsilon || f >= liquidThreshold)
        return false;
    if (IsSolidCell(index) || !SolidBelow(index, false))
        return false;

    for (int d = 0; d < 4; d++)
    {
        int neighbor = HorizontalNeighbor(index, d);
        if (IsLiquidCell(neighbor) || IsSourceCell(neighbor))
            return false;
    }
    int above = NeighborCell(index, 1, 1);
    return !IsLiquidCell(above) && !IsSourceCell(above);
}

// Fill the thin water cell from passes to its horizontal neighbor to this tick
float ThinWaterFlow(int from, int to)
{
    if (!IsThinWater(from) || (cellState[to] & (STATE_LIQUID | STATE_SOURCE)) != 0)
        return 0.0f;
    float fromFill = fill[from];
    float toFill = fill[to];
    if (toFill >= fromFill)
        return 0.0f;
    if (IsSolidCell(to))
        return 0.0f;
    if (fromFill <= thinWaterMinSpread && SolidBelow(to, true))
        return 0.0f;
    precise float flow = thinWaterRate * (fromFill - toFill);
    return flow;
}

// Per cell: fill after this tick's thin water exchange with its horizontal neighbors, into fillRaw
[numthreads(8, 8, 8)]
void thinWaterFlow(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    precise float delta = 0.0f;
    for (int d = 0; d < 4; d++)
    {
        int neighbor = HorizontalNeighbor(index, d);
        if (neighbor >= 0)
            delta += ThinWaterFlow(neighbor, index) - ThinWaterFlow(index, neighbor);
    }
    precise float raw = fill[index] + delta;
    fillRaw[index] = raw;
}

// Takes fillRaw over into fill; cells that received thin water become tracked
[numthreads(8, 8, 8)]
void thinWaterApply(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int index = ((int) groupID.x << CELL_SHIFT) | (int) localIndex;
    float raw = fillRaw[index];
    fillRw[index] = raw;
    if (raw > 0.0f)
        cellStateRw[index] |= STATE_TRACKED;
}

// ---------------------------------------------------------------------------------------------------------
// Pruning: removes cells in a drain or in solid, cells holding (almost) nothing, and dim cells that went too long without gaining fill
// (with evaporationStep > 0, those lose that much fill per tick instead, until nearly empty)

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
            {
                precise float left = ownFill - evaporationStep;
                if (evaporationStep > 0.0f && left > minFillEpsilon)
                {
                    kind = KIND_EVAPORATED;
                    precise float lost = ownFill - left;
                    volume = lost;
                    fillRw[index] = left;
                }
                else
                    kind = LEDGER_PRUNED_STAGNANT;
            }
        }

        if (kind != 0 && kind != KIND_EVAPORATED)
        {
            volume = ownFill;
            fillRw[index] = 0.0f;
            state &= STATE_PRESENT_ALL | STATE_LIQUID; // the liquid bit stays until the next classification, RestsOnLiquid of the cell above reads it
        }
        cellStateRw[index] = state;
    }

    // x = solid, y = empty, z = stagnant, w = drained, summed over the brick by halving the range each step
    float4 select = float4(kind == LEDGER_PRUNED_SOLID, kind == LEDGER_PRUNED_EMPTY, kind == LEDGER_PRUNED_STAGNANT, kind == LEDGER_DRAINED);
    sharedV[localIndex] = (select + float4(0.0f, 0.0f, kind == KIND_EVAPORATED, 0.0f)) * volume;
    sharedC[localIndex] = select;
    for (uint stride = BRICK_CELLS / 2; stride > 0; stride >>= 1)
    {
        GroupMemoryBarrierWithGroupSync();
        if (localIndex < stride)
        {
            sharedV[localIndex] += sharedV[localIndex + stride];
            sharedC[localIndex] += sharedC[localIndex + stride];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (localIndex == 0)
    {
        float4 volumes = sharedV[0];
        float4 counts = sharedC[0];
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

// Adds the per-brick parts of a range of ledger slots to the ledger, by one group of threads:
// each takes every REDUCE_THREADS-th brick, then the parts are summed by halving the range each step
[numthreads(REDUCE_THREADS, 1, 1)]
void sumBrickLedger(uint3 threadID : SV_GroupThreadID)
{
    uint thread = threadID.x;
    for (int k = ledgerFirst; k < ledgerFirst + ledgerCount; k++)
    {
        float part = 0.0f;
        for (int slot = (int) thread; slot < slotCount; slot += REDUCE_THREADS)
            part += brickLedger[slot * LEDGER_SIZE + k];
        reduceSum[thread] = part;
        for (uint stride = REDUCE_THREADS / 2; stride > 0; stride >>= 1)
        {
            GroupMemoryBarrierWithGroupSync();
            if (thread < stride)
                reduceSum[thread] += reduceSum[thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread == 0)
            ledger[k] += reduceSum[0];
        GroupMemoryBarrierWithGroupSync();
    }
}

// Per brick, one group of threads: along which faces tracked cells lie, whether any face is present, and how many cells are tracked and liquid
[numthreads(8, 8, 8)]
void summarizeBricks(uint3 groupID : SV_GroupID, uint localIndex : SV_GroupIndex)
{
    int slot = (int) groupID.x;
    if (localIndex == 0)
    {
        sharedBits = 0;
        sharedTracked = 0;
        sharedLiquid = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint state = cellState[(slot << CELL_SHIFT) | (int) localIndex];
    uint bits = (state & STATE_PRESENT_ALL) != 0 ? 64 : 0;
    if ((state & STATE_LIQUID) != 0)
        InterlockedAdd(sharedLiquid, 1);
    if ((state & STATE_TRACKED) != 0)
    {
        InterlockedAdd(sharedTracked, 1);
        int x = (int) localIndex & BRICK_MASK;
        int y = ((int) localIndex >> BRICK_SHIFT) & BRICK_MASK;
        int z = (int) localIndex >> (BRICK_SHIFT * 2);
        bits |= (x == 0 ? 1 : 0) | (x == BRICK_MASK ? 2 : 0) | (y == 0 ? 4 : 0) | (y == BRICK_MASK ? 8 : 0) | (z == 0 ? 16 : 0) | (z == BRICK_MASK ? 32 : 0);
    }
    if (bits != 0)
        InterlockedOr(sharedBits, bits);
    GroupMemoryBarrierWithGroupSync();

    if (localIndex == 0)
        brickSummary[slot] = sharedBits | (sharedTracked << 8) | (sharedLiquid << 18);
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

// One thread per 32 cells: tracked, above the visibility threshold, and moving faster than fastSpeedSquared allows,
// the speed taken at the cell center from the average of each axis' two faces
[numthreads(64, 1, 1)]
void buildFastMask(uint3 globalID : SV_DispatchThreadID)
{
    int element = (int) globalID.x;
    if (element >= slotCount * (BRICK_CELLS / 32))
        return;

    int start = element * 32;
    uint bits = 0;
    for (int k = 0; k < 32; k++)
    {
        int index = start + k;
        if ((cellState[index] & STATE_TRACKED) == 0 || fill[index] <= visibilityThreshold)
            continue;
        float speedSquared = 0.0f;
        for (int axis = 0; axis < 3; axis++)
        {
            float center = (ReadFace(axis, index) + ReadFace(axis, NeighborCell(index, axis, -1))) * 0.5f;
            speedSquared += center * center;
        }
        if (speedSquared > fastSpeedSquared)
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
    pass ExtrapolateIntoAir
    {
        ComputeShader = compile cs_5_0 extrapolateIntoAir();
    }

    pass AdvectVelocity
    {
        ComputeShader = compile cs_5_0 advectVelocity();
    }

    pass ZeroSolidFaces
    {
        ComputeShader = compile cs_5_0 zeroSolidFaces();
    }

    pass MarkLiquidNeighbors
    {
        ComputeShader = compile cs_5_0 markLiquidNeighbors();
    }

    pass BuildSystem
    {
        ComputeShader = compile cs_5_0 buildSystem();
    }

    pass InitSealedLabels
    {
        ComputeShader = compile cs_5_0 initSealedLabels();
    }

    pass StartSealedLabels
    {
        ComputeShader = compile cs_5_0 startSealedLabels();
    }

    pass FinishLabelRound
    {
        ComputeShader = compile cs_5_0 finishLabelRound();
    }

    pass ForceSettleSealedLabels
    {
        ComputeShader = compile cs_5_0 forceSettleSealedLabels();
    }

    pass PropagateSealedLabels
    {
        ComputeShader = compile cs_5_0 propagateSealedLabels();
    }

    pass AssignSealedGroups
    {
        ComputeShader = compile cs_5_0 assignSealedGroups();
    }

    pass SumSealedResiduals
    {
        ComputeShader = compile cs_5_0 sumSealedResiduals();
    }

    pass RemoveSealedMeans
    {
        ComputeShader = compile cs_5_0 removeSealedMeans();
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

    pass FillCorrectionTotals
    {
        ComputeShader = compile cs_5_0 fillCorrectionTotals();
    }

    pass FillSurfaceOrientation
    {
        ComputeShader = compile cs_5_0 fillSurfaceOrientation();
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

    pass ThinWaterFlow
    {
        ComputeShader = compile cs_5_0 thinWaterFlow();
    }

    pass ThinWaterApply
    {
        ComputeShader = compile cs_5_0 thinWaterApply();
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

    pass BuildFastMask
    {
        ComputeShader = compile cs_5_0 buildFastMask();
    }
}
