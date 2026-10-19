using Microsoft.Xna.Framework;
using NAADF.Common;
using System;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * Sparse Volume-of-Fluid storage on a staggered (MAC) grid, held in bricks of 8^3 cells
    *
    * Bricks are taken from a pool wherever there is fluid; every per-cell value lives in a flat array indexed by brick slot * 512 + local index
    * A tracked cell holds a fill fraction in [0,1]; an untracked cell holds no fluid
    * Velocity lives on cell faces, one component per face: face[axis][i] is the velocity through the +axis face of cell i
    * A face that is not present has zero velocity
    */
    public class SparseFluidGrid
    {
        public struct Cell
        {
            public float fill;
            public int ticksStagnant; // consecutive ticks at or below the visibility threshold without gaining fill
        }

        public static readonly Point3[] AxisOffsets =
        {
            new Point3(1, 0, 0), new Point3(0, 1, 0), new Point3(0, 0, 1),
        };

        public const int BrickShift = 3;
        public const int BrickSize = 1 << BrickShift;
        public const int BrickMask = BrickSize - 1;
        public const int BrickCells = BrickSize * BrickSize * BrickSize;
        public const int CellShift = BrickShift * 3;

        public const byte SolidFree = 1, SolidBlocked = 2, SolidDrain = 4;

        // Layout of a cell's state word, as used by PackState and UnpackState: the flags byte sits at StateFlagsShift,
        // one bit per axis from StatePresent on says whether the face is present, ticksStagnant fills the upper half
        public const uint StateTracked = 1, StatePresent = 16;
        public const int StateFlagsShift = 1, StateFlagsMask = 7, StateStagnantShift = 16;

        private const int NeighborCount = 27;
        private const int NeighborSelf = 13;
        private static readonly int[] NeighborStep = { 1, 3, 9 };

        // Per-cell data
        public byte[] tracked;
        public float[] fill;
        public int[] ticksStagnant;
        public float[][] face = new float[3][];
        public byte[][] facePresent = new byte[3][];
        public float[][] faceScratch = new float[3][];

        // Per-cell working data owned by the solver
        public byte[] solid;      // 0 = not known, otherwise SolidFree or SolidBlocked, plus SolidDrain for a cell that empties itself
        public byte[] flags;
        public float[] pressure;
        public int[] liquidIndex; // index into the pressure system + 1, 0 = not liquid
        public float[] outTotal, outScaled, inTotal, delta;

        // Per-brick data
        private int[] brickX, brickY, brickZ;
        private bool[] brickAllocated;
        private int[] brickTracked;
        private byte[] brickBorder; // whether tracked cells lie along each of the brick's six faces: bit axis * 2 + (0 = low side, 1 = high side)
        public int[] brickFacesPresent;
        private int[] neighbor; // slot of each of the 27 surrounding bricks (self included), -1 where there is none
        private bool[] brickNeeded;

        private readonly Dictionary<long, int> brickMap = new Dictionary<long, int>();
        private readonly Stack<int> freeSlots = new Stack<int>();
        private int slotCapacity = 0;

        public int SlotCount { get; private set; }
        public int Count { get; private set; }
        public int BrickCount => brickMap.Count;

        // Creates an empty grid with room for 16 bricks
        public SparseFluidGrid()
        {
            Resize(16);
        }

        // Grows every per-cell and per-brick array to hold newCapacity bricks, keeping their contents
        // The arrays are replaced, so references to them taken before a call that can allocate a brick are stale afterwards
        private void Resize(int newCapacity)
        {
            int cells = newCapacity * BrickCells;
            Array.Resize(ref tracked, cells);
            Array.Resize(ref fill, cells);
            Array.Resize(ref ticksStagnant, cells);
            for (int axis = 0; axis < 3; axis++)
            {
                Array.Resize(ref face[axis], cells);
                Array.Resize(ref facePresent[axis], cells);
                Array.Resize(ref faceScratch[axis], cells);
            }
            Array.Resize(ref solid, cells);
            Array.Resize(ref flags, cells);
            Array.Resize(ref pressure, cells);
            Array.Resize(ref liquidIndex, cells);
            Array.Resize(ref outTotal, cells);
            Array.Resize(ref outScaled, cells);
            Array.Resize(ref inTotal, cells);
            Array.Resize(ref delta, cells);

            Array.Resize(ref brickX, newCapacity);
            Array.Resize(ref brickY, newCapacity);
            Array.Resize(ref brickZ, newCapacity);
            Array.Resize(ref brickAllocated, newCapacity);
            Array.Resize(ref brickTracked, newCapacity);
            Array.Resize(ref brickBorder, newCapacity);
            Array.Resize(ref brickFacesPresent, newCapacity);
            Array.Resize(ref brickNeeded, newCapacity);
            Array.Resize(ref neighbor, newCapacity * NeighborCount);
            slotCapacity = newCapacity;
        }

        // Zeroes all per-cell data of one brick slot
        private void ClearSlotData(int slot)
        {
            int start = slot * BrickCells;
            Array.Clear(tracked, start, BrickCells);
            Array.Clear(fill, start, BrickCells);
            Array.Clear(ticksStagnant, start, BrickCells);
            for (int axis = 0; axis < 3; axis++)
            {
                Array.Clear(face[axis], start, BrickCells);
                Array.Clear(facePresent[axis], start, BrickCells);
                Array.Clear(faceScratch[axis], start, BrickCells);
            }
            Array.Clear(solid, start, BrickCells);
            Array.Clear(flags, start, BrickCells);
            Array.Clear(pressure, start, BrickCells);
            Array.Clear(liquidIndex, start, BrickCells);
            Array.Clear(outTotal, start, BrickCells);
            Array.Clear(outScaled, start, BrickCells);
            Array.Clear(inTotal, start, BrickCells);
            Array.Clear(delta, start, BrickCells);
        }

        // Removes every brick and cell, leaving an empty grid that keeps its allocated capacity
        public void Clear()
        {
            for (int slot = 0; slot < SlotCount; slot++)
            {
                if (brickAllocated[slot])
                    ClearSlotData(slot);
                brickAllocated[slot] = false;
                brickTracked[slot] = 0;
                brickBorder[slot] = 0;
                brickFacesPresent[slot] = 0;
            }
            brickMap.Clear();
            freeSlots.Clear();
            SlotCount = 0;
            Count = 0;
        }

        // ---- Keys (world voxel position packed into a long, 21 bits per axis) ----

        // Packs a non-negative cell position into a key
        public static long Key(Point3 p) => (long)p.X | ((long)p.Y << 21) | ((long)p.Z << 42);

        // Recovers the cell position from a key made by Key
        public static Point3 Unpack(long key)
        {
            int x = (int)(key & 0x1FFFFF);
            int y = (int)((key >> 21) & 0x1FFFFF);
            int z = (int)((key >> 42) & 0x1FFFFF);
            return new Point3(x, y, z);
        }

        // Dictionary key of a brick position; each axis is masked to 21 bits so negative brick positions get a key of their own
        private static long BrickKey(int bx, int by, int bz) => ((long)bx & 0x1FFFFF) | (((long)by & 0x1FFFFF) << 21) | (((long)bz & 0x1FFFFF) << 42);

        // ---- Bricks ----

        // Whether a slot currently holds a brick
        public bool IsSlotAllocated(int slot) => brickAllocated[slot];

        // The raw neighbor table, 27 slots per brick slot, for uploading to the GPU
        public int[] NeighborTable => neighbor;

        // Forgets the cached solid state of every cell in the bricks overlapping a box of cells (bounds inclusive)
        public void ClearSolid(Point3 min, Point3 max)
        {
            for (int bz = min.Z >> BrickShift; bz <= max.Z >> BrickShift; bz++)
                for (int by = min.Y >> BrickShift; by <= max.Y >> BrickShift; by++)
                    for (int bx = min.X >> BrickShift; bx <= max.X >> BrickShift; bx++)
                    {
                        int slot = FindBrick(bx, by, bz);
                        if (slot >= 0)
                            Array.Clear(solid, slot << CellShift, BrickCells);
                    }
        }

        // Writes four ints per slot: brick x, y, z and whether the slot is allocated
        public void FillBrickInfo(int[] target)
        {
            for (int slot = 0; slot < SlotCount; slot++)
            {
                target[slot * 4] = brickX[slot];
                target[slot * 4 + 1] = brickY[slot];
                target[slot * 4 + 2] = brickZ[slot];
                target[slot * 4 + 3] = brickAllocated[slot] ? 1 : 0;
            }
        }

        // Number of tracked cells in a brick slot
        public int TrackedInSlot(int slot) => brickTracked[slot];

        // World position of the lowest corner cell of a brick slot
        public Point3 SlotOrigin(int slot) => new Point3(brickX[slot] << BrickShift, brickY[slot] << BrickShift, brickZ[slot] << BrickShift);

        // Slot of the brick at a brick position, or -1 when there is none
        private int FindBrick(int bx, int by, int bz) => brickMap.TryGetValue(BrickKey(bx, by, bz), out int slot) ? slot : -1;

        // Brick lookup that goes through the neighbor table of a nearby brick when it can
        private int FindBrickNear(int hintSlot, int bx, int by, int bz)
        {
            if (hintSlot >= 0)
            {
                int dx = bx - brickX[hintSlot], dy = by - brickY[hintSlot], dz = bz - brickZ[hintSlot];
                if (dx >= -1 && dx <= 1 && dy >= -1 && dy <= 1 && dz >= -1 && dz <= 1)
                    return neighbor[hintSlot * NeighborCount + NeighborSelf + dx + dy * 3 + dz * 9];
            }
            return FindBrick(bx, by, bz);
        }

        // Slot of the brick at a brick position, allocating it (reusing a freed slot first) and linking it to its neighbors when missing
        private int EnsureBrick(int bx, int by, int bz)
        {
            long key = BrickKey(bx, by, bz);
            if (brickMap.TryGetValue(key, out int slot))
                return slot;

            if (freeSlots.Count > 0)
                slot = freeSlots.Pop();
            else
            {
                if (SlotCount == slotCapacity)
                    Resize(slotCapacity * 2);
                slot = SlotCount++;
            }

            brickMap[key] = slot;
            brickAllocated[slot] = true;
            brickX[slot] = bx;
            brickY[slot] = by;
            brickZ[slot] = bz;
            brickTracked[slot] = 0;
            brickBorder[slot] = 0;
            brickFacesPresent[slot] = 0;

            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int direction = dx + dy * 3 + dz * 9;
                        int other = direction == 0 ? slot : FindBrick(bx + dx, by + dy, bz + dz);
                        neighbor[slot * NeighborCount + NeighborSelf + direction] = other;
                        if (other >= 0)
                            neighbor[other * NeighborCount + NeighborSelf - direction] = slot;
                    }
            return slot;
        }

        // Unlinks a brick from its neighbors, clears its data and returns its slot to the free list
        private void FreeBrick(int slot)
        {
            for (int direction = -NeighborSelf; direction <= NeighborSelf; direction++)
            {
                int other = neighbor[slot * NeighborCount + NeighborSelf + direction];
                if (other >= 0 && other != slot)
                    neighbor[other * NeighborCount + NeighborSelf - direction] = -1;
                neighbor[slot * NeighborCount + NeighborSelf + direction] = -1;
            }

            Count -= brickTracked[slot];
            ClearSlotData(slot);
            brickMap.Remove(BrickKey(brickX[slot], brickY[slot], brickZ[slot]));
            brickAllocated[slot] = false;
            brickTracked[slot] = 0;
            brickBorder[slot] = 0;
            brickFacesPresent[slot] = 0;
            freeSlots.Push(slot);
        }

        // Works out, from the cells, along which faces of each brick tracked cells lie
        public void ScanBorders()
        {
            for (int slot = 0; slot < SlotCount; slot++)
            {
                brickBorder[slot] = 0;
                if (!brickAllocated[slot] || brickTracked[slot] == 0)
                    continue;

                int start = slot * BrickCells;
                for (int axis = 0; axis < 3; axis++)
                {
                    int stride = 1 << (BrickShift * axis);
                    int strideA = 1 << (BrickShift * ((axis + 1) % 3));
                    int strideB = 1 << (BrickShift * ((axis + 2) % 3));
                    for (int side = 0; side < 2; side++)
                    {
                        int layer = start + (side == 0 ? 0 : BrickMask * stride);
                        bool any = false;
                        for (int a = 0; a < BrickSize && !any; a++)
                            for (int b = 0; b < BrickSize; b++)
                                if (tracked[layer + a * strideA + b * strideB] != 0)
                                {
                                    any = true;
                                    break;
                                }
                        if (any)
                            brickBorder[slot] |= (byte)(1 << (axis * 2 + side));
                    }
                }
            }
        }

        // Takes the per-brick figures from a summary computed elsewhere instead of from the cells:
        // bits 0-5 as in brickBorder, bit 6 any face present, bits 8-17 the number of tracked cells
        public void SetBrickSummary(uint[] summary, int slots)
        {
            int count = 0;
            for (int slot = 0; slot < slots; slot++)
            {
                uint value = brickAllocated[slot] ? summary[slot] : 0;
                brickBorder[slot] = (byte)(value & 0x3F);
                brickFacesPresent[slot] = (int)((value >> 6) & 1);
                brickTracked[slot] = (int)((value >> 8) & 0x3FF);
                count += brickTracked[slot];
            }
            Count = count;
        }

        // Allocates the bricks next to any brick face that has tracked cells along it, and frees bricks holding
        // no tracked cells and no present faces that aren't such a neighbor
        public void UpdateTopology()
        {
            int slotEnd = SlotCount;
            for (int slot = 0; slot < slotEnd; slot++)
                brickNeeded[slot] = brickAllocated[slot] && (brickTracked[slot] > 0 || brickFacesPresent[slot] > 0);

            for (int slot = 0; slot < slotEnd; slot++)
            {
                if (!brickAllocated[slot] || brickTracked[slot] == 0)
                    continue;

                for (int axis = 0; axis < 3; axis++)
                    for (int side = 0; side < 2; side++)
                    {
                        if ((brickBorder[slot] & (1 << (axis * 2 + side))) == 0)
                            continue;

                        int dir = side == 0 ? -1 : 1;
                        int other = EnsureBrick(brickX[slot] + (axis == 0 ? dir : 0), brickY[slot] + (axis == 1 ? dir : 0), brickZ[slot] + (axis == 2 ? dir : 0));
                        brickNeeded[other] = true;
                    }
            }

            for (int slot = 0; slot < slotEnd; slot++)
                if (brickAllocated[slot] && !brickNeeded[slot])
                    FreeBrick(slot);
        }

        // Writes one state word per cell: tracked, the solver's flags, which faces are present and ticksStagnant
        public void PackState(uint[] target, int cells)
        {
            for (int i = 0; i < cells; i++)
            {
                uint word = 0;
                if (tracked[i] != 0)
                    word = StateTracked | (uint)((flags[i] & StateFlagsMask) << StateFlagsShift) | ((uint)Math.Min(ticksStagnant[i], 0xFFFF) << StateStagnantShift);
                if (facePresent[0][i] != 0) word |= StatePresent;
                if (facePresent[1][i] != 0) word |= StatePresent << 1;
                if (facePresent[2][i] != 0) word |= StatePresent << 2;
                target[i] = word;
            }
        }

        // The reverse of PackState, including the per-brick counts that follow from the cells
        public void UnpackState(uint[] source, int cells)
        {
            Array.Clear(brickTracked, 0, SlotCount);
            Array.Clear(brickFacesPresent, 0, SlotCount);
            int count = 0;
            for (int i = 0; i < cells; i++)
            {
                uint word = source[i];
                bool isTracked = (word & StateTracked) != 0;
                tracked[i] = isTracked ? (byte)1 : (byte)0;
                flags[i] = isTracked ? (byte)((word >> StateFlagsShift) & StateFlagsMask) : (byte)0;
                ticksStagnant[i] = isTracked ? (int)(word >> StateStagnantShift) : 0;
                int present = 0;
                for (int axis = 0; axis < 3; axis++)
                {
                    byte value = (word & (StatePresent << axis)) != 0 ? (byte)1 : (byte)0;
                    facePresent[axis][i] = value;
                    present += value;
                }

                if (isTracked)
                {
                    brickTracked[i >> CellShift]++;
                    count++;
                }
                brickFacesPresent[i >> CellShift] += present;
            }
            Count = count;
        }

        // Cell indices

        // Index of a cell within its brick, from its world position
        private static int LocalIndex(int x, int y, int z) => (x & BrickMask) | ((y & BrickMask) << BrickShift) | ((z & BrickMask) << (BrickShift * 2));

        // Index of the cell at a world position, or -1 when no brick covers it
        public int CellIndex(int x, int y, int z)
        {
            int slot = FindBrick(x >> BrickShift, y >> BrickShift, z >> BrickShift);
            return slot < 0 ? -1 : (slot << CellShift) | LocalIndex(x, y, z);
        }

        // Index of the cell at a world position, or -1 when no brick covers it
        public int CellIndex(Point3 p) => CellIndex(p.X, p.Y, p.Z);

        // Index of the cell at a world position, allocating its brick when missing
        public int EnsureCellIndex(Point3 p)
        {
            int slot = EnsureBrick(p.X >> BrickShift, p.Y >> BrickShift, p.Z >> BrickShift);
            return (slot << CellShift) | LocalIndex(p.X, p.Y, p.Z);
        }

        // World position of the cell at an index
        public Point3 CellPosition(int index)
        {
            int slot = index >> CellShift;
            return new Point3(
                (brickX[slot] << BrickShift) | (index & BrickMask),
                (brickY[slot] << BrickShift) | ((index >> BrickShift) & BrickMask),
                (brickZ[slot] << BrickShift) | ((index >> (BrickShift * 2)) & BrickMask));
        }

        // Index of the cell one step along an axis (direction +1 or -1), or -1 when no brick covers it
        public int NeighborIndex(int index, int axis, int direction)
        {
            int shift = BrickShift * axis;
            int coordinate = (index >> shift) & BrickMask;
            if (direction > 0 ? coordinate < BrickMask : coordinate > 0)
                return index + (direction << shift);

            int other = neighbor[(index >> CellShift) * NeighborCount + NeighborSelf + direction * NeighborStep[axis]];
            if (other < 0)
                return -1;
            return (other << CellShift) | ((index & (BrickCells - 1)) - direction * (BrickMask << shift));
        }

        // Cells

        // Marks a cell as holding fluid; its fill is left for the caller to set
        public void Track(int index)
        {
            if (tracked[index] != 0)
                return;
            tracked[index] = 1;
            brickTracked[index >> CellShift]++;
            Count++;
        }

        // Marks a cell as holding no fluid and zeroes its fill and stagnation counter
        public void Untrack(int index)
        {
            if (tracked[index] == 0)
                return;
            tracked[index] = 0;
            fill[index] = 0f;
            ticksStagnant[index] = 0;
            brickTracked[index >> CellShift]--;
            Count--;
        }

        // Keys of all tracked cells, as a new array
        public long[] SnapshotKeys()
        {
            long[] keys = new long[Count];
            int n = 0;
            for (int slot = 0; slot < SlotCount; slot++)
            {
                if (brickTracked[slot] == 0)
                    continue;
                int start = slot << CellShift;
                for (int local = 0; local < BrickCells; local++)
                    if (tracked[start + local] != 0)
                        keys[n++] = Key(CellPosition(start + local));
            }
            return keys;
        }

        // Fill and stagnation counter of the cell at a key, false when the cell isn't tracked
        public bool TryGetCell(long key, out Cell cell)
        {
            int index = CellIndex(Unpack(key));
            if (index < 0 || tracked[index] == 0)
            {
                cell = default;
                return false;
            }
            cell = new Cell { fill = fill[index], ticksStagnant = ticksStagnant[index] };
            return true;
        }

        // Fill and stagnation counter of the cell at a key, all zero when the cell isn't tracked
        public Cell GetCell(long key)
        {
            TryGetCell(key, out Cell cell);
            return cell;
        }

        // Fill of the cell at a key, 0 when no brick covers it
        public float GetFill(long key)
        {
            int index = CellIndex(Unpack(key));
            return index < 0 ? 0f : fill[index];
        }

        // Tracks the cell at a key, allocating its brick when missing, and sets its fill and stagnation counter
        public void SetCell(long key, Cell cell)
        {
            int index = EnsureCellIndex(Unpack(key));
            Track(index);
            fill[index] = cell.fill;
            ticksStagnant[index] = cell.ticksStagnant;
        }

        // Untracks the cell at a key, if a brick covers it
        public void RemoveKey(long key)
        {
            int index = CellIndex(Unpack(key));
            if (index >= 0)
                Untrack(index);
        }

        // Faces

        // Sets the velocity through the +axis face of the cell at an index and marks that face present
        public void SetFaceAt(int axis, int index, float value)
        {
            face[axis][index] = value;
            if (facePresent[axis][index] == 0)
            {
                facePresent[axis][index] = 1;
                brickFacesPresent[index >> CellShift]++;
            }
        }

        // Sets the velocity through the +axis face of the cell at a position, allocating its brick when missing
        public void SetFace(int axis, Point3 lowerCell, float value) => SetFaceAt(axis, EnsureCellIndex(lowerCell), value);

        // Velocity through the +axis face of the cell at a key, 0 when no brick covers it
        public float GetFace(int axis, long key)
        {
            int index = CellIndex(Unpack(key));
            return index < 0 ? 0f : face[axis][index];
        }

        // Velocity through the +axis face of the cell at a key; false when that face isn't present
        public bool TryGetFace(int axis, long key, out float value)
        {
            int index = CellIndex(Unpack(key));
            if (index < 0 || facePresent[axis][index] == 0)
            {
                value = 0f;
                return false;
            }
            value = face[axis][index];
            return true;
        }

        // Every present face of one axis as (lower cell position, velocity), for diagnostics
        public IEnumerable<KeyValuePair<Point3, float>> PresentFaces(int axis)
        {
            for (int slot = 0; slot < SlotCount; slot++)
            {
                if (!brickAllocated[slot] || brickFacesPresent[slot] == 0)
                    continue;
                int start = slot << CellShift;
                for (int local = 0; local < BrickCells; local++)
                    if (facePresent[axis][start + local] != 0)
                        yield return new KeyValuePair<Point3, float>(CellPosition(start + local), face[axis][start + local]);
            }
        }

        // Trilinear sample of one velocity component at an arbitrary position (cell centers sit on integer coordinates)
        // hintSlot is any brick near the position, used to skip the brick lookup
        public float SampleComponent(int axis, Vector3 pos, int hintSlot = -1)
        {
            Vector3 q = pos;
            if (axis == 0) q.X -= 0.5f;
            else if (axis == 1) q.Y -= 0.5f;
            else q.Z -= 0.5f;

            int x0 = (int)MathF.Floor(q.X), y0 = (int)MathF.Floor(q.Y), z0 = (int)MathF.Floor(q.Z);
            float fx = q.X - x0, fy = q.Y - y0, fz = q.Z - z0;
            float[] f = face[axis];

            float v000, v100, v010, v110, v001, v101, v011, v111;
            int lx = x0 & BrickMask, ly = y0 & BrickMask, lz = z0 & BrickMask;
            if (lx < BrickMask && ly < BrickMask && lz < BrickMask)
            {
                int slot = FindBrickNear(hintSlot, x0 >> BrickShift, y0 >> BrickShift, z0 >> BrickShift);
                if (slot < 0)
                    return 0f;

                int i = (slot << CellShift) | lx | (ly << BrickShift) | (lz << (BrickShift * 2));
                const int sy = BrickSize, sz = BrickSize * BrickSize;
                v000 = f[i]; v100 = f[i + 1];
                v010 = f[i + sy]; v110 = f[i + sy + 1];
                v001 = f[i + sz]; v101 = f[i + sz + 1];
                v011 = f[i + sz + sy]; v111 = f[i + sz + sy + 1];
            }
            else
            {
                v000 = FaceNear(f, hintSlot, x0, y0, z0); v100 = FaceNear(f, hintSlot, x0 + 1, y0, z0);
                v010 = FaceNear(f, hintSlot, x0, y0 + 1, z0); v110 = FaceNear(f, hintSlot, x0 + 1, y0 + 1, z0);
                v001 = FaceNear(f, hintSlot, x0, y0, z0 + 1); v101 = FaceNear(f, hintSlot, x0 + 1, y0, z0 + 1);
                v011 = FaceNear(f, hintSlot, x0, y0 + 1, z0 + 1); v111 = FaceNear(f, hintSlot, x0 + 1, y0 + 1, z0 + 1);
            }

            float c00 = MathHelper.Lerp(v000, v100, fx);
            float c10 = MathHelper.Lerp(v010, v110, fx);
            float c01 = MathHelper.Lerp(v001, v101, fx);
            float c11 = MathHelper.Lerp(v011, v111, fx);

            return MathHelper.Lerp(MathHelper.Lerp(c00, c10, fy), MathHelper.Lerp(c01, c11, fy), fz);
        }

        // Value of one face array at a cell position, 0 when no brick covers it; hintSlot as in SampleComponent
        private float FaceNear(float[] f, int hintSlot, int x, int y, int z)
        {
            int slot = FindBrickNear(hintSlot, x >> BrickShift, y >> BrickShift, z >> BrickShift);
            return slot < 0 ? 0f : f[(slot << CellShift) | LocalIndex(x, y, z)];
        }

        // Trilinear sample of the full velocity at an arbitrary position, one component at a time
        public Vector3 SampleVelocity(Vector3 pos, int hintSlot = -1) => new Vector3(SampleComponent(0, pos, hintSlot), SampleComponent(1, pos, hintSlot), SampleComponent(2, pos, hintSlot));

        // Average of each axis' two faces, for diagnostics
        public Vector3 CenterVelocity(Point3 c)
        {
            long key = Key(c);
            return new Vector3(
                (GetFace(0, key) + GetFace(0, Key(c - AxisOffsets[0]))) * 0.5f,
                (GetFace(1, key) + GetFace(1, Key(c - AxisOffsets[1]))) * 0.5f,
                (GetFace(2, key) + GetFace(2, Key(c - AxisOffsets[2]))) * 0.5f);
        }
    }
}
