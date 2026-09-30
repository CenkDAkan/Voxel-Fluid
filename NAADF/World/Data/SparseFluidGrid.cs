using Microsoft.Xna.Framework;
using NAADF.Common;
using System;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * Sparse Volume-of-Fluid storage on a staggered (MAC) grid
    *
    * Each tracked cell holds a fill fraction in [0,1]; a cell not present holds no fluid
    * Velocity lives on cell faces, one component per face: faces[axis][Key(c)] is the velocity through the +axis face of cell c
    * A face not present has zero velocity
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

        private readonly Dictionary<long, Cell> cells = new Dictionary<long, Cell>();
        public Dictionary<long, float>[] faces = { new Dictionary<long, float>(), new Dictionary<long, float>(), new Dictionary<long, float>() };

        public int Count => cells.Count;

        public long[] SnapshotKeys() => new List<long>(cells.Keys).ToArray();

        public bool ContainsKey(long key) => cells.ContainsKey(key);

        public bool TryGetCell(long key, out Cell cell) => cells.TryGetValue(key, out cell);

        public Cell GetCell(long key)
        {
            cells.TryGetValue(key, out Cell cell);
            return cell;
        }

        public float GetFill(long key) => cells.TryGetValue(key, out Cell cell) ? cell.fill : 0f;

        public void SetCell(long key, Cell cell) => cells[key] = cell;

        public void RemoveKey(long key) => cells.Remove(key);

        public void Clear()
        {
            cells.Clear();
            foreach (Dictionary<long, float> f in faces)
                f.Clear();
        }

        public float GetFace(int axis, long key) => faces[axis].TryGetValue(key, out float v) ? v : 0f;

        public static long Key(Point3 p) => (long)p.X | ((long)p.Y << 21) | ((long)p.Z << 42);

        public static Point3 Unpack(long key)
        {
            int x = (int)(key & 0x1FFFFF);
            int y = (int)((key >> 21) & 0x1FFFFF);
            int z = (int)((key >> 42) & 0x1FFFFF);
            return new Point3(x, y, z);
        }

        // Trilinear sample of one velocity component at an arbitrary position (cell centers sit on integer coordinates)
        public float SampleComponent(int axis, Vector3 pos)
        {
            Vector3 q = pos;
            if (axis == 0) q.X -= 0.5f;
            else if (axis == 1) q.Y -= 0.5f;
            else q.Z -= 0.5f;

            int x0 = (int)MathF.Floor(q.X), y0 = (int)MathF.Floor(q.Y), z0 = (int)MathF.Floor(q.Z);
            float fx = q.X - x0, fy = q.Y - y0, fz = q.Z - z0;
            Dictionary<long, float> f = faces[axis];

            float Get(int x, int y, int z) => f.TryGetValue(Key(new Point3(x, y, z)), out float v) ? v : 0f;

            float c00 = MathHelper.Lerp(Get(x0, y0, z0), Get(x0 + 1, y0, z0), fx);
            float c10 = MathHelper.Lerp(Get(x0, y0 + 1, z0), Get(x0 + 1, y0 + 1, z0), fx);
            float c01 = MathHelper.Lerp(Get(x0, y0, z0 + 1), Get(x0 + 1, y0, z0 + 1), fx);
            float c11 = MathHelper.Lerp(Get(x0, y0 + 1, z0 + 1), Get(x0 + 1, y0 + 1, z0 + 1), fx);

            return MathHelper.Lerp(MathHelper.Lerp(c00, c10, fy), MathHelper.Lerp(c01, c11, fy), fz);
        }

        public Vector3 SampleVelocity(Vector3 pos) => new Vector3(SampleComponent(0, pos), SampleComponent(1, pos), SampleComponent(2, pos));

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
