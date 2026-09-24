using Microsoft.Xna.Framework;
using NAADF.Common;
using System;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * Sparse Volume-of-Fluid grid: a cell's fluid content is a fill fraction in [0,1] - 0 empty, 1 fully full
    * Only cells that actually contain fluid are tracked at all
    * A cell not present here unambiguously means no fluid there
    *
    * fill is moved between cells by direct face fluxes
    * Velocity is still advected by semi-Lagrangian sampling (SampleVelocity below)
    * losing a little accuracy in velocity doesn't make matter vanish, only losing it in the mass/presence field does
    */
    public class SparseFluidGrid
    {
        public struct Cell
        {
            public float fill;
            public Vector3 velocity;

            // Consecutive ticks this cell has sat at or below visibilityThreshold without gaining fill
            // Reset to 0 the instant a cell gains fill; a fresh cell (default struct) starts at 0
            public int ticksStagnant;
        }

        private readonly Dictionary<long, Cell> cells = new Dictionary<long, Cell>();

        public int Count => cells.Count;

        public long[] SnapshotKeys() => new List<long>(cells.Keys).ToArray();

        public bool ContainsKey(long key) => cells.ContainsKey(key);

        public bool TryGetCell(long key, out Cell cell) => cells.TryGetValue(key, out cell);

        public Cell GetCell(long key)
        {
            cells.TryGetValue(key, out Cell cell);
            return cell;
        }

        public void SetCell(long key, Cell cell) => cells[key] = cell;

        public void RemoveKey(long key) => cells.Remove(key);

        public void Clear() => cells.Clear();

        public void Set(Point3 p, Cell cell) => cells[Key(p)] = cell;

        public Vector3 GetVelocity(Point3 p) => cells.TryGetValue(Key(p), out Cell cell) ? cell.velocity : Vector3.Zero;
        public static long Key(Point3 p) => (long)p.X | ((long)p.Y << 21) | ((long)p.Z << 42);

        public static Point3 Unpack(long key)
        {
            int x = (int)(key & 0x1FFFFF);
            int y = (int)((key >> 21) & 0x1FFFFF);
            int z = (int)((key >> 42) & 0x1FFFFF);
            return new Point3(x, y, z);
        }

        // Trilinear blend of the 8 cells around pos, for a semi-Lagrangian back-traced velocity sample that
        // almost never lands exactly on a cell center. Untracked corners fall back to zero (no fluid, no velocity) rather than throwing.
        public Vector3 SampleVelocity(Vector3 pos)
        {
            Point3 c000 = new Point3((int)MathF.Floor(pos.X), (int)MathF.Floor(pos.Y), (int)MathF.Floor(pos.Z));
            Vector3 f = pos - c000.ToVector3();

            Vector3 x00 = Vector3.Lerp(GetVelocity(c000 + new Point3(0, 0, 0)), GetVelocity(c000 + new Point3(1, 0, 0)), f.X);
            Vector3 x10 = Vector3.Lerp(GetVelocity(c000 + new Point3(0, 1, 0)), GetVelocity(c000 + new Point3(1, 1, 0)), f.X);
            Vector3 x01 = Vector3.Lerp(GetVelocity(c000 + new Point3(0, 0, 1)), GetVelocity(c000 + new Point3(1, 0, 1)), f.X);
            Vector3 x11 = Vector3.Lerp(GetVelocity(c000 + new Point3(0, 1, 1)), GetVelocity(c000 + new Point3(1, 1, 1)), f.X);

            Vector3 y0 = Vector3.Lerp(x00, x10, f.Y);
            Vector3 y1 = Vector3.Lerp(x01, x11, f.Y);

            return Vector3.Lerp(y0, y1, f.Z);
        }
    }
}
