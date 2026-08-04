using Microsoft.Xna.Framework;
using NAADF.Common;
using System;

namespace NAADF.World.Data
{
    /*
    * A dense, fixed-size fluid grid: reproduces the reference master's thesis's(Voxel-based fluid simulation in Unity) storage approach as a comparison
    * baseline against FluidGrid's sparse dictionary. Unlike FluidGrid, every cell in a bounded domain gets a slot
    * in a flat array up front and keeps it regardless of whether that cell currently holds any fluid, so both
    * memory and any full-domain sweep over it cost the same no matter how much of the domain is actually active
    * The domain is deliberately capped rather than spanning the full world, mirroring the thesis's own capped grid (120^3)
    */
    public class DenseFluidGrid
    {
        public readonly Point3 origin; // world-voxel position of this grid's local (0,0,0) cell, fixed for the grid's lifetime
        public readonly Point3 size;   // domain dimensions in voxels

        private readonly Vector3[] velocity;
        private readonly float[] density; // continuous fluid quantity per cell

        public DenseFluidGrid(Point3 origin, Point3 size)
        {
            this.origin = origin;
            this.size = size;

            int cellCount = size.X * size.Y * size.Z;
            velocity = new Vector3[cellCount];
            density = new float[cellCount];
        }

        // Every cell in size.X*Y*Z is always allocated, so this is a direct flatten
        // The difference between the two Index-equivalents is the actual variable this module exists to measure against FluidGrid
        public int Index(Point3 local)
        {
            return local.X + local.Y * size.X + local.Z * size.X * size.Y;
        }

        public bool IsInsideLocal(Point3 local)
        {
            return local.X >= 0 && local.Y >= 0 && local.Z >= 0
                && local.X < size.X && local.Y < size.Y && local.Z < size.Z;
        }

        public Point3 WorldToLocal(Point3 worldVoxel) => worldVoxel - origin;
        public Point3 LocalToWorld(Point3 local) => local + origin;

        public Vector3 GetVelocity(Point3 local) => IsInsideLocal(local) ? velocity[Index(local)] : Vector3.Zero;

        public void SetVelocity(Point3 local, Vector3 value)
        {
            if (IsInsideLocal(local))
                velocity[Index(local)] = value;
        }

        public float GetDensity(Point3 local) => IsInsideLocal(local) ? density[Index(local)] : 0f;

        public void SetDensity(Point3 local, float value)
        {
            if (IsInsideLocal(local))
                density[Index(local)] = value;
        }

        // The 8 integer cells surrounding a continuous local-space position, plus how far into that cell the position sits 
        // on each axis (0..1) which is shared by SampleDensity/SampleVelocity below so the mapping from a
        // back-traced position to its surrounding cells only lives in one place
        private void TrilinearCorners(Vector3 localPos, out Point3 c000, out Vector3 frac)
        {
            c000 = new Point3((int)MathF.Floor(localPos.X), (int)MathF.Floor(localPos.Y), (int)MathF.Floor(localPos.Z));
            frac = localPos - c000.ToVector3();
        }

        // Trilinearly blends the 8 cells around localPos so a semi-Lagrangian back-traced position (which almost
        // never lands exactly on a cell center) gets a smooth read instead of snapping to one cell
        // to keep advection from looking blocky/aliased
        public float SampleDensity(Vector3 localPos)
        {
            TrilinearCorners(localPos, out Point3 c, out Vector3 f);

            float x00 = MathHelper.Lerp(GetDensity(c + new Point3(0, 0, 0)), GetDensity(c + new Point3(1, 0, 0)), f.X);
            float x10 = MathHelper.Lerp(GetDensity(c + new Point3(0, 1, 0)), GetDensity(c + new Point3(1, 1, 0)), f.X);
            float x01 = MathHelper.Lerp(GetDensity(c + new Point3(0, 0, 1)), GetDensity(c + new Point3(1, 0, 1)), f.X);
            float x11 = MathHelper.Lerp(GetDensity(c + new Point3(0, 1, 1)), GetDensity(c + new Point3(1, 1, 1)), f.X);

            float y0 = MathHelper.Lerp(x00, x10, f.Y);
            float y1 = MathHelper.Lerp(x01, x11, f.Y);

            return MathHelper.Lerp(y0, y1, f.Z);
        }

        public Vector3 SampleVelocity(Vector3 localPos)
        {
            TrilinearCorners(localPos, out Point3 c, out Vector3 f);

            Vector3 x00 = Vector3.Lerp(GetVelocity(c + new Point3(0, 0, 0)), GetVelocity(c + new Point3(1, 0, 0)), f.X);
            Vector3 x10 = Vector3.Lerp(GetVelocity(c + new Point3(0, 1, 0)), GetVelocity(c + new Point3(1, 1, 0)), f.X);
            Vector3 x01 = Vector3.Lerp(GetVelocity(c + new Point3(0, 0, 1)), GetVelocity(c + new Point3(1, 0, 1)), f.X);
            Vector3 x11 = Vector3.Lerp(GetVelocity(c + new Point3(0, 1, 1)), GetVelocity(c + new Point3(1, 1, 1)), f.X);

            Vector3 y0 = Vector3.Lerp(x00, x10, f.Y);
            Vector3 y1 = Vector3.Lerp(x01, x11, f.Y);

            return Vector3.Lerp(y0, y1, f.Z);
        }
    }
}
