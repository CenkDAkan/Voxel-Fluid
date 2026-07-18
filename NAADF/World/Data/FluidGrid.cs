using Microsoft.Xna.Framework;
using NAADF.Common;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * A sparse empty/fluid grid, fully decoupled from the engine's solid-voxel chunk/block/pointer node tree.
    * Test worlds here already run into the billions of voxels, far past what a single dense int32-indexed array could address,
    * so positions are tracked in a dictionary instead. Memory scales with how many cells are actually fluid, not with world size.
    * Presence of a key is what makes a cell fluid; the value is that cell's velocity, which is the field the fluid simulation equations operate on.
    */
    public class FluidGrid
    {
        public readonly Point3 size;
        private readonly Dictionary<long, Vector3> velocities = new Dictionary<long, Vector3>();

        public FluidGrid(Point3 size)
        {
            this.size = size;
        }

        public bool IsInside(Point3 p)
        {
            return p.X >= 0 && p.Y >= 0 && p.Z >= 0
                && p.X < size.X && p.Y < size.Y && p.Z < size.Z;
        }

        public bool IsFluid(Point3 p)
        {
            return velocities.ContainsKey(Key(p));
        }

        public Vector3 GetVelocity(Point3 p)
        {
            velocities.TryGetValue(Key(p), out Vector3 velocity);
            return velocity;
        }

        // Only affects cells already marked fluid via SetFluid - callers are expected to set fluid state first.
        public void SetVelocity(Point3 p, Vector3 velocity)
        {
            if (IsInside(p))
                velocities[Key(p)] = velocity;
        }

        public void SetFluid(Point3 p, bool isFluid)
        {
            if (!IsInside(p))
                return;

            long key = Key(p);
            if (isFluid)
            {
                if (!velocities.ContainsKey(key))
                    velocities[key] = Vector3.Zero;
            }
            else
                velocities.Remove(key);
        }

        // Packs a world-voxel position into a single long, 21 bits per axis (up to ~2 million per axis),
        // well past any world size this engine will realistically use. Avoids ever computing a flat int32 index/volume.
        private static long Key(Point3 p)
        {
            return (long)p.X | ((long)p.Y << 21) | ((long)p.Z << 42);
        }
    }
}
