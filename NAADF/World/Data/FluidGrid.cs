using NAADF.Common;
using System.Collections.Generic;

namespace NAADF.World.Data
{
    /*
    * A sparse empty/fluid grid, fully decoupled from the engine's solid-voxel chunk/block/pointer node tree. 
    * Test worlds here already run into the billions of voxels, far past what a single dense int32-indexed array could address, 
    * so positions are tracked in a set instead. Memory scales with how many cells are actually fluid, not with world size.
    */
    public class FluidGrid
    {
        public readonly Point3 size;
        private readonly HashSet<long> fluidCells = new HashSet<long>();

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
            return fluidCells.Contains(Key(p));
        }

        public void SetFluid(Point3 p, bool isFluid)
        {
            if (!IsInside(p))
                return;

            if (isFluid)
                fluidCells.Add(Key(p));
            else
                fluidCells.Remove(Key(p));
        }

        // Packs a world-voxel position into a single long, 21 bits per axis (up to ~2 million per axis),
        // well past any world size this engine will realistically use. Avoids ever computing a flat int32 index/volume.
        private static long Key(Point3 p)
        {
            return (long)p.X | ((long)p.Y << 21) | ((long)p.Z << 42);
        }
    }
}
