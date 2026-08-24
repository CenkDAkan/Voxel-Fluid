using Microsoft.Xna.Framework.Graphics;
using NAADF.Common;

namespace NAADF.World.Generator
{
    /*
    * A world generator that explicitly clears each segment, for testing and benchmarking the fluid simulation in isolation
    */
    public class WorldGeneratorEmpty : WorldGenerator
    {
        private uint[] zeroSegment;

        public override bool IsValid()
        {
            return true;
        }
        public override void CopyToChunkData(Point3 chunkPos, Point3 chunkSize, Point3 sizeInVoxels, StructuredBuffer chunkDataGpu, int worldSizeY, int voxelGroupSize = 1)
        {
            zeroSegment ??= new uint[chunkDataGpu.ElementCount];
            chunkDataGpu.SetData(zeroSegment);
        }
    }
}
