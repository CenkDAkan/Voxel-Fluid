namespace NAADF.World.Generator
{
    /*
    * A world generator that does nothing, for testing and benchmarking the fluid simulation in isolation
    */
    public class WorldGeneratorEmpty : WorldGenerator
    {
        public override bool IsValid()
        {
            return true;
        }
    }
}
