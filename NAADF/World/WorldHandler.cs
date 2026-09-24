using Microsoft.Xna.Framework;
using NAADF.Common;
using NAADF.Gui;
using NAADF.World.Data;
using NAADF.World.Generator;
using NAADF.World.Model;
using NAADF.World.Render;
using System;
using System.IO;

namespace NAADF.World
{
    public class WorldHandler
    {
        public WorldGeneratorModel worldGenerator;
        public WorldData worldData;
        public PathHandler pathHandler;
        public VoxelTypeHandler voxelTypeHandler;
        public ModelHandler modelHandler;
        public int worldGenSegmentSizeInGroups = 4; // A group is 4^3 chunks or 64^3 voxels
        public Point3 worldSizeToUseInWorldGenSegments = new Point3(16, 2, 16);
        private uint testFloorTypeRenderIndex;
        private bool testFloorTypeRegistered = false;

        public WorldHandler()
        {
            modelHandler = new ModelHandler();
            voxelTypeHandler = new VoxelTypeHandler();
            WorldRender.Initialize();
            pathHandler = new PathHandler();
        }

        public void Initialize()
        {
            worldData = new WorldData(worldSizeToUseInWorldGenSegments * worldGenSegmentSizeInGroups * 64, worldGenSegmentSizeInGroups);

            // Try to load default sample scene
            LoadModelScene("Content\\oasis.cvox");
        }

        public void LoadModelScene(string fileName)
        {
            worldGenerator = new WorldGeneratorModel();

            if (File.Exists(fileName))
            {
                ModelData modelData = null;
                if (fileName.EndsWith(".cvox"))
                    modelData = ModelData.Load(fileName);
                else if (fileName.EndsWith(".vox"))
                    modelData = ModelData.ImportFromVox(fileName);
                else if (fileName.EndsWith(".vl32"))
                    modelData = ModelData.ImportFromVL32(fileName);

                worldGenerator.SetModel(modelData);
            }

            worldData.GenerateWorld(worldGenerator);
        }

        public void ApplyAndGenerateNewWorldData(WorldData newWorldData, WorldGenerator generator)
        {
            worldData?.Dispose();
            worldData = newWorldData;
            worldData.GenerateWorld(generator);
        }

        private static readonly Point3 flatSceneAnchor = new Point3(2048, 200, 2048);

        public void LoadEmptyTestScene()
        {
            worldData.ApplyFluidSimulationMode(FluidSimulationMode.None);
            worldData.GenerateWorld(new WorldGeneratorEmpty());

            WorldRender.camera.SetPos(new Vector3(flatSceneAnchor.X, flatSceneAnchor.Y + 40, flatSceneAnchor.Z));
            WorldRender.camera.SetDir(Vector3.UnitZ);
        }

        // Goes back to the default oasis scene
        public void LoadOasisScene()
        {
            worldData.ApplyFluidSimulationMode(FluidSimulationMode.None);
            LoadModelScene("Content\\oasis.cvox");

            WorldRender.camera.SetPos(new Vector3(500, 200, 40));
        }

        private void EnsureTestFloorTypeRegistered()
        {
            if (testFloorTypeRegistered)
                return;

            VoxelType testFloorType = new VoxelType
            {
                ID = "fluid_test_floor",
                colorBase = new Vector3(0.6f, 0.6f, 0.6f),
                colorLayered = Vector3.Zero,
                materialBase = MaterialTypeBase.Diffuse,
                materialLayer = MaterialTypeLayer.None,
                roughness = 0.8f,
            };
            testFloorTypeRenderIndex = voxelTypeHandler.ApplyVoxelType(testFloorType).renderIndex;
            testFloorTypeRegistered = true;
        }

        private void CarveTestFloorVoxel(Point3 worldVoxel)
        {
            Point3 chunkPos = worldVoxel / 16;
            Point3 voxelPosInChunk = worldVoxel % 16;
            uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
            worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, (1u << 15) | testFloorTypeRenderIndex);
        }

        public void BuildFlatTestFloor()
        {
            EnsureTestFloorTypeRegistered();

            const int halfExtent = 64; // 128x128 pad, comfortably covers a 24-64 voxel fluid domain placed nearby
            const int floorThickness = 4;

            int minX = Math.Clamp(flatSceneAnchor.X - halfExtent, 0, worldData.sizeInVoxels.X - 1);
            int maxX = Math.Clamp(flatSceneAnchor.X + halfExtent, 0, worldData.sizeInVoxels.X - 1);
            int minZ = Math.Clamp(flatSceneAnchor.Z - halfExtent, 0, worldData.sizeInVoxels.Z - 1);
            int maxZ = Math.Clamp(flatSceneAnchor.Z + halfExtent, 0, worldData.sizeInVoxels.Z - 1);
            int floorTopY = Math.Clamp(flatSceneAnchor.Y, floorThickness, worldData.sizeInVoxels.Y - 1);

            for (int x = minX; x <= maxX; x++)
                for (int z = minZ; z <= maxZ; z++)
                    for (int y = floorTopY - floorThickness + 1; y <= floorTopY; y++)
                        CarveTestFloorVoxel(new Point3(x, y, z));

            worldData.editingHandler.processChunks(false);
        }

        // Same flat floor as BuildFlatTestFloor, plus a solid ring of walls around its perimeter (open at the
        // top, hollow in the middle) - a basin, specifically for testing genuinely RESTING fluid. An open flat
        // floor has nothing to stop a puddle from spreading indefinitely thinner forever (confirmed during the
        // sparse Eulerian arm's investigation: a seeded blob on the flat floor never stopped spreading and
        // eventually thinned past the visibility threshold entirely, even once mass conservation and the fall/
        // contact bugs were fixed - there was no bug left at that point, an unbounded floor just has no
        // equilibrium footprint to settle into). A bounded basin sidesteps that by giving the fluid somewhere to
        // actually pool - the same reason Minecraft/Terraria-style voxel fluids need a container to show
        // standing water rather than an ever-spreading film on open ground.
        public void BuildPitTestFloor()
        {
            EnsureTestFloorTypeRegistered();

            // Inverted (stepped) pyramid, not a flat-bottomed basin (2026-09-04) - roughly a quarter the area of
            // the original flat-bottomed pit, deepest at the center, sloping up to the rim at the interior's edge,
            // for a seeded blob to visibly settle at the bottom rather than needing the full open interior to
            // come to rest. A hard wall ring still rises beyond the interior for a guaranteed containment margin.
            const int interiorHalfExtent = 10; // 20x20 open interior
            const int wallThickness = 3;
            const int wallHeight = 8;
            const int floorThickness = 4;
            const int pitDepth = 6; // how far below the rim (floorTopY) the very center sits
            int outerHalfExtent = interiorHalfExtent + wallThickness;

            int minX = Math.Clamp(flatSceneAnchor.X - outerHalfExtent, 0, worldData.sizeInVoxels.X - 1);
            int maxX = Math.Clamp(flatSceneAnchor.X + outerHalfExtent, 0, worldData.sizeInVoxels.X - 1);
            int minZ = Math.Clamp(flatSceneAnchor.Z - outerHalfExtent, 0, worldData.sizeInVoxels.Z - 1);
            int maxZ = Math.Clamp(flatSceneAnchor.Z + outerHalfExtent, 0, worldData.sizeInVoxels.Z - 1);
            int floorTopY = Math.Clamp(flatSceneAnchor.Y, pitDepth + floorThickness, worldData.sizeInVoxels.Y - 1 - wallHeight);
            int baseY = floorTopY - pitDepth - floorThickness + 1; // fixed floor bottom, floorThickness below even the deepest point

            for (int x = minX; x <= maxX; x++)
                for (int z = minZ; z <= maxZ; z++)
                {
                    // Chebyshev distance (max of the two axis offsets), not Euclidean - a square cross-section at
                    // every depth level, i.e. an actual (stepped) pyramid rather than a smooth cone.
                    int distance = Math.Max(Math.Abs(x - flatSceneAnchor.X), Math.Abs(z - flatSceneAnchor.Z));

                    if (distance <= interiorHalfExtent)
                    {
                        float depthFraction = (float)distance / interiorHalfExtent; // 0 at center, 1 at the interior's edge
                        int columnTopY = floorTopY - (int)MathF.Round(pitDepth * (1f - depthFraction));
                        for (int y = baseY; y <= columnTopY; y++)
                            CarveTestFloorVoxel(new Point3(x, y, z));
                    }
                    else
                    {
                        // Beyond the sloped interior: a flat floor plus a wall ring rising above the rim, same
                        // shape as BuildFlatTestFloor, for a hard containment margin past the pyramid's own slope.
                        for (int y = floorTopY - floorThickness + 1; y <= floorTopY; y++)
                            CarveTestFloorVoxel(new Point3(x, y, z));
                        for (int y = floorTopY + 1; y <= floorTopY + wallHeight; y++)
                            CarveTestFloorVoxel(new Point3(x, y, z));
                    }
                }

            worldData.editingHandler.processChunks(false);
        }

        public void ScreenUpdate()
        {
            WorldRender.render.ScreenUpdate();
        }

        public void Update(float gameTime)
        {
            voxelTypeHandler.Update();
            pathHandler.Update(gameTime);
            WorldRender.render.Update(gameTime);
            worldData?.Update(gameTime);
        }

        public void Render(float gameTime)
        {
            if (worldData != null)
                WorldRender.render.Render(worldData, gameTime);
        }

        public void RenderUi()
        {
            if (GuiHandler.ShowUi)
            {
                pathHandler.RenderUi();
            }
        }
    }
}
