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

        public void BuildFlatTestFloor()
        {
            if (!testFloorTypeRegistered)
            {
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
                    {
                        Point3 worldVoxel = new Point3(x, y, z);
                        Point3 chunkPos = worldVoxel / 16;
                        Point3 voxelPosInChunk = worldVoxel % 16;

                        uint pointer = worldData.editingHandler.getChunkDataToEdit(chunkPos);
                        worldData.editingHandler.setVoxelData(pointer, voxelPosInChunk, (1u << 15) | testFloorTypeRenderIndex);
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
