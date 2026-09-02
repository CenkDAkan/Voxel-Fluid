
// BUILD FLAGS START
// NOTE: Make sure they are the same as in "Content/shaders/settings.fxh"

#define ENTITIES
//#define HDR // Note that HDR requires fullscreen

// BUILD FLAGS END


using ImGuiNET;
using NAADF.Gui;
using NAADF.World.Data;
using NAADF.World.Render;
using System;
using System.IO;
using System.Text.Json;

namespace NAADF
{
    public static class BuildFlags
    {
#if ENTITIES
        public const bool Entities = true;
#else
    public const bool Entities = false;
#endif
#if HDR
        public const bool Hdr = true;
#else
        public const bool Hdr = false;
#endif
    }

    public class SettingDataGeneral
    {
        public float exposure = 1.0f;
        public float toneMappingFac = 1.5f;
        public float fov = 90;
        public bool lockTo60fps = false;

        public void RenderImGui()
        {
            if (ImGui.Checkbox("Lock to 60 fps", ref lockTo60fps))
                App.app.IsFixedTimeStep = lockTo60fps;
            ImGui.SliderFloat("Exposure", ref exposure, 0.1f, 10, "%.3g", ImGuiSliderFlags.Logarithmic);
            ImGui.SliderFloat("Tone Mapping", ref toneMappingFac, 0.1f, 10.0f, "%.3g", ImGuiSliderFlags.Logarithmic);
            ImGui.SliderFloat("FOV", ref fov, 1, 120, "%.7g", ImGuiSliderFlags.None);
        }
    }

    public class SettingDataRender
    {
        public bool showSteps = false;
        public RenderVersion version = RenderVersion.Base;
        public SettingDataRenderAlbedo renderAlbedo = new();
        public SettingDataRenderBase renderBase = new();
        public SettingDataRenderPathTracer renderPathTracer = new();

        public void RenderImGui()
        {
            ImGui.Checkbox("Show ray steps", ref showSteps);
            ImGuiCommon.HelperIcon("Shows the amount of steps done during primary ray traversal. Brighter means more steps", 500);
            if (ImGui.BeginCombo("Render version", version.ToString()))
            {
                foreach (RenderVersion curVersion in Enum.GetValues(typeof(RenderVersion)))
                {
                    bool isSelected = version == curVersion;
                    if (ImGui.Selectable(curVersion.ToString(), isSelected))
                    {
                        version = curVersion;
                        WorldRender.ApplyRenderVersion(version);
                    }
                    if (isSelected)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }

            if (version == RenderVersion.Albedo)
                renderAlbedo.RenderImGui();
            else if (version == RenderVersion.Base)
                renderBase.RenderImGui();
            else
                renderPathTracer.RenderImGui();
        }
    }

    // Selects which fluid comparison approach is live, letting benchmarking/testing switch between them on the fly instead of needing a rebuild per approach
    public class SettingDataFluid
    {
        public FluidSimulationMode mode = FluidSimulationMode.None;

        public void RenderImGui()
        {
            if (ImGui.Button("Load empty test scene"))
            {
                App.worldHandler.LoadEmptyTestScene(); // resets the world's own activeFluidMode to None internally
                mode = FluidSimulationMode.None;        // keep this combo's displayed selection in sync with that reset
            }
            ImGuiCommon.HelperIcon("Regenerates the world with every voxel cleared - a genuinely blank canvas, no floor, no leftover geometry - so a controlled scene (e.g. a waterfall) can be built from scratch with the normal editing tools. Also resets the fluid mode below to None.", 500);

            ImGui.SameLine();
            if (ImGui.Button("Build flat floor"))
                App.worldHandler.BuildFlatTestFloor();
            ImGuiCommon.HelperIcon("Optional: carves a flat 128x128 reference floor at the empty scene's anchor point, for testing that wants a flat surface instead of custom-built terrain. Best used right after \"Load empty test scene\" - carves into whatever terrain is currently there otherwise.", 500);

            ImGui.SameLine();
            if (ImGui.Button("Load oasis scene"))
            {
                App.worldHandler.LoadOasisScene();
                mode = FluidSimulationMode.None;
            }
            ImGuiCommon.HelperIcon("Reloads the original oasis.cvox terrain and resets the fluid mode below to None.", 500);

            if (ImGui.BeginCombo("Fluid simulation", mode.ToString()))
            {
                foreach (FluidSimulationMode curMode in Enum.GetValues(typeof(FluidSimulationMode)))
                {
                    bool isSelected = mode == curMode;
                    if (ImGui.Selectable(curMode.ToString(), isSelected))
                    {
                        mode = curMode;
                        App.worldHandler.worldData.ApplyFluidSimulationMode(mode);
                    }
                    if (isSelected)
                        ImGui.SetItemDefaultFocus();
                }
                ImGui.EndCombo();
            }
            ImGuiCommon.HelperIcon("Switches the fluid comparison approach in use. Clears whatever the previous approach drew and seeds the new one with a fixed starting scenario.", 500);

            if (mode == FluidSimulationMode.DenseEulerian)
            {
                ImGui.Checkbox("Enable dense gravity", ref App.worldHandler.worldData.denseFluidHandler.enableGravity);
                ImGuiCommon.HelperIcon("Off by default: a freshly placed domain starts suspended so diffusion's spreading effect can be checked in isolation, without racing the still-unimplemented floor-resting behavior at low framerates. Check this to let it fall.", 500);

                ImGui.SliderFloat("Gravity strength", ref App.worldHandler.worldData.denseFluidHandler.gravityStrength, 0f, 100f);
                ImGuiCommon.HelperIcon("Downward acceleration, same arbitrary unit scale as everything else here. Default (20) matches FluidHandler's sparse-particle gravity.", 500);

                ImGui.Checkbox("Enable cohesion", ref App.worldHandler.worldData.denseFluidHandler.enableCohesion);
                ImGuiCommon.HelperIcon("Off by default. Curvature-driven attraction meant to counteract diffusion/advection's spreading - the thesis has no cohesion term at all, this is this project's own addition.", 500);

                ImGui.SliderFloat("Cohesion strength", ref App.worldHandler.worldData.denseFluidHandler.cohesionCoefficient, 0f, 200f);
                ImGuiCommon.HelperIcon("Sigma in the CSF force formula, same arbitrary unit scale as gravity. Starting value (20) is deliberately far below the derived stability ceiling (~3900) - tune up from here.", 500);

                ImGui.SliderFloat("Cohesion density strength", ref App.worldHandler.worldData.denseFluidHandler.cohesionDensityCoefficient, 0f, 20f);
                ImGuiCommon.HelperIcon("Applied directly to density inside every one of diffusion's 30 Jacobi iterations, giving cohesion the same repeated access diffusion has, instead of one indirect nudge through velocity per tick. Much smaller scale than cohesion strength above - start low, unbounded growth over 30 iterations is the main risk.", 500);

                ImGui.SliderInt("Curvature smooth iterations", ref App.worldHandler.worldData.denseFluidHandler.curvatureSmoothIterations, 0, 20);
                ImGuiCommon.HelperIcon("How many denoising passes the density copy gets before curvature is computed from it. More passes = smoother curvature = less tick-to-tick flicker in which cells cross the visibility threshold, at the cost of blurring the actual interface it's measuring if pushed too high.", 500);

                ImGui.SliderFloat("Curvature smooth blend", ref App.worldHandler.worldData.denseFluidHandler.curvatureSmoothBlend, 0f, 1f);
                ImGuiCommon.HelperIcon("How much of each pass's neighbor average gets blended in (0 = no smoothing, 1 = fully replace with neighbor average each pass). Works together with the iteration count above.", 500);

                ImGui.SliderInt("Domain size", ref App.worldHandler.worldData.denseFluidHandler.domainSize, 4, 128);
                ImGuiCommon.HelperIcon("Cubic domain side length. Doesn't resize the currently placed domain - click \"Replace domain\" below (or reselect Dense Eulerian above) to tear down and recreate it at the new size. Large values can make the engine unresponsive.", 500);

                if (ImGui.Button("Replace domain"))
                    App.worldHandler.worldData.ApplyFluidSimulationMode(FluidSimulationMode.DenseEulerian);
                ImGuiCommon.HelperIcon("Clears the current domain and places a fresh one at the size set above, same as switching the mode above away and back.", 500);
            }
        }
    }

    public class SettingData
    {
        public SettingDataGeneral general = new();
        public SettingDataRender render = new();
        public SettingDataFluid fluid = new();

        public void RenderImGui()
        {
            if (ImGui.Button("Save"))
                Settings.Save();

            ImGui.Spacing();
            ImGui.Separator();
            ImGui.Spacing();

            if (ImGui.BeginTabBar("SettingsTabs"))
            {
                if (ImGui.BeginTabItem("Render"))
                {
                    render.RenderImGui();
                    ImGui.EndTabItem();
                }
                if (ImGui.BeginTabItem("General"))
                {
                    general.RenderImGui();
                    ImGui.EndTabItem();
                }
                if (ImGui.BeginTabItem("Fluid"))
                {
                    fluid.RenderImGui();
                    ImGui.EndTabItem();
                }

                ImGui.EndTabBar();
            }
        }

        public string getJson()
        {
            return JsonSerializer.Serialize(this, new JsonSerializerOptions { IncludeFields = true });
        }

        public static SettingData fromJson(string json)
        {
            return JsonSerializer.Deserialize<SettingData>(json, new JsonSerializerOptions { IncludeFields = true });
        }
    }

    public static class Settings
    {
        public static SettingData data = new SettingData();
        public static bool isOpen = true;

        public static void Load()
        {
            if (File.Exists("settings.json"))
                data = SettingData.fromJson(File.ReadAllText("settings.json"));
        }

        public static void Save()
        {
            File.WriteAllText("settings.json", data.getJson());
        }

        public static void RenderImGui()
        {
            if (!isOpen)
                return;

            ImGui.SetNextWindowSize(new System.Numerics.Vector2(400, 500), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(5, 260 + 18), ImGuiCond.FirstUseEver);
            if (ImGui.Begin("Settings", ref isOpen))
            {
                data.RenderImGui();
            }
            ImGui.End();
        }
    }
}
