"""Low cost UE backdrop applied whenever a common scene is loaded.

Materials are generated once and reused. The original map is not saved here;
the loader reapplies this preset for editor sessions, PIE and batch rendering.
"""
from __future__ import annotations

import unreal


ASSET_ROOT = "/Game/Bedlam/Generated/RealUSStudio"
FOLDER = "GeneratedScene/environment"
_ORIGINAL_RENDER_CVARS: dict[str, int] = {}


def _expression(material, expression_class, **properties):
    node = unreal.MaterialEditingLibrary.create_material_expression(material, expression_class)
    for name, value in properties.items():
        node.set_editor_property(name, value)
    return node


def _connect(source, target, input_name, output_name=""):
    # Unary expression inputs are unnamed in UE 5.3's material editor API.
    target_input = "" if input_name == "Input" else input_name
    if not unreal.MaterialEditingLibrary.connect_material_expressions(source, output_name, target, target_input):
        raise RuntimeError(f"Cannot connect studio material input {input_name}")


def _material(name, build):
    path = f"{ASSET_ROOT}/{name}"
    existing = unreal.load_asset(path) if unreal.EditorAssetLibrary.does_asset_exist(path) else None
    if existing is not None:
        return existing
    material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, ASSET_ROOT, unreal.Material, unreal.MaterialFactoryNew(),
    )
    if material is None:
        raise RuntimeError(f"Cannot create studio material {path}")
    build(material)
    unreal.MaterialEditingLibrary.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def _build_ground(material):
    color = _expression(
        material, unreal.MaterialExpressionVectorParameter,
        parameter_name="BaseColor", default_value=unreal.LinearColor(0.055, 0.060, 0.070, 1.0),
    )
    roughness = _expression(material, unreal.MaterialExpressionConstant, r=0.94)
    specular = _expression(material, unreal.MaterialExpressionConstant, r=0.18)
    library = unreal.MaterialEditingLibrary
    library.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    library.connect_material_property(roughness, "", unreal.MaterialProperty.MP_ROUGHNESS)
    library.connect_material_property(specular, "", unreal.MaterialProperty.MP_SPECULAR)


def _build_sky(material):
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    material.set_editor_property("two_sided", True)
    material.set_editor_property("is_sky", True)
    # Camera-relative direction gives a continuous three-color vertical gradient.
    direction = _expression(material, unreal.MaterialExpressionCameraVectorWS)
    height = _expression(material, unreal.MaterialExpressionComponentMask, r=False, g=False, b=True, a=False)
    _connect(direction, height, "Input")
    negative = _expression(material, unreal.MaterialExpressionMultiply, const_b=-1.0)
    _connect(height, negative, "A")
    upper = _expression(material, unreal.MaterialExpressionClamp, min_default=0.0, max_default=1.0)
    lower = _expression(material, unreal.MaterialExpressionClamp, min_default=0.0, max_default=1.0)
    _connect(negative, upper, "Input")
    _connect(height, lower, "Input")
    horizon = _expression(material, unreal.MaterialExpressionConstant3Vector, constant=unreal.LinearColor(0.42, 0.44, 0.47, 1.0))
    zenith = _expression(material, unreal.MaterialExpressionConstant3Vector, constant=unreal.LinearColor(0.24, 0.26, 0.29, 1.0))
    bottom = _expression(material, unreal.MaterialExpressionConstant3Vector, constant=unreal.LinearColor(0.12, 0.13, 0.15, 1.0))
    top_gradient = _expression(material, unreal.MaterialExpressionLinearInterpolate)
    gradient = _expression(material, unreal.MaterialExpressionLinearInterpolate)
    _connect(horizon, top_gradient, "A")
    _connect(zenith, top_gradient, "B")
    _connect(upper, top_gradient, "Alpha")
    _connect(top_gradient, gradient, "A")
    _connect(bottom, gradient, "B")
    _connect(lower, gradient, "Alpha")
    unreal.MaterialEditingLibrary.connect_material_property(gradient, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)


def _spawn(actor_class, label, location=(0.0, 0.0, 0.0)):
    actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        actor_class, unreal.Vector(*location),
    )
    actor.set_actor_label(label)
    actor.set_folder_path(FOLDER)
    return actor


def _mesh(label, mesh_path, material, scale, z=0.0):
    actor = _spawn(unreal.StaticMeshActor, label, (0.0, 0.0, z))
    component = actor.static_mesh_component
    component.set_static_mesh(unreal.load_asset(mesh_path))
    component.set_material(0, material)
    component.set_editor_property("cast_shadow", False)
    component.set_editor_property("affect_distance_field_lighting", False)
    component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    actor.set_actor_scale3d(unreal.Vector(*scale))
    return actor


def _postprocess():
    actor = _spawn(unreal.PostProcessVolume, "GEN_studio_postprocess")
    actor.set_editor_property("unbound", True)
    actor.set_editor_property("priority", 1000.0)
    settings = unreal.PostProcessSettings()
    values = {
        "dynamic_global_illumination_method": unreal.DynamicGlobalIlluminationMethod.NONE,
        "reflection_method": unreal.ReflectionMethod.NONE,
        "auto_exposure_min_brightness": 1.0,
        "auto_exposure_max_brightness": 1.0,
        "auto_exposure_bias": 0.0,
        "auto_exposure_apply_physical_camera_exposure": False,
        "motion_blur_amount": 0.0,
        "bloom_intensity": 0.0,
        "vignette_intensity": 0.0,
        "ambient_occlusion_intensity": 0.45,
        "ambient_occlusion_radius": 25.0,
    }
    for name, value in values.items():
        settings.set_editor_property(f"override_{name}", True)
        settings.set_editor_property(name, value)
    actor.set_editor_property("settings", settings)


def apply_scene_environment(environment) -> dict:
    preset = environment.ue_environment_preset
    if preset == "inherit":
        for name, value in _ORIGINAL_RENDER_CVARS.items():
            unreal.SystemLibrary.execute_console_command(None, f"{name} {value}")
        _ORIGINAL_RENDER_CVARS.clear()
        return {"preset": preset}
    if preset != "soft_studio":
        raise ValueError(f"Unsupported UE environment preset: {preset!r}")

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    disabled = []
    for actor in actor_subsystem.get_all_level_actors():
        label = str(actor.get_actor_label())
        class_name = str(actor.get_class().get_name())
        if label.startswith("GEN_"):
            continue
        if isinstance(actor, unreal.PostProcessVolume):
            actor.set_editor_property("enabled", False)
            disabled.append(label)
        elif any(word in class_name.lower() for word in ("hdribackdrop", "skyatmosphere", "skysphere", "sunsky", "volumetriccloud", "heightfog")) or isinstance(actor, unreal.Light):
            # Retain BEDLAM helper actors/bindings, but disable their render components.
            actor.set_actor_hidden_in_game(True)
            for component in actor.get_components_by_class(unreal.SceneComponent):
                component.set_visibility(False, True)
            for component in actor.get_components_by_class(unreal.LightComponentBase):
                component.set_editor_property("intensity", 0.0)
            disabled.append(label)

    ground = _mesh(
        "GEN_studio_ground", "/Engine/BasicShapes/Plane.Plane",
        _material("M_StudioGround_v1", _build_ground), (2000.0, 2000.0, 1.0),
        z=float(environment.ground_plane_z_m) * 100.0 - 0.2,
    )
    ground.static_mesh_component.set_vector_parameter_value_on_materials(
        "BaseColor", unreal.Vector(*environment.ue_ground_plane_color[:3]),
    )
    _mesh(
        "GEN_studio_sky", "/Engine/BasicShapes/Sphere.Sphere",
        _material("M_StudioSky_v1", _build_sky), (4000.0, 4000.0, 4000.0),
    )
    sky = _spawn(unreal.SkyLight, "GEN_studio_skylight", (0.0, 0.0, 100.0)).light_component
    sky.set_mobility(unreal.ComponentMobility.MOVABLE)
    sky.set_editor_property("real_time_capture", False)
    sky.set_editor_property("cubemap_resolution", 128)
    sky.set_editor_property("sky_distance_threshold", 10000.0)
    sky.set_editor_property("lower_hemisphere_is_black", False)
    sky.set_editor_property("cast_shadows", False)
    sky.set_intensity(1.6)
    sky.recapture_sky()

    key_actor = _spawn(unreal.DirectionalLight, "GEN_studio_key")
    key_actor.set_actor_rotation(unreal.Rotator(pitch=-48.0, yaw=-35.0, roll=0.0), False)
    key = key_actor.light_component
    key.set_mobility(unreal.ComponentMobility.MOVABLE)
    key.set_intensity(2.0)
    key.set_light_color(unreal.LinearColor(1.0, 0.97, 0.93, 1.0))
    key.set_light_source_angle(8.0)
    key.set_editor_property("cast_raytraced_shadow", unreal.CastRayTracedShadow.DISABLED)
    key.set_editor_property("dynamic_shadow_distance_movable_light", 1500.0)
    key.set_editor_property("dynamic_shadow_cascades", 2)
    # A distant, non-volumetric haze merges the plane with the neutral horizon.
    # It starts beyond the working area so the cameras retain local contrast.
    fog = _spawn(unreal.ExponentialHeightFog, "GEN_studio_horizon_fog").component
    fog.set_fog_density(0.035)
    fog.set_fog_height_falloff(0.05)
    fog.set_start_distance(1000.0)
    fog.set_fog_inscattering_color(unreal.LinearColor(0.42, 0.44, 0.47, 1.0))
    fog.set_editor_property("enable_volumetric_fog", False)
    _postprocess()
    # Classic shadow maps avoid virtual-shadow page allocation for the very large backdrop.
    for name in ("r.Shadow.Virtual.Enable", "r.RayTracing.ForceAllRayTracingEffects"):
        if name not in _ORIGINAL_RENDER_CVARS:
            _ORIGINAL_RENDER_CVARS[name] = unreal.SystemLibrary.get_console_variable_int_value(name)
        unreal.SystemLibrary.execute_console_command(None, f"{name} 0")
    result = {
        "preset": preset, "disabled_map_environment_actors": disabled,
        "ground_color_linear": list(environment.ue_ground_plane_color),
        "ground_width_m": 2000.0, "sky_radius_m": 2000.0,
        "skylight_realtime_capture": False, "skylight_cubemap_resolution": 128,
        "volumetric_fog": False,
        "lumen": False, "raytraced_effects": False,
    }
    unreal.log(f"UE_SCENE_ENVIRONMENT: {result}")
    return result
