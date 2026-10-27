# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
#  SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from omni.isaac.kit import SimulationApp
import os
import argparse
from palletjack_assets import resolve_local_palletjack_usdz_paths
from hard_negative_assets import (
    combine_hard_negative_asset_pool,
    combine_regular_distractor_asset_pool,
    resolve_local_hard_negative_usdz_paths,
    resolve_local_scene_distractor_usdz_paths,
    should_add_palletjack_targets,
)

parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False, help="Launch script headless, default is False")
parser.add_argument("--height", type=int, default=544, help="Height of image")
parser.add_argument("--width", type=int, default=960, help="Width of image")
parser.add_argument("--num_frames", type=int, default=1000, help="Number of frames to record")
parser.add_argument("--distractors", type=str, default="warehouse", 
                    help="Options are 'warehouse' (default), 'additional' or None")
parser.add_argument(
    "--distractors-only",
    action="store_true",
    help="Generate target-free hard negatives using the complete distractor pool",
)
parser.add_argument("--data_dir", type=str, default=os.getcwd() + "/_palletjack_data", 
                    help="Location where data will be output")

args, unknown_args = parser.parse_known_args()

# Check local Experiment 2 assets before launching Isaac Sim, and resolve from this
# script so generation is independent of the shell's current working directory.
LOCAL_PALLETJACKS = [
    str(path) for path in resolve_local_palletjack_usdz_paths(__file__)
]
LOCAL_SCENE_DISTRACTORS = [
    str(path) for path in resolve_local_scene_distractor_usdz_paths(__file__).values()
]
LOCAL_STACKER_PATHS = [
    str(path) for path in resolve_local_hard_negative_usdz_paths(__file__).values()
]

# This is the config used to launch simulation. 
CONFIG = {"renderer": "RayTracedLighting", "headless": args.headless, 
          "width": args.width, "height": args.height, "num_frames": args.num_frames}

simulation_app = SimulationApp(launch_config=CONFIG)


## This is the path which has the background scene in which objects will be added.
ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"

import carb
import omni
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from pxr import Semantics
import omni.replicator.core as rep

from omni.isaac.core.utils.semantics import get_semantics

# Increase subframes if shadows/ghosting appears of moving objects
# See known issues: https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator.html#known-issues
rep.settings.carb_settings("/omni/replicator/RTSubframes", 32)


# This is the location of the palletjacks in the simready asset library
PALLETJACKS = ["http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Scale_A/PalletTruckScale_A01_PR_NVD_01.usd",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Heavy_Duty_A/HeavyDutyPalletTruck_A01_PR_NVD_01.usd",
            "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/DigitalTwin/Assets/Warehouse/Equipment/Pallet_Trucks/Low_Profile_A/LowProfilePalletTruck_A01_PR_NVD_01.usd"]

# The warehouse distractors which will be added to the scene and randomized
DISTRACTORS_WAREHOUSE = 2 * ["/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_A_03.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_C_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticB_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticA_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticD_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BottlePlasticE_01.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_BucketPlastic_B.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1262.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1268.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1482.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_1683.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_291.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1454.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01_1513.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_A_04.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_03.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_B_05.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_C_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlastic_E_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_PushcartA_02.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_04.usd",
                            "/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_03.usd"]


## Additional distractors which can be added to the scene
DISTRACTORS_ADDITIONAL = ["/Isaac/Environments/Hospital/Props/Pharmacy_Low.usd",
                        "/Isaac/Environments/Hospital/Props/SM_BedSideTable_01b.usd",
                        "/Isaac/Environments/Hospital/Props/SM_BooksSet_26.usd",
                        "/Isaac/Environments/Hospital/Props/SM_BottleB.usd",
                        "/Isaac/Environments/Hospital/Props/SM_BottleA.usd",
                        "/Isaac/Environments/Hospital/Props/SM_BottleC.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Cart_01a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Chair_02a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Chair_01a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Computer_02b.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Desk_04a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_DisposalStand_02.usd",
                        "/Isaac/Environments/Hospital/Props/SM_FirstAidKit_01a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_GasCart_01c.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Gurney_01b.usd",
                        "/Isaac/Environments/Hospital/Props/SM_HospitalBed_01b.usd",
                        "/Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Mirror.usd",
                        "/Isaac/Environments/Hospital/Props/SM_MopSet_01b.usd",
                        "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",
                        "/Isaac/Environments/Hospital/Props/SM_SupplyCabinet_01c.usd",
                        "/Isaac/Environments/Hospital/Props/SM_SupplyCart_01e.usd",
                        "/Isaac/Environments/Hospital/Props/SM_TrashCan.usd",
                        "/Isaac/Environments/Hospital/Props/SM_Washbasin.usd",
                        "/Isaac/Environments/Hospital/Props/SM_WheelChair_01a.usd",
                        "/Isaac/Environments/Office/Props/SM_WaterCooler.usd",
                        "/Isaac/Environments/Office/Props/SM_TV.usd",
                        "/Isaac/Environments/Office/Props/SM_TableC.usd",
                        "/Isaac/Environments/Office/Props/SM_Recliner.usd",
                        "/Isaac/Environments/Office/Props/SM_Personenleitsystem_Red1m.usd",
                        "/Isaac/Environments/Office/Props/SM_Lamp02_162.usd",
                        "/Isaac/Environments/Office/Props/SM_Lamp02.usd",
                        "/Isaac/Environments/Office/Props/SM_HandDryer.usd",
                        "/Isaac/Environments/Office/Props/SM_Extinguisher.usd"]


# The textures which will be randomized for the wall and floor
TEXTURES = ["/Isaac/Materials/Textures/Patterns/nv_asphalt_yellow_weathered.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_green_white.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_square_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_concrete_aged_with_lines.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stone_painted_grey.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_tile_hexagonal_various.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_carpet_abstract_pattern.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_siding_weathered_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_animalfur_pattern_greys.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_artificialgrass_green.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_reclaimed.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_brick_red_stacked.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_fireplace_wall.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_fabric_square_grid.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_granite_tile.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_marble.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_gravel_grey_leaves.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_plastic_blue.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stone_red_hatch.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stucco_red_painted.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_rubber_woven_charcoal.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_stucco_smooth_blue.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wood_shingles_brown.jpg",
            "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg"]


def update_semantics(stage, keep_semantics=[]):
    """ Remove semantics from the stage except for keep_semantic classes"""
    for prim in stage.Traverse():
        if prim.HasAPI(Semantics.SemanticsAPI):
            processed_instances = set()
            for property in prim.GetProperties():
                is_semantic = Semantics.SemanticsAPI.IsSemanticsAPIPath(property.GetPath())
                if is_semantic:
                    instance_name = property.SplitName()[1]
                    if instance_name in processed_instances:
                        # Skip repeated instance, instances are iterated twice due to their two semantic properties (class, data)
                        continue
                    
                    processed_instances.add(instance_name)
                    sem = Semantics.SemanticsAPI.Get(prim, instance_name)
                    type_attr = sem.GetSemanticTypeAttr()
                    data_attr = sem.GetSemanticDataAttr()


                    for semantic_class in keep_semantics:
                    # Check for our data classes needed for the model
                        if data_attr.Get() == semantic_class:
                            continue
                        else:
                            # remove semantics of all other prims
                            prim.RemoveProperty(type_attr.GetName())
                            prim.RemoveProperty(data_attr.GetName())
                            prim.RemoveAPI(Semantics.SemanticsAPI, instance_name)
    

# needed for loading textures correctly
def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception("Nucleus server not found, could not access Isaac Sim assets folder")
    return assets_root_path + relative_path


def full_distractors_list(distractor_type="warehouse"):
    """Resolve one existing distractor group or combine both groups for hard negatives."""
    if distractor_type == "warehouse":
        distractor_assets = DISTRACTORS_WAREHOUSE
    elif distractor_type == "additional":
        distractor_assets = DISTRACTORS_ADDITIONAL
    elif distractor_type == "all":
        distractor_assets = combine_hard_negative_asset_pool(
            DISTRACTORS_WAREHOUSE,
            DISTRACTORS_ADDITIONAL,
            [],
        )
    else:
        print("No Distractors being added to the current scene for SDG")
        return []

    resolved_isaac_assets = [prefix_with_isaac_asset_server(path) for path in distractor_assets]
    return combine_regular_distractor_asset_pool(
        resolved_isaac_assets,
        LOCAL_SCENE_DISTRACTORS,
        LOCAL_STACKER_PATHS,
    )


def full_textures_list():
    full_tex_list = []
    for texture in TEXTURES:
        full_tex_list.append(prefix_with_isaac_asset_server(texture))

    return full_tex_list


def add_palletjacks():
    original_palletjack_objects = [
        rep.create.from_usd(path, semantics=[("class", "palletjack")], count=2)
        for path in PALLETJACKS
    ]
    local_original_material_objects = [
        rep.create.from_usd(path, semantics=[("class", "palletjack")], count=2)
        for path in LOCAL_PALLETJACKS
    ]
    local_randomized_material_objects = [
        rep.create.from_usd(path, semantics=[("class", "palletjack")], count=1)
        for path in LOCAL_PALLETJACKS
    ]
    return (
        rep.create.group(original_palletjack_objects),
        rep.create.group(local_original_material_objects),
        rep.create.group(local_randomized_material_objects),
    )


def add_distractors(distractor_type="warehouse"):
    full_distractors = full_distractors_list(distractor_type)
    distractors = [rep.create.from_usd(distractor_path, count=1) for distractor_path in full_distractors]
    distractor_group = rep.create.group(distractors)
    with distractor_group:
        rep.modify.semantics(semantics=[], mode="clear")
    return distractor_group


# This will handle replicator
def run_orchestrator():

    rep.orchestrator.run()

    # Wait until started
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Wait until stopped
    while rep.orchestrator.get_is_started():
        simulation_app.update()

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()


def main():
    # Open the environment in a new stage
    print(f"Loading Stage {ENV_URL}")
    open_stage(prefix_with_isaac_asset_server(ENV_URL))
    stage = get_current_stage()

    # Run some app updates to make sure things are properly loaded
    for i in range(100):
        if i % 10 == 0:
            print(f"App uppdate {i}..")
        simulation_app.update()


    textures = full_textures_list()
    include_palletjack_targets = should_add_palletjack_targets(args.distractors_only)
    if include_palletjack_targets:
        (
            rep_palletjack_group,
            rep_local_palletjack_original_group,
            rep_local_palletjack_randomized_group,
        ) = add_palletjacks()
        rep_distractor_group = add_distractors(distractor_type=args.distractors)
    else:
        # Load the combined pool once. Hard-negative frames use the same
        # per-frame pose randomization and camera capture as positive scenes.
        rep_distractor_group = add_distractors(distractor_type="all")

    # We only need labels for the palletjack objects
    update_semantics(stage=stage, keep_semantics=["palletjack"])

    # Create camera with Replicator API for gathering data
    cam = rep.create.camera(clipping_range=(0.1, 1000000))

    # trigger replicator pipeline
    with rep.trigger.on_frame(num_frames=CONFIG["num_frames"]):

        # Move the camera around in the scene, focus on the center of warehouse
        with cam:
            rep.modify.pose(position=rep.distribution.uniform((-9.2, -11.8, 0.4), (7.2, 15.8, 4)),
                            look_at=(0, 0, 0))

        if not args.distractors_only:
            # Get the Palletjack body mesh and modify its color
            with rep.get.prims(path_pattern="SteerAxles"):
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

            # Alternate between the source USDZ materials and randomized color materials.
            # Separate instances let the original material bindings remain untouched.
            with rep_local_palletjack_original_group:
                rep.modify.visibility(rep.distribution.sequence([True, False]))

            with rep_local_palletjack_randomized_group:
                rep.modify.visibility(rep.distribution.sequence([False, True]))
                rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

            # Randomize the pose of all the added palletjacks
            with rep_palletjack_group:
                rep.modify.pose(position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                                scale=rep.distribution.uniform((0.01, 0.01, 0.01), (0.01, 0.01, 0.01)))

            # Keep the USDZ-authored sizes comparable to the target pallet jacks.
            with rep_local_palletjack_original_group:
                rep.modify.pose(position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                                scale=2.4)

            with rep_local_palletjack_randomized_group:
                rep.modify.pose(position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                                scale=2.4)

        # Modify the pose of the fixed distractor group for existing generation modes.
        if args.distractors_only or args.distractors != "None":
            with rep_distractor_group:
                rep.modify.pose(position=rep.distribution.uniform((-6, -6, 0), (6, 12, 0)),
                                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                                    scale=rep.distribution.uniform(1, 2.4))

        # Randomize the lighting of the scene
        with rep.get.prims(path_pattern="RectLight"):
            rep.modify.attribute("color", rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
            rep.modify.attribute("intensity", rep.distribution.normal(100000.0, 600000.0))
            rep.modify.visibility(rep.distribution.choice([True, False, False, False, False, False, False]))

        # select floor material
        random_mat_floor = rep.create.material_omnipbr(diffuse_texture=rep.distribution.choice(textures),
                                                    roughness=rep.distribution.uniform(0, 1),
                                                    metallic=rep.distribution.choice([0, 1]),
                                                    emissive_texture=rep.distribution.choice(textures),
                                                    emissive_intensity=rep.distribution.uniform(0, 1000),)
        
        
        with rep.get.prims(path_pattern="SM_Floor"):
            rep.randomizer.materials(random_mat_floor)

        # select random wall material
        random_mat_wall = rep.create.material_omnipbr(diffuse_texture=rep.distribution.choice(textures),
                                                    roughness=rep.distribution.uniform(0, 1),
                                                    metallic=rep.distribution.choice([0, 1]),
                                                    emissive_texture=rep.distribution.choice(textures),
                                                    emissive_intensity=rep.distribution.uniform(0, 1000),)

        
        with rep.get.prims(path_pattern="SM_Wall"):
            rep.randomizer.materials(random_mat_wall)


    # Set up the writer
    writer = rep.WriterRegistry.get("KittiWriter")

    # output directory of writer
    output_directory = args.data_dir
    print("Outputting data to ", output_directory)

    # use writer for bounding boxes, rgb and segmentation
    writer.initialize(output_dir=output_directory,
                    omit_semantic_type=True,)


    # attach camera render products to wrieter so that data is outputted
    RESOLUTION = (CONFIG["width"], CONFIG["height"])
    render_product  = rep.create.render_product(cam, RESOLUTION)
    writer.attach(render_product)

    # run rep pipeline
    run_orchestrator()
    simulation_app.update()



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Exception: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
