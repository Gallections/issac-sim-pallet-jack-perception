from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDG_SCRIPT_DIR = PROJECT_ROOT / "simulation" / "scripts" / "palletjack_sdg"
sys.path.insert(0, str(SDG_SCRIPT_DIR))

from hard_negative_assets import (  # noqa: E402
    HARD_NEGATIVE_LOCAL_ASSET_FILENAMES,
    LOCAL_SCENE_DISTRACTOR_USDZ_FILENAMES,
    combine_regular_distractor_asset_pool,
    combine_hard_negative_asset_pool,
    resolve_local_hard_negative_usdz_paths,
    resolve_local_scene_distractor_usdz_paths,
    should_add_palletjack_targets,
)


class HardNegativeAssetTests(unittest.TestCase):
    def test_electric_truck_is_positive_and_stacker_is_the_only_local_hard_negative(self) -> None:
        self.assertEqual(
            HARD_NEGATIVE_LOCAL_ASSET_FILENAMES,
            {"pallet_stacker": "pallet_jack_advanced.usdz"},
        )

    def test_local_scene_distractor_assets_resolve_from_simulation_assets(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            script_path = root / "simulation" / "scripts" / "palletjack_sdg" / "generator.py"
            script_path.parent.mkdir(parents=True)
            script_path.touch()
            asset_directory = root / "simulation" / "assets"
            asset_directory.mkdir(parents=True)
            for filename in LOCAL_SCENE_DISTRACTOR_USDZ_FILENAMES.values():
                (asset_directory / filename).touch()

            resolved = resolve_local_scene_distractor_usdz_paths(script_path)

            self.assertEqual(
                resolved,
                {
                    name: asset_directory / filename
                    for name, filename in LOCAL_SCENE_DISTRACTOR_USDZ_FILENAMES.items()
                },
            )

    def test_regular_warehouse_and_additional_groups_both_include_local_distractors_and_stacker(self) -> None:
        local_assets = ["forklift", "ladder", "cart", "rack"]
        stacker = "stacker"

        warehouse_pool = combine_regular_distractor_asset_pool(
            ["warehouse-prop"], local_assets, [stacker]
        )
        additional_pool = combine_regular_distractor_asset_pool(
            ["additional-prop"], local_assets, [stacker]
        )

        self.assertEqual(warehouse_pool, ["warehouse-prop", *local_assets, stacker])
        self.assertEqual(additional_pool, ["additional-prop", *local_assets, stacker])

    def test_local_equipment_paths_are_resolved_relative_to_generator(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            script_path = root / "simulation" / "scripts" / "palletjack_sdg" / "generator.py"
            script_path.parent.mkdir(parents=True)
            script_path.touch()
            asset_directory = root / "simulation" / "assets"
            asset_directory.mkdir(parents=True)
            for filename in HARD_NEGATIVE_LOCAL_ASSET_FILENAMES.values():
                (asset_directory / filename).touch()

            resolved = resolve_local_hard_negative_usdz_paths(script_path)

            self.assertEqual(
                resolved,
                {
                    "pallet_stacker": asset_directory / "pallet_jack_advanced.usdz",
                },
            )

    def test_missing_local_hard_negative_asset_names_are_reported(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            script_path = root / "simulation" / "scripts" / "palletjack_sdg" / "generator.py"
            script_path.parent.mkdir(parents=True)
            script_path.touch()
            asset_directory = root / "simulation" / "assets"
            asset_directory.mkdir(parents=True)
            (asset_directory / "electric_pallet_jack.usdz").touch()

            with self.assertRaisesRegex(FileNotFoundError, "pallet_jack_advanced.usdz"):
                resolve_local_hard_negative_usdz_paths(script_path)

    def test_asset_pool_combines_both_existing_groups_and_only_stacker_as_hard_negative(self) -> None:
        pool = combine_hard_negative_asset_pool(
            ["warehouse-box", "warehouse-cart"],
            ["additional-cart"],
            ["pallet-stacker"],
        )

        self.assertEqual(
            pool,
            [
                "warehouse-box",
                "warehouse-cart",
                "additional-cart",
                "pallet-stacker",
            ],
        )

    def test_distractors_only_mode_omits_labeled_palletjack_targets(self) -> None:
        self.assertFalse(should_add_palletjack_targets(distractors_only=True))
        self.assertTrue(should_add_palletjack_targets(distractors_only=False))

    def test_generator_has_a_target_free_distractors_mode(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        flag_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--distractors-only"
        ]

        self.assertEqual(len(flag_calls), 1)
        self.assertTrue(
            any(
                keyword.arg == "action"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "store_true"
                for keyword in flag_calls[0].keywords
            )
        )

    def test_target_free_scene_randomizers_clear_semantics_from_sampled_assets(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        function_names = {"sample_hard_negative_props", "sample_forklift_stacker"}
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name in function_names
        }

        self.assertEqual(set(functions), function_names)
        for function in functions.values():
            instantiate_calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "instantiate"
            ]
            self.assertEqual(len(instantiate_calls), 1)
            clear_semantics_calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "semantics"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "modify"
            ]
            self.assertEqual(len(clear_semantics_calls), 1)
            self.assertEqual(
                next(
                    ast.literal_eval(keyword.value)
                    for keyword in clear_semantics_calls[0].keywords
                    if keyword.arg == "semantics"
                ),
                [],
            )
            self.assertEqual(
                next(
                    ast.literal_eval(keyword.value)
                    for keyword in clear_semantics_calls[0].keywords
                    if keyword.arg == "mode"
                ),
                "clear",
            )

    def test_regular_distractor_assets_have_semantics_cleared(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        add_distractors = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "add_distractors"
        )
        clear_calls = [
            node
            for node in ast.walk(add_distractors)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "semantics"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "modify"
        ]

        self.assertEqual(len(clear_calls), 1)
        self.assertEqual(
            next(
                ast.literal_eval(keyword.value)
                for keyword in clear_calls[0].keywords
                if keyword.arg == "semantics"
            ),
            [],
        )
        self.assertEqual(
            next(
                ast.literal_eval(keyword.value)
                for keyword in clear_calls[0].keywords
                if keyword.arg == "mode"
            ),
            "clear",
        )

    def test_main_skips_labeled_targets_in_distractors_only_mode(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        target_mode_assignment = next(
            node
            for node in main_function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "include_palletjack_targets"
                for target in node.targets
            )
        )
        self.assertIsInstance(target_mode_assignment.value, ast.Call)
        self.assertEqual(target_mode_assignment.value.func.id, "should_add_palletjack_targets")

        target_gate = next(
            node
            for node in main_function.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "include_palletjack_targets"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "add_palletjacks"
                for statement in target_gate.body
                for node in ast.walk(statement)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "register_hard_negative_randomizers"
                for statement in target_gate.orelse
                for node in ast.walk(statement)
            )
        )

    def test_single_experiment_three_launcher_generates_positive_and_hard_negative_groups(self) -> None:
        script_directory = PROJECT_ROOT / "simulation" / "scripts"
        for filename in ("generate_data.bat", "generate_data.sh"):
            contents = (script_directory / filename).read_text(encoding="utf-8")
            self.assertIn("experiment-03", contents)
            self.assertIn("hard_negatives", contents)
            self.assertIn("--distractors-only", contents)
            self.assertIn("distractors_warehouse", contents)
            self.assertIn("distractors_additional", contents)
            self.assertIn("no_distractors", contents)
            self.assertGreaterEqual(contents.count("--data_dir"), 4)

        for filename in ("generate_hard_negatives.bat", "generate_hard_negatives.sh"):
            self.assertFalse((script_directory / filename).exists())

    def test_regular_generation_launchers_write_experiment_three_without_touching_experiment_two(self) -> None:
        script_directory = PROJECT_ROOT / "simulation" / "scripts"
        for filename in ("generate_data.bat", "generate_data.sh"):
            launcher = script_directory / filename
            contents = launcher.read_text(encoding="utf-8")
            self.assertIn("experiment-03", contents)
            self.assertNotIn("experiment-02", contents)
            self.assertIn("mkdir", contents.lower())


if __name__ == "__main__":
    unittest.main()
