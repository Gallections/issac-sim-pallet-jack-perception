from __future__ import annotations

import sys
import tempfile
import unittest
import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDG_SCRIPT_DIR = PROJECT_ROOT / "simulation" / "scripts" / "palletjack_sdg"
sys.path.insert(0, str(SDG_SCRIPT_DIR))

from palletjack_assets import resolve_local_palletjack_usdz_paths


class ResolveLocalPalletjackUsdzPathsTests(unittest.TestCase):
    def test_replicator_uses_ten_rt_subframes_per_capture(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
        subframe_values = [
            node.args[1].value
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "carb_settings"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "/omni/replicator/RTSubframes"
            and isinstance(node.args[1], ast.Constant)
        ]

        self.assertEqual(subframe_values, [10])

    def test_resolves_both_experiment_two_assets_from_script_location(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            script_path = root / "simulation" / "scripts" / "palletjack_sdg" / "standalone_palletjack_sdg.py"
            script_path.parent.mkdir(parents=True)
            script_path.touch()

            asset_directory = root / "simulation" / "assets" / "pallet_jacks" / "experiment-02"
            asset_directory.mkdir(parents=True)
            filenames = ("electric_pallet_jack.usdz", "pallet_jack_advanced.usdz")
            for filename in filenames:
                (asset_directory / filename).touch()

            resolved_paths = resolve_local_palletjack_usdz_paths(script_path)

            self.assertEqual(
                resolved_paths,
                [asset_directory / filename for filename in filenames],
            )

    def test_reports_missing_local_asset_names(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            root = Path(temporary_directory)
            script_path = root / "simulation" / "scripts" / "palletjack_sdg" / "standalone_palletjack_sdg.py"
            script_path.parent.mkdir(parents=True)
            script_path.touch()

            with self.assertRaisesRegex(FileNotFoundError, "electric_pallet_jack.usdz"):
                resolve_local_palletjack_usdz_paths(script_path)

    def test_local_usdz_instances_use_requested_one_point_nine_scale(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
        local_pose_calls = []

        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.With):
                continue
            uses_local_group = any(
                isinstance(item.context_expr, ast.Name)
                and item.context_expr.id in {
                    "rep_local_palletjack_original_group",
                    "rep_local_palletjack_randomized_group",
                }
                for item in node.items
            )
            if not uses_local_group:
                continue
            local_pose_calls.extend(
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "pose"
                and isinstance(call.func.value, ast.Attribute)
                and call.func.value.attr == "modify"
            )

        self.assertEqual(len(local_pose_calls), 2)
        for pose_call in local_pose_calls:
            scale_argument = next(
                keyword.value
                for keyword in pose_call.keywords
                if keyword.arg == "scale"
            )
            self.assertIsInstance(scale_argument, ast.Constant)
            self.assertEqual(scale_argument.value, 1.9)

    def test_original_and_randomized_local_material_variants_alternate_by_frame(self) -> None:
        script_path = SDG_SCRIPT_DIR / "standalone_palletjack_sdg.py"
        syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
        sequence_by_group = {}

        for node in ast.walk(syntax_tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                if not isinstance(item.context_expr, ast.Name):
                    continue
                group_name = item.context_expr.id
                if group_name not in {
                    "rep_local_palletjack_original_group",
                    "rep_local_palletjack_randomized_group",
                }:
                    continue
                for call in ast.walk(node):
                    if not (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "visibility"
                        and isinstance(call.func.value, ast.Attribute)
                        and call.func.value.attr == "modify"
                    ):
                        continue
                    sequence = call.args[0]
                    self.assertIsInstance(sequence, ast.Call)
                    self.assertEqual(sequence.func.attr, "sequence")
                    sequence_values = tuple(
                        element.value for element in sequence.args[0].elts
                    )
                    sequence_by_group[group_name] = sequence_values

        self.assertEqual(
            sequence_by_group,
            {
                "rep_local_palletjack_original_group": (True, False),
                "rep_local_palletjack_randomized_group": (False, True),
            },
        )


if __name__ == "__main__":
    unittest.main()
