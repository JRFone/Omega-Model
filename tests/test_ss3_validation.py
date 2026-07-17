from __future__ import annotations

import math
import tempfile
import unittest
import zipfile
from unittest import mock
from pathlib import Path

import numpy as np

from stock_model.advanced_structures import SeasonalSpatialSettings, SexSpec, life_history
from stock_model.ss3_validation import (
    capability_matrix,
    parse_ss3_control,
    parse_ss3_data,
    parse_ss3_starter,
    selectivity_from_inflection_width,
    ss3_l1_l2_to_linf_t0,
    validate_model_directory,
    von_bertalanffy_length,
    weight_at_length,
    write_validation_report,
    download_latest_ss3_executable,
)
from noaa_validation_app import comparison_rows


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "validation_data" / "noaa_ss3" / "Simple"


class NOAAValidationTests(unittest.TestCase):
    def test_parse_official_simple_fixture(self) -> None:
        starter = parse_ss3_starter((FIXTURE / "starter.ss").read_text())
        data = parse_ss3_data((FIXTURE / "data.ss").read_text())
        control = parse_ss3_control((FIXTURE / "control.ss").read_text())
        self.assertEqual(starter.data_file, "data.ss")
        self.assertEqual(starter.control_file, "control.ss")
        self.assertEqual(data.start_year, 1971)
        self.assertEqual(data.end_year, 2001)
        self.assertEqual(len(data.catches), 31)
        self.assertEqual(len(data.indices), 21)
        self.assertAlmostEqual(control.parameters["NatM_uniform_Fem_GP_1"], 0.1)
        self.assertEqual(len(control.recruitment_deviations), 31)
        self.assertEqual(len(control.fishing_mortality_by_fleet["FISHERY"]), 31)

    def test_stock_recruit_parser_ignores_descriptive_global_assignment_comment(self) -> None:
        text = """
1 # not yet implemented; Future usage: Spawner-Recruitment: 1=global; 2=by area
3 #_Spawner-Recruitment; Options: 1=NA; 2=Ricker; 3=std_B-H
"""
        control = parse_ss3_control(text)
        self.assertEqual(control.stock_recruit_option, 3)

    def test_growth_conversion_reproduces_reference_lengths(self) -> None:
        linf, t0 = ss3_l1_l2_to_linf_t0(21.6591, 71.654, 0.0, 25.0, 0.14724)
        lengths = von_bertalanffy_length([0.0, 25.0], linf, 0.14724, t0)
        self.assertAlmostEqual(float(lengths[0]), 21.6591, places=8)
        self.assertAlmostEqual(float(lengths[1]), 71.654, places=8)

    def test_ss3_width_selectivity_hits_five_and_ninety_five_percent(self) -> None:
        values = selectivity_from_inflection_width([40.0, 50.0, 60.0], 50.0, 20.0)
        self.assertAlmostEqual(float(values[0]), 0.05, places=12)
        self.assertAlmostEqual(float(values[1]), 0.50, places=12)
        self.assertAlmostEqual(float(values[2]), 0.95, places=12)

    def test_weight_length_reference(self) -> None:
        self.assertAlmostEqual(float(weight_at_length(50.0, 2.44e-06, 3.34694)), 1.1850604182217688, places=12)

    def test_length_based_maturity_in_spatial_engine(self) -> None:
        settings = SeasonalSpatialSettings(
            years=2,
            max_age=12,
            sexes=(
                SexSpec(
                    "female",
                    maturity_model="length_logistic",
                    maturity_length50_mm=500.0,
                    maturity_slope_coefficient=-0.05,
                ),
            ),
            areas=(),
        )
        # life_history does not require areas, so this isolates the biological curve.
        life = life_history(settings)
        index = int(np.argmin(np.abs(life["length_mm"][0] - 500.0)))
        self.assertLess(abs(float(life["maturity"][0, index]) - 0.5), 0.2)

    def test_offline_noaa_validation_passes(self) -> None:
        result = validate_model_directory(FIXTURE, model_name="Simple")
        self.assertEqual(result.summary["validation_status"], "PASS")
        self.assertEqual(result.source_mode, "embedded_fixture")
        self.assertEqual(result.summary["checks_failed"], 0)
        self.assertGreaterEqual(result.summary["checks_passed"], 30)

    def test_noaa_comparison_exposes_both_answers_and_tolerance(self) -> None:
        from dataclasses import asdict

        result = validate_model_directory(FIXTURE, model_name="Simple")
        rows = comparison_rows(asdict(result))
        self.assertEqual(len(rows), result.summary["checks_total"])
        start_year = next(row for row in rows if row["comparison"] == "Start year")
        self.assertEqual(start_year["NOAA_reference"], 1971)
        self.assertEqual(start_year["Omega_result"], 1971)
        self.assertEqual(start_year["verdict"], "PASS")
        numeric = next(row for row in rows if row["comparison"] == "Converted Linf")
        self.assertIn("difference", numeric)
        self.assertIn("allowed_difference", numeric)

    def test_validation_report_export(self) -> None:
        result = validate_model_directory(FIXTURE, model_name="Simple")
        with tempfile.TemporaryDirectory() as temporary:
            outputs = write_validation_report(result, temporary)
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["html"]).exists())

    def test_official_executable_downloader_selects_windows_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("bin/ss3_win.exe", b"fake-binary")
            payload = archive_path.read_bytes()
            release = {
                "tag_name": "v-test",
                "name": "Test release",
                "assets": [
                    {"name": "source-code.zip", "browser_download_url": "https://example/source.zip", "size": 100},
                    {"name": "ss3-windows-x64.zip", "browser_download_url": "https://example/ss3.zip", "size": len(payload)},
                ],
            }
            with mock.patch("stock_model.ss3_validation._request_json", return_value=release), mock.patch(
                "stock_model.ss3_validation._download_bytes", return_value=payload
            ):
                manifest = download_latest_ss3_executable(Path(temporary) / "downloads")
            self.assertTrue(Path(manifest["executable"]).exists())
            self.assertEqual(manifest["release_tag"], "v-test")
            self.assertIn("ss3-windows", manifest["asset_name"])

    def test_capability_matrix_exposes_unimplemented_stress_features(self) -> None:
        rows = capability_matrix("Sablefish2015")
        statuses = {row["feature"]: row["omega_status"] for row in rows}
        self.assertEqual(statuses["cubic_spline_selectivity"], "not_implemented")
        self.assertEqual(statuses["double_normal_selectivity"], "not_implemented")


if __name__ == "__main__":
    unittest.main()


def test_competitive_scorecard_does_not_overclaim_numerical_equivalence():
    from stock_model.ss3_validation import competitive_scorecard

    rows = competitive_scorecard()
    by_dimension = {row["dimension"]: row for row in rows}
    assert by_dimension["Native numerical equivalence"]["current_position"] == "pending"
    assert by_dimension["Independent scientific acceptance"]["current_position"] == "ss3_leads"
    assert any(row["current_position"] == "omega_advantage_verified" for row in rows)
