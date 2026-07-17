from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stock_model.interactive_charts import (
    ChartProfile,
    ChartProfileStore,
    InteractiveChartFactory,
    SeriesSpec,
    lttb_downsample,
)


class InteractiveChartsTests(unittest.TestCase):
    def test_lttb_preserves_endpoints_and_threshold(self) -> None:
        x = list(range(1000))
        y = [float(np.sin(value / 17.0)) for value in x]
        sampled_x, sampled_y, indexes = lttb_downsample(x, y, 120)
        self.assertEqual(len(sampled_x), 120)
        self.assertEqual(len(sampled_y), 120)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 999)
        self.assertTrue(all(left < right for left, right in zip(indexes, indexes[1:])))

    def test_profile_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profiles.json"
            store = ChartProfileStore(path)
            profile = ChartProfile(name="My science profile", template="plotly_dark", line_width=4.0, range_slider=False)
            store.save(profile)
            loaded = store.load_all()["My science profile"]
            self.assertEqual(loaded.template, "plotly_dark")
            self.assertEqual(loaded.line_width, 4.0)
            self.assertFalse(loaded.range_slider)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)

    def test_offline_chart_and_dashboard_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            factory = InteractiveChartFactory(ChartProfile(default_height=480, range_slider=True))
            figure = factory.time_series(
                [
                    SeriesSpec("A", list(range(10)), [float(value) for value in range(10)]),
                    SeriesSpec("B", list(range(10)), [float(value * value) for value in range(10)]),
                ],
                title="Test chart",
            )
            chart_path = factory.write_html(figure, root / "chart.html", include_plotlyjs="directory")
            self.assertTrue(chart_path.exists())
            self.assertTrue((root / "plotly.min.js").exists())
            self.assertGreater(chart_path.stat().st_size, 1000)

            heatmap = factory.residual_heatmap([[0.1, -0.2], [0.3, -0.4]], x_labels=[2000, 2001], y_labels=["Index", "Age"])
            dashboard = factory.write_dashboard(
                {"Series": figure, "Residuals": heatmap},
                root / "dashboard.html",
                metadata={"Test": "yes"},
                include_plotlyjs="directory",
            )
            text = dashboard.read_text(encoding="utf-8")
            self.assertIn("tab-button", text)
            self.assertIn("Residuals", text)
            self.assertTrue((root / "plotly.min.js").exists())

    def test_specialised_charts_build(self) -> None:
        factory = InteractiveChartFactory(ChartProfile())
        figures = [
            factory.jitter_distribution([
                {"optimizer": "A", "objective": 1.0},
                {"optimizer": "A", "objective": 1.2},
                {"optimizer": "B", "objective": 1.1},
            ]),
            factory.optimizer_agreement([
                {"optimizer": "A", "objective": 1.0, "terminal_depletion": 0.3},
                {"optimizer": "B", "objective": 1.1, "terminal_depletion": 0.31},
            ]),
            factory.likelihood_profile([
                {"value": 0.1, "objective": 3.0},
                {"value": 0.2, "objective": 1.0},
                {"value": 0.3, "objective": 2.0},
            ]),
            factory.interval_coverage([
                {"parameter": "K", "nominal": 0.5, "empirical": 0.48},
                {"parameter": "K", "nominal": 0.9, "empirical": 0.87},
            ]),
        ]
        self.assertTrue(all(len(figure.data) >= 1 for figure in figures))


if __name__ == "__main__":
    unittest.main()
