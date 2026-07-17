from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from ui.datasets import DatasetEntry, DatasetLibrary, materialize_omega_timeseries
from ui.tutorial import FIRST_MODEL_STEPS
from ui.model_health import assess_model_health
from tools.download_noaa_test_data import _starter_inputs


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_library_discovers_beginner_demo() -> None:
    entries = DatasetLibrary(ROOT / "Data_Sets").scan()
    demo = next(item for item in entries if item.identifier == "omega-age-structured-demonstration")
    assert demo.difficulty == "Beginner"
    assert demo.primary_file is not None and demo.primary_file.exists()
    assert demo.age_composition is not None and demo.age_composition.exists()
    assert {"catch", "CPUE/index", "biomass", "age", "length"}.issubset(set(demo.data_types))


def test_dataset_library_includes_full_diagnostics_reference() -> None:
    entries = DatasetLibrary(ROOT / "Data_Sets").scan()
    diagnostics = next(item for item in entries if item.identifier == "omega-diagnostics-reference")
    assert diagnostics.coverage == "Full Omega dataset"
    assert diagnostics.primary_file is not None and diagnostics.primary_file.exists()
    assert diagnostics.age_composition is not None and diagnostics.age_composition.exists()
    assert diagnostics.length_composition is not None and diagnostics.length_composition.exists()
    chart_data = diagnostics.root / "all_functions_chart_data.csv"
    assert chart_data.exists()
    chart_columns = set(pd.read_csv(chart_data, nrows=1).columns)
    assert {"fishing_mortality", "recruitment", "parameter_x", "parameter_y", "fitness", "peel", "nominal", "empirical"}.issubset(chart_columns)
    assert {"Priority Diagnostics", "Likelihood profiles", "Age-structured ASPM", "Interval coverage"}.issubset(
        set(diagnostics.recommended_tools)
    )


def test_ss3_dataset_can_be_transparently_adapted_for_omega_workspaces(tmp_path: Path) -> None:
    fixture = ROOT / "validation_data" / "noaa_ss3" / "Simple"
    entry = DatasetEntry(
        identifier="fixture-simple",
        display_name="NOAA Simple",
        root=fixture,
        source="NOAA",
        model_type="Stock Synthesis",
    )
    adapted, note = materialize_omega_timeseries(entry, tmp_path)
    from stock_model.data_io import read_stock_file

    dataset = read_stock_file(adapted)
    assert len(dataset.frame) == 31
    assert dataset.frame["catch"].notna().all()
    assert "biomass and compositions remain unavailable" in note
    assert (adapted.parent / "source_adapter.json").exists()


def test_quick_model_health_distinguishes_confounding_from_accuracy() -> None:
    profile = assess_model_health(
        "Running fully refitted profile...",
        {
            "summary": {"status": "PASS", "points": 7, "failed_points": 0, "nonconverged_points": 0},
            "profile": [{"delta_nll": value} for value in (0.0, 0.1, 0.3, 0.5)],
        },
    )
    assert profile["quick_verdict"] == "POSSIBLE CONFOUNDING"
    assert profile["accuracy_evidence"].startswith("Not tested")
    coverage = assess_model_health(
        "Running formal coverage test...",
        {
            "summary": {
                "status": "FAIL",
                "maximum_absolute_coverage_error": 0.4,
                "maximum_absolute_mean_relative_bias": 0.3,
                "failure_fraction": 0.2,
            }
        },
    )
    assert coverage["quick_verdict"] == "INACCURATE / UNRELIABLE INTERVALS"


def test_first_model_tutorial_covers_live_fit_and_scientific_limits() -> None:
    actions = [step.action for step in FIRST_MODEL_STEPS]
    assert "load_beginner" in actions
    assert "run_fit" in actions
    assert "show_biomass" in actions
    assert "show_diagnostics" in actions
    assert "priority" in actions
    assert "mse" in actions
    text = " ".join(step.body + " " + step.caution for step in FIRST_MODEL_STEPS).lower()
    assert "not proof" in text or "not the unknown true biomass" in text


def test_guided_tutorial_requires_the_learner_to_use_real_controls() -> None:
    click_actions = {step.action for step in FIRST_MODEL_STEPS if step.requires_click}
    assert {
        "home",
        "datasets",
        "load_beginner",
        "integrated",
        "run_fit",
        "show_biomass",
        "show_diagnostics",
        "priority",
        "mse",
    }.issubset(click_actions)
    setup = next(step for step in FIRST_MODEL_STEPS if step.action == "configure_quick_fit")
    assert setup.requires_click is False


def test_noaa_starter_discovers_nonstandard_data_and_control_names(tmp_path: Path) -> None:
    data = tmp_path / "three_area_dat.ss"
    control = tmp_path / "three_area_ctl.ss"
    data.write_text("# data", encoding="utf-8")
    control.write_text("# control", encoding="utf-8")
    starter = tmp_path / "starter.ss"
    starter.write_text(
        "# Stock Synthesis starter\n\nthree_area_dat.ss #_datfile\nthree_area_ctl.ss #_ctlfile\n",
        encoding="utf-8",
    )
    assert _starter_inputs(starter) == (data, control)


def test_all_workspaces_embed_in_one_shell() -> None:
    tkinter = pytest.importorskip("tkinter")
    from omega_desktop import OmegaShell, WORKSPACES

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    try:
        shell = OmegaShell(root)
        diagnostics = next(item for item in shell.dataset_library.scan() if item.identifier == "omega-diagnostics-reference")
        shell.set_active_dataset(diagnostics)
        for mode in WORKSPACES:
            shell.navigate(mode)
            root.update_idletasks()
            assert shell.current_mode == mode
            assert shell.current_app is not None
        assert len(shell.frames) >= len(WORKSPACES)
        assert len(shell.main_pane.panes()) == 2
        assert any("Diagnostics Reference" in value for value in shell.dataset_picker.cget("values"))
        assert root.bind_all("<MouseWheel>")
        shell.navigate("settings")
        assert shell.page_title.get() == "Settings"
        shell.reset_standard_defaults()
        shell.navigate("integrated")
        assert shell.current_app.fit_population.get() == "36"
        assert shell.current_app.fit_generations.get() == "24"
        shell.navigate("parameters")
        assert Path(shell.current_app.dataset_path.get()) == diagnostics.primary_file
        assert shell.current_app.default_values["catch_multiplier"] == 1.0
        shell.navigate("priority")
        assert shell.current_app.analysis_level.get() == "standard"
        assert Path(shell.current_app.dataset_path.get()) == diagnostics.primary_file
        assert Path(shell.current_app.age_path.get()) == diagnostics.age_composition
        health = assess_model_health(
            "Running fully refitted profile...",
            {
                "summary": {"status": "PASS"},
                "profile": [{"delta_nll": value} for value in (0.0, 0.1, 0.3, 0.5)],
            },
        )
        shell.current_app._show_result("{}", health)
        root.update_idletasks()
        assert shell.current_app.health_tree.row_index == 2
        result_labels = [
            widget
            for widget in shell.current_app.health_tree.inner.winfo_children()
            if int(widget.grid_info().get("row", 0)) == 1
        ]
        assert len(result_labels) == 6
        assert all(int(widget.cget("wraplength")) >= 80 for widget in result_labels)
        shell.navigate("truthmse")
        assert shell.current_app.analysis_level.get() == "standard"
        shell.navigate("expert")
        assert shell.current_app.speed.get() == "standard"
        auto_started: list[bool] = []
        shell.current_app.run = lambda: auto_started.append(True)
        shell.run_full_auto()
        assert auto_started == [True]
        assert shell.current_app.mode.get() == "automatic"
        assert shell.current_app.skip_steps.get() == ""
        shell.navigate("quant")
        assert shell.current_app.population.get() == "48"
        assert shell.current_app.generations.get() == "35"
        shell.navigate("charts")
        assert shell.current_app.source_path == diagnostics.root / "all_functions_chart_data.csv"
        for position, orientation in (("Left", "horizontal"), ("Top", "vertical"), ("Right", "horizontal"), ("Bottom", "vertical")):
            shell.profile["sidebar_position"] = position
            shell._apply_sidebar_position()
            root.update_idletasks()
            assert shell.main_pane.cget("orient") == orientation
            assert len(shell.main_pane.panes()) == 2
        ss3 = DatasetEntry(
            identifier="fixture-simple",
            display_name="NOAA Simple",
            root=ROOT / "validation_data" / "noaa_ss3" / "Simple",
            source="NOAA",
            model_type="Stock Synthesis",
        )
        shell.set_active_dataset(ss3)
        shell.navigate("integrated")
        assert len(shell.current_app.dataset.frame) == 31
        shell.navigate("parameters")
        assert len(shell.current_app.dataset.frame) == 31
        shell.navigate("priority")
        assert Path(shell.current_app.dataset_path.get()).exists()
        shell.navigate("charts")
        assert shell.current_app.source_path is not None and shell.current_app.source_path.exists()
        shell.navigate("noaa")
        assert Path(shell.current_app.folder_var.get()) == ss3.root
        shell.reset_standard_defaults()
    finally:
        root.destroy()


def test_visual_parameter_lab_updates_the_chart_when_a_slider_moves() -> None:
    tkinter = pytest.importorskip("tkinter")
    from visual_parameter_lab_app import VisualParameterLabApp

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display is unavailable")
    root.geometry("1100x700")
    try:
        app = VisualParameterLabApp(root)
        root.update()
        assert app.dataset is not None
        before = app.terminal_biomass.get()
        app.catch_multiplier.set(1.6)
        app.update_scenario()
        root.update_idletasks()
        assert app.terminal_biomass.get() != before
        assert app.catch_slider.value_text.get() == "1.60 × observed"
        assert len(app.chart.find_all()) > 10
        assert len(app._sliders()) == 9
        app.reset_realistic_defaults()
        baseline = app.terminal_biomass.get()
        app.natural_mortality.set(0.40)
        app.update_scenario()
        assert app.terminal_biomass.get() != baseline
        app.reset_realistic_defaults()
        app.recruitment_strength.set(0.50)
        app.update_scenario()
        assert app.terminal_biomass.get() != baseline
        assert app.terminal_fishing_mortality.get().endswith("/ yr")
        app.reset_realistic_defaults()
        assert app.catch_multiplier.get() == 1.0
    finally:
        root.destroy()


def test_integrated_workspace_keeps_compact_controls_and_visible_results() -> None:
    tkinter = pytest.importorskip("tkinter")
    from omega_desktop import OmegaShell

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display is unavailable")
    root.geometry("1180x760")
    try:
        shell = OmegaShell(root)
        shell.navigate("integrated")
        root.update()
        app = shell.current_app
        assert app is not None
        app._set_initial_control_width()
        root.update_idletasks()

        control_width = app.body.sashpos(0)
        results_width = app.body.winfo_width() - control_width
        assert 285 <= control_width <= 340
        assert results_width >= 600
        assert getattr(app.control_canvas, "omega_role", "") == "controls"
        sections = []
        pending = list(app.controls.winfo_children())
        while pending:
            widget = pending.pop()
            if widget.__class__.__name__ == "CollapsibleSection":
                sections.append(widget)
            pending.extend(widget.winfo_children())
        assert len(sections) == 5
        assert any(not section.expanded for section in sections)
        collapsed = next(section for section in sections if not section.expanded)
        collapsed.toggle()
        assert collapsed.expanded
        assert app.notebook.tab(app.history_tab, "text") == "Biomass & F"
        assert app.notebook.tab(app.mse_tab, "text") == "MSE"
    finally:
        root.destroy()


def test_expert_workflow_completion_opens_a_combined_report(tmp_path: Path) -> None:
    tkinter = pytest.importorskip("tkinter")
    from expert_workflow_app import ExpertWorkflowApp
    from stock_model.interactive_charts import InteractiveChartFactory, SeriesSpec

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    try:
        app = ExpertWorkflowApp(root)
        figure = InteractiveChartFactory().time_series(
            [SeriesSpec("Depletion", [2022, 2023, 2024], [0.62, 0.59, 0.57])],
            title="Assessment trajectory",
            y_title="Depletion",
        )
        app.report_figures = {"Assessment trajectory": figure}
        app.result_path = tmp_path / "expert_workflow_latest.json"
        app.dashboard_path = tmp_path / "expert_workflow_dashboard.html"
        result = {
            "summary": {
                "status": "WARN",
                "reliability_grade": "B",
                "steps_completed": 1,
                "steps": 1,
                "required_failures": 0,
                "warnings": 1,
                "skipped": 0,
                "terminal_depletion": 0.57,
                "mode": "automatic",
                "speed": "standard",
            },
            "steps": [{"name": "Retrospective", "status": "WARN", "required": True, "message": "Inspect Mohn's rho", "error": None}],
            "results": {
                "reliability": {
                    "items": [{"name": "Retrospective bias", "status": "WARN", "value": 0.18, "criterion": "absolute rho", "impact": "medium", "why": "Peels diverge."}]
                }
            },
        }
        app._show_result(result)
        root.update_idletasks()
        assert app.notebook.select() == str(app.report_tab)
        assert app.report_chart_preview.figure is figure
        assert len(app.report_table.get_children()) >= 3
        assert "scientifically correct" in app.report_overview_text.get("1.0", "end").lower()
        assert str(app.dashboard_path) in app.report_files_text.get("1.0", "end")
    finally:
        root.destroy()
