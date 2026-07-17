from __future__ import annotations

import json
import subprocess
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, BOTTOM, END, LEFT, RIGHT, TOP, X, Y, BooleanVar, Canvas, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Callable

import numpy as np

from chart_studio_app import NativeChartPreview, ScrollFrame
from stock_model.core import ModelSettings, ProjectionSettings, fit, project
from stock_model.data_io import StockDataset, read_stock_file
from stock_model.quant_lab import (
    QuantOptimizerSettings,
    detect_index_regime_shift,
    objective_surface,
    run_global_optimizer,
    run_hcr_genetic_optimization,
    run_stress_tests,
    sobol_projection_screen,
)
from stock_model.quant_report import generate_quant_report
from stock_model.quant_validation import (
    EnsembleSettings,
    OptimizerAgreementSettings,
    WalkForwardSettings,
    run_model_ensemble,
    run_optimizer_agreement,
    run_walk_forward_validation,
)
from stock_model.interactive_charts import ChartProfile, InteractiveChartFactory, SeriesSpec


APP_TITLE = "Omega FISH Model — Quant Lab"
ROOT = Path(__file__).resolve().parent


class QuantLabApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1480x900")
        self.root.minsize(1100, 720)
        self.dataset: StockDataset | None = None
        self.fit_result = None
        self.optimizer_output: dict[str, Any] | None = None
        self.walk_forward_output: dict[str, Any] | None = None
        self.optimizer_agreement_output: dict[str, Any] | None = None
        self.model_ensemble_output: dict[str, Any] | None = None
        self.risk_output: dict[str, Any] | None = None
        self.stress_output: dict[str, Any] | None = None
        self.sobol_output: dict[str, Any] | None = None
        self.regime_output: dict[str, Any] | None = None
        self.full_output: dict[str, Any] | None = None
        self.surface_rows: list[dict[str, Any]] = []
        self.surface_angle = 42.0
        self.busy = BooleanVar(value=False)
        self.processing_text = StringVar(value="")
        self.status = StringVar(value="Load a CSV or Excel dataset containing at least year and catch.")
        self.model = StringVar(value="schaefer")
        self.algorithm = StringVar(value="differential_evolution")
        self.population = StringVar(value="48")
        self.generations = StringVar(value="35")
        self.search_draws = StringVar(value="300")
        self.projection_years = StringVar(value="20")
        self.projection_iterations = StringVar(value="300")
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=TOP, fill=X)
        ttk.Button(toolbar, text="Load dataset", command=self.load_dataset).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Run baseline fit", command=self.run_baseline).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Run full Quant Lab", command=self.run_full_lab).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Export results", command=self.export_results).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Open Integrated Assessment Lab", command=self.open_integrated_assessment).pack(side=LEFT, padx=8)
        ttk.Button(toolbar, text="Expert Workflow", command=lambda: self._open_workspace("expert")).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Interactive Charts", command=lambda: self._open_workspace("charts")).pack(side=LEFT, padx=4)
        ttk.Button(toolbar, text="Open Validation and MSE", command=self.open_existing_interface).pack(side=LEFT, padx=8)
        ttk.Label(toolbar, textvariable=self.status).pack(side=LEFT, padx=12)

        self.processing_strip = ttk.Frame(self.root, padding=(10, 5))
        ttk.Label(self.processing_strip, textvariable=self.processing_text).pack(side=LEFT, padx=(0, 10))
        self.processing_bar = ttk.Progressbar(self.processing_strip, mode="indeterminate", length=360)
        self.processing_bar.pack(side=LEFT, fill=X, expand=True)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        self.body = body
        body.pack(fill=BOTH, expand=True)
        controls_shell = ScrollFrame(body, width=310)
        body.add(controls_shell, weight=0)
        controls = controls_shell.inner
        self.controls_scroll = controls_shell
        main = ttk.Frame(body, padding=6)
        body.add(main, weight=1)

        ttk.Label(controls, text="Model and run controls", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        self._label_combo(controls, "Production model", self.model, ["schaefer", "fox", "pella"])
        self._label_combo(controls, "Global optimiser", self.algorithm, ["differential_evolution", "genetic", "cma_es", "nelder_mead", "random_multistart"])
        self._label_entry(controls, "Population", self.population)
        self._label_entry(controls, "Generations", self.generations)
        self._label_entry(controls, "Baseline search draws", self.search_draws)
        self._label_entry(controls, "Projection years", self.projection_years)
        self._label_entry(controls, "Projection simulations", self.projection_iterations)
        ttk.Separator(controls).pack(fill=X, pady=10)
        ttk.Button(controls, text="Global parameter optimisation", command=self.run_optimizer).pack(fill=X, pady=3)
        ttk.Button(controls, text="Five-optimizer agreement", command=self.run_optimizer_agreement_ui).pack(fill=X, pady=3)
        ttk.Button(controls, text="Cross-model ensemble", command=self.run_model_ensemble_ui).pack(fill=X, pady=3)
        ttk.Button(controls, text="Walk-forward validation", command=self.run_walk_forward_ui).pack(fill=X, pady=3)
        ttk.Button(controls, text="8D diagnostic explorer", command=self.show_high_dimensional).pack(fill=X, pady=3)
        ttk.Button(controls, text="Optimization grid (MetaTrader style)", command=self.run_surface).pack(fill=X, pady=3)
        ttk.Button(controls, text="Genetic HCR risk frontier", command=self.run_risk).pack(fill=X, pady=3)
        ttk.Button(controls, text="Stress-test data", command=self.run_stress).pack(fill=X, pady=3)
        ttk.Button(controls, text="Sensitivity / Sobol screen", command=self.run_sobol).pack(fill=X, pady=3)
        ttk.Button(controls, text="Regime-shift screen", command=self.run_regime).pack(fill=X, pady=3)
        ttk.Separator(controls).pack(fill=X, pady=10)
        ttk.Label(
            controls,
            text=(
                "Interpretation rule:\nOptimisation finds parameter combinations that score well. "
                "It does not prove the model, data or assumptions are correct. Use the linked diagnostics, "
                "stress tests and risk frontier before accepting a result."
            ),
            wraplength=270,
            justify="left",
        ).pack(anchor="w")

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=BOTH, expand=True)
        self.data_tab = ttk.Frame(self.notebook)
        self.fit_tab = ttk.Frame(self.notebook)
        self.optimizer_tab = ttk.Frame(self.notebook)
        self.agreement_tab = ttk.Frame(self.notebook)
        self.ensemble_tab = ttk.Frame(self.notebook)
        self.validation_tab = ttk.Frame(self.notebook)
        self.highd_tab = ttk.Frame(self.notebook)
        self.surface_tab = ttk.Frame(self.notebook)
        self.risk_tab = ttk.Frame(self.notebook)
        self.stress_tab = ttk.Frame(self.notebook)
        self.sobol_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        for tab, title in [
            (self.data_tab, "Data"),
            (self.fit_tab, "Fit"),
            (self.optimizer_tab, "Optimiser"),
            (self.agreement_tab, "Optimiser Agreement"),
            (self.ensemble_tab, "Model Ensemble"),
            (self.validation_tab, "Walk-Forward"),
            (self.highd_tab, "8D Diagnostics"),
            (self.surface_tab, "Optimization Grid"),
            (self.risk_tab, "Risk Frontier"),
            (self.stress_tab, "Stress Tests"),
            (self.sobol_tab, "Sensitivity"),
            (self.log_tab, "Log"),
        ]:
            self.notebook.add(tab, text=title)

        self.data_tree = self._tree(self.data_tab)
        self.fit_tree = self._tree(self.fit_tab)
        optimizer_split = ttk.Panedwindow(self.optimizer_tab, orient="vertical")
        optimizer_split.pack(fill=BOTH, expand=True)
        optimizer_chart = ttk.LabelFrame(optimizer_split, text="Ranked objective values")
        optimizer_table = ttk.LabelFrame(optimizer_split, text="Candidate parameter sets")
        optimizer_split.add(optimizer_chart, weight=1)
        optimizer_split.add(optimizer_table, weight=2)
        self.optimizer_preview = NativeChartPreview(optimizer_chart)
        self.optimizer_preview.pack(fill=BOTH, expand=True)
        self.optimizer_tree = self._tree(optimizer_table)
        self.agreement_tree = self._tree(self.agreement_tab)
        self.ensemble_tree = self._tree(self.ensemble_tab)
        self.validation_tree = self._tree(self.validation_tab)
        self.parallel_canvas = Canvas(self.highd_tab, background="white", height=390)
        self.parallel_canvas.pack(fill=BOTH, expand=True)
        highd_tables = ttk.Panedwindow(self.highd_tab, orient="horizontal")
        highd_tables.pack(fill=BOTH, expand=True)
        importance_frame = ttk.LabelFrame(highd_tables, text="Parameter influence and rank correlations")
        highd_tables.add(importance_frame, weight=1)
        self.importance_tree = self._tree(importance_frame)
        ident_frame = ttk.LabelFrame(highd_tables, text="Local curvature, profiles and weak directions")
        highd_tables.add(ident_frame, weight=1)
        self.identifiability_tree = self._tree(ident_frame)
        self.surface_views = ttk.Notebook(self.surface_tab)
        self.surface_views.pack(fill=BOTH, expand=True)
        surface_grid_tab = ttk.Frame(self.surface_views)
        surface_3d_tab = ttk.Frame(self.surface_views)
        self.surface_views.add(surface_grid_tab, text="Parameter grid")
        self.surface_views.add(surface_3d_tab, text="Rotatable 3D")
        self.surface_preview = NativeChartPreview(surface_grid_tab)
        self.surface_preview.pack(fill=BOTH, expand=True)
        self.surface_canvas = Canvas(surface_3d_tab, background="white")
        self.surface_canvas.pack(fill=BOTH, expand=True)
        self.surface_canvas.bind("<Configure>", lambda _event: self.draw_surface())
        self.surface_canvas.bind("<ButtonPress-1>", self._surface_drag_start)
        self.surface_canvas.bind("<B1-Motion>", self._surface_drag)
        self._surface_last_x = 0
        self.risk_tree = self._tree(self.risk_tab)
        self.stress_tree = self._tree(self.stress_tab)
        self.sobol_tree = self._tree(self.sobol_tab)
        self.log_text = __import__("tkinter").Text(self.log_tab, wrap="word", font=("Consolas", 10))
        self.log_text.pack(fill=BOTH, expand=True)

    def _label_entry(self, parent, label, variable) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 1))
        ttk.Entry(parent, textvariable=variable).pack(fill=X)

    def _label_combo(self, parent, label, variable, values) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 1))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").pack(fill=X)

    def _tree(self, parent) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=BOTH, expand=True)
        tree = ttk.Treeview(frame, show="headings")
        ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        ybar.pack(side=RIGHT, fill=Y)
        xbar.pack(side="bottom", fill=X)
        return tree

    def load_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load fish-stock dataset",
            filetypes=[("Data files", "*.csv *.xlsx *.xlsm"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.dataset = read_stock_file(path)
            self.fit_result = None
            self.optimizer_output = None
            self.walk_forward_output = None
            self.optimizer_agreement_output = None
            self.model_ensemble_output = None
            self.risk_output = None
            self.stress_output = None
            self.sobol_output = None
            self.regime_output = None
            self.full_output = None
            self._populate_tree(self.data_tree, self.dataset.frame.head(500).where(self.dataset.frame.notna(), "").to_dict(orient="records"))
            self.status.set(f"Loaded {self.dataset.name}: {len(self.dataset.frame)} years.")
            self.notebook.select(self.data_tab)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _settings(self) -> ModelSettings:
        return ModelSettings(
            model=self.model.get(),
            search_draws=max(120, int(self.search_draws.get())),
            seed=4107,
        )

    def _require_dataset(self) -> StockDataset:
        if self.dataset is None:
            raise ValueError("Load a dataset first.")
        return self.dataset

    def _run_background(self, label: str, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        if self.busy.get():
            self.status.set(f"Still processing: {self.processing_text.get()}")
            return
        self.busy.set(True)
        self.processing_text.set(label)
        self.processing_strip.pack(side=BOTTOM, fill=X, before=self.body)
        self.processing_bar.start(12)
        self.status.set(label)
        outcome: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def target() -> None:
            try:
                outcome.put(("ok", work()))
            except Exception:
                outcome.put(("error", traceback.format_exc()))

        def poll() -> None:
            try:
                state, value = outcome.get_nowait()
            except Empty:
                try:
                    if self.root.winfo_exists():
                        self.root.after(30, poll)
                except Exception:
                    pass
                return
            if state == "ok":
                self._finish_background(done, value, label)
            else:
                self._fail_background(label, str(value))

        threading.Thread(target=target, daemon=True).start()
        self.root.after(30, poll)

    def _finish_background(self, done: Callable[[Any], None], result: Any, label: str) -> None:
        try:
            done(result)
            self.status.set(label.replace("Running", "Completed"))
        finally:
            self._stop_processing()

    def _fail_background(self, label: str, trace: str) -> None:
        self.log(trace)
        self.status.set(f"Failed: {label}")
        self._stop_processing()
        shell = getattr(self.root, "omega_shell", None)
        if shell is not None:
            shell.log_error(label, RuntimeError(trace.splitlines()[-1] if trace.splitlines() else label), trace)
        messagebox.showerror(APP_TITLE, trace.splitlines()[-1] if trace.splitlines() else trace)

    def _stop_processing(self) -> None:
        self.processing_bar.stop()
        self.processing_strip.pack_forget()
        self.processing_text.set("")
        self.busy.set(False)

    def run_baseline(self) -> None:
        try:
            dataset = self._require_dataset()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background("Running baseline fit...", lambda: fit(dataset, self._settings()), self._show_baseline)

    def _show_baseline(self, result) -> None:
        self.fit_result = result
        rows = [{"quantity": key, "value": value} for key, value in result.best.items()]
        components = result.diagnostics.get("objective_components") or {}
        rows.extend({"quantity": f"objective: {key}", "value": value} for key, value in components.items())
        self._populate_tree(self.fit_tree, rows)
        self.notebook.select(self.fit_tab)
        self.log(json.dumps({"baseline_fit": result.best, "diagnostics": result.diagnostics}, indent=2, default=str))

    def run_optimizer(self) -> None:
        try:
            dataset = self._require_dataset()
            config = QuantOptimizerSettings(
                algorithm=self.algorithm.get(),
                population=max(12, int(self.population.get())),
                generations=max(1, int(self.generations.get())),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running global optimiser...",
            lambda: run_global_optimizer(dataset, self._settings(), config),
            self._show_optimizer,
        )

    def _show_optimizer(self, output: dict[str, Any]) -> None:
        self.optimizer_output = output
        rows = []
        for row in output.get("candidates", [])[:200]:
            rows.append({key: value for key, value in row.items() if key != "objective_components"})
        self._populate_tree(self.optimizer_tree, rows)
        objectives = [float(row["objective"]) for row in rows if row.get("objective") is not None]
        if objectives:
            figure = InteractiveChartFactory(ChartProfile()).time_series(
                [SeriesSpec("Objective", list(range(1, len(objectives) + 1)), objectives, mode="lines+markers")],
                title="Optimizer candidates ranked from best to worst",
                x_title="Candidate rank",
                y_title="Objective (lower is better)",
            )
            self.optimizer_preview.show_figure(figure, "Optimizer candidates ranked from best to worst", "Candidate rank", "Objective")
        self.notebook.select(self.optimizer_tab)
        self.log(json.dumps({"optimizer_summary": output.get("summary"), "best": rows[0] if rows else {}}, indent=2, default=str))

    def show_high_dimensional(self) -> None:
        if not self.optimizer_output:
            self.run_optimizer()
            return
        diagnostics = self.optimizer_output.get("diagnostics") or {}
        self.draw_parallel_coordinates(diagnostics)
        self._populate_tree(self.importance_tree, diagnostics.get("importance") or [])
        ident = diagnostics.get("local_identifiability") or {}
        ident_rows = [
            {"quantity": "status", "value": ident.get("status")},
            {"quantity": "effective_rank", "value": ident.get("effective_rank")},
            {"quantity": "dimensions", "value": ident.get("dimensions")},
            {"quantity": "condition_number", "value": ident.get("condition_number")},
        ]
        for row in ident.get("weak_directions") or []:
            ident_rows.append(
                {
                    "quantity": f"weak_direction_{row.get('direction')}",
                    "value": row.get("curvature_eigenvalue"),
                    "loadings": json.dumps(row.get("dominant_loadings") or {}, separators=(",", ":")),
                }
            )
        self._populate_tree(self.identifiability_tree, ident_rows)
        self.notebook.select(self.highd_tab)

    def run_optimizer_agreement_ui(self) -> None:
        try:
            dataset = self._require_dataset()
            settings = OptimizerAgreementSettings(
                population=max(12, int(self.population.get()) // 2),
                generations=max(2, int(self.generations.get()) // 3),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running independent optimiser agreement test...",
            lambda: run_optimizer_agreement(dataset, self._settings(), settings),
            self._show_optimizer_agreement,
        )

    def _show_optimizer_agreement(self, output: dict[str, Any]) -> None:
        self.optimizer_agreement_output = output
        rows = list(output.get("runs") or [])
        rows.extend(
            {
                "algorithm": f"AGREEMENT: {row.get('quantity')}",
                "objective": row.get("mean"),
                "objective_delta": row.get("range"),
                "terminal_depletion": row.get("coefficient_of_variation"),
                "identifiability_status": "",
            }
            for row in output.get("agreement") or []
        )
        self._populate_tree(self.agreement_tree, rows)
        self.notebook.select(self.agreement_tab)
        self.log(json.dumps({"optimizer_agreement": output.get("summary")}, indent=2, default=str))

    def run_model_ensemble_ui(self) -> None:
        try:
            dataset = self._require_dataset()
            settings = EnsembleSettings(
                search_draws=max(120, int(self.search_draws.get()) // 2),
                projection_years=max(1, int(self.projection_years.get())),
                projection_iterations=max(60, int(self.projection_iterations.get())),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running Schaefer, Fox and Pella ensemble...",
            lambda: run_model_ensemble(dataset, self._settings(), settings),
            self._show_model_ensemble,
        )

    def _show_model_ensemble(self, output: dict[str, Any]) -> None:
        self.model_ensemble_output = output
        rows = list(output.get("models") or [])
        rows.extend(
            {
                "model": f"YEAR {row.get('year')}",
                "terminal_depletion": row.get("candidate_weighted_depletion"),
                "projection_terminal_depletion": row.get("model_depletion_range"),
                "projection_terminal_limit_risk": row.get("candidate_weighted_limit_risk"),
                "risk_adjusted_yield_index": row.get("candidate_weighted_catch"),
            }
            for row in output.get("combined_projection") or []
        )
        self._populate_tree(self.ensemble_tree, rows)
        self.notebook.select(self.ensemble_tab)
        self.log(json.dumps({"model_ensemble": output.get("summary")}, indent=2, default=str))

    def run_walk_forward_ui(self) -> None:
        try:
            dataset = self._require_dataset()
            settings = WalkForwardSettings(
                minimum_training_years=6,
                holdout_years=1,
                search_draws=max(120, int(self.search_draws.get()) // 2),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running rolling walk-forward validation...",
            lambda: run_walk_forward_validation(dataset, self._settings(), settings),
            self._show_walk_forward,
        )

    def _show_walk_forward(self, output: dict[str, Any]) -> None:
        self.walk_forward_output = output
        rows = list(output.get("folds") or [])
        if not rows:
            rows = [{"status": output.get("summary", {}).get("status"), **(output.get("summary") or {})}]
        self._populate_tree(self.validation_tree, rows)
        self.notebook.select(self.validation_tab)
        self.log(json.dumps({"walk_forward": output.get("summary")}, indent=2, default=str))

    def run_surface(self) -> None:
        try:
            dataset = self._require_dataset()
            config = QuantOptimizerSettings(
                algorithm=self.algorithm.get(),
                population=max(12, int(self.population.get())),
                generations=max(1, int(self.generations.get())),
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def work():
            optimizer = self.optimizer_output or run_global_optimizer(dataset, self._settings(), config)
            best = optimizer["candidates"][0]
            return optimizer, objective_surface(dataset, self._settings(), best, points=20)

        self._run_background(
            "Running optimization and parameter grid...",
            work,
            self._show_surface_package,
        )

    def _show_surface_package(self, value) -> None:
        optimizer, rows = value
        if self.optimizer_output is None:
            self._show_optimizer(optimizer)
        self._show_surface(rows)

    def _show_surface(self, rows) -> None:
        self.surface_rows = rows
        figure = InteractiveChartFactory(ChartProfile()).optimization_surface(
            rows,
            x_key="r",
            y_key="k",
            value_key="objective_delta",
            maximize=False,
            title="r × K optimization grid — star marks the best cell",
        )
        self.surface_preview.show_figure(figure, "r × K optimization grid — star marks the best cell", "Growth rate r", "Carrying capacity K")
        self.draw_surface()
        self.notebook.select(self.surface_tab)
        self.surface_views.select(0)

    def run_risk(self) -> None:
        try:
            dataset = self._require_dataset()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def work():
            result = self.fit_result or fit(dataset, self._settings())
            return result, run_hcr_genetic_optimization(
                result,
                years=max(1, int(self.projection_years.get())),
                iterations=max(40, int(self.projection_iterations.get())),
                population=max(12, int(self.population.get()) // 2),
                generations=max(2, int(self.generations.get()) // 2),
            )

        self._run_background("Running genetic HCR risk frontier...", work, self._show_risk)

    def _show_risk(self, value) -> None:
        self.fit_result, output = value
        self.risk_output = output
        rows = [{key: val for key, val in row.items() if key != "objectives"} for row in output.get("pareto", [])]
        self._populate_tree(self.risk_tree, rows)
        self.notebook.select(self.risk_tab)
        self.log(json.dumps({"risk_frontier": output.get("summary"), "pareto": rows}, indent=2, default=str))

    def run_stress(self) -> None:
        try:
            dataset = self._require_dataset()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running controlled data stress tests...",
            lambda: run_stress_tests(dataset, self._settings(), search_draws=max(120, int(self.search_draws.get()) // 2)),
            self._show_stress,
        )

    def _show_stress(self, output) -> None:
        self.stress_output = output
        self._populate_tree(self.stress_tree, output.get("stress_tests") or [])
        self.notebook.select(self.stress_tab)
        self.log(json.dumps(output.get("summary"), indent=2, default=str))

    def run_sobol(self) -> None:
        try:
            dataset = self._require_dataset()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def work():
            result = self.fit_result or fit(dataset, self._settings())
            return result, sobol_projection_screen(result, years=max(1, int(self.projection_years.get())), samples=128)

        self._run_background("Running Saltelli-style sensitivity screen...", work, self._show_sobol)

    def _show_sobol(self, value) -> None:
        self.fit_result, output = value
        self.sobol_output = output
        self._populate_tree(self.sobol_tree, output.get("sensitivity") or [])
        self.notebook.select(self.sobol_tab)
        self.log(json.dumps(output, indent=2, default=str))

    def run_regime(self) -> None:
        try:
            output = detect_index_regime_shift(self._require_dataset())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self.regime_output = output
        rows = output.get("candidates") or [output]
        self._populate_tree(self.sobol_tree, rows)
        self.notebook.select(self.sobol_tab)
        self.log(json.dumps(output, indent=2, default=str))
        self.status.set("Regime-shift screen completed.")

    def run_full_lab(self) -> None:
        try:
            dataset = self._require_dataset()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        def work():
            baseline = fit(dataset, self._settings())
            optimizer = run_global_optimizer(
                dataset,
                self._settings(),
                QuantOptimizerSettings(
                    algorithm=self.algorithm.get(),
                    population=max(12, int(self.population.get())),
                    generations=max(1, int(self.generations.get())),
                ),
            )
            surface = objective_surface(dataset, self._settings(), optimizer["candidates"][0], points=18)
            risk = run_hcr_genetic_optimization(
                baseline,
                years=max(1, int(self.projection_years.get())),
                iterations=max(40, int(self.projection_iterations.get())),
                population=max(12, int(self.population.get()) // 2),
                generations=max(2, int(self.generations.get()) // 2),
            )
            stress = run_stress_tests(dataset, self._settings(), search_draws=max(120, int(self.search_draws.get()) // 2))
            sobol = sobol_projection_screen(baseline, years=max(1, int(self.projection_years.get())), samples=128)
            regime = detect_index_regime_shift(dataset)
            walk_forward = run_walk_forward_validation(
                dataset,
                self._settings(),
                WalkForwardSettings(
                    minimum_training_years=6,
                    holdout_years=1,
                    search_draws=max(120, int(self.search_draws.get()) // 2),
                ),
            )
            agreement = run_optimizer_agreement(
                dataset,
                self._settings(),
                OptimizerAgreementSettings(
                    population=max(12, int(self.population.get()) // 2),
                    generations=max(2, int(self.generations.get()) // 4),
                ),
            )
            ensemble = run_model_ensemble(
                dataset,
                self._settings(),
                EnsembleSettings(
                    search_draws=max(120, int(self.search_draws.get()) // 2),
                    projection_years=max(1, int(self.projection_years.get())),
                    projection_iterations=max(60, int(self.projection_iterations.get())),
                ),
            )
            return {
                "baseline": baseline,
                "optimizer": optimizer,
                "surface": surface,
                "risk": risk,
                "stress": stress,
                "sobol": sobol,
                "regime": regime,
                "walk_forward": walk_forward,
                "optimizer_agreement": agreement,
                "model_ensemble": ensemble,
            }

        self._run_background("Running full Quant Lab...", work, self._show_full_lab)

    def _show_full_lab(self, output) -> None:
        self.full_output = output
        self.fit_result = output["baseline"]
        self.optimizer_output = output["optimizer"]
        self.surface_rows = output["surface"]
        self.risk_output = output["risk"]
        self.stress_output = output["stress"]
        self.sobol_output = output["sobol"]
        self.regime_output = output["regime"]
        self.walk_forward_output = output["walk_forward"]
        self.optimizer_agreement_output = output["optimizer_agreement"]
        self.model_ensemble_output = output["model_ensemble"]
        self._show_baseline(self.fit_result)
        self._show_optimizer(self.optimizer_output)
        self.show_high_dimensional()
        self.draw_surface()
        self._populate_tree(self.risk_tree, [{k: v for k, v in row.items() if k != "objectives"} for row in output["risk"].get("pareto", [])])
        self._populate_tree(self.stress_tree, output["stress"].get("stress_tests") or [])
        self._populate_tree(self.sobol_tree, output["sobol"].get("sensitivity") or [])
        self._populate_tree(self.validation_tree, output["walk_forward"].get("folds") or [])
        self._populate_tree(self.agreement_tree, output["optimizer_agreement"].get("runs") or [])
        self._populate_tree(self.ensemble_tree, output["model_ensemble"].get("models") or [])
        self.log(json.dumps({key: value.get("summary", value.get("best", {})) if isinstance(value, dict) else {} for key, value in output.items() if key != "baseline"}, indent=2, default=str))
        self.notebook.select(self.highd_tab)

    def draw_parallel_coordinates(self, diagnostics: dict[str, Any]) -> None:
        canvas = self.parallel_canvas
        canvas.delete("all")
        rows = diagnostics.get("parallel_coordinates") or []
        names = diagnostics.get("parameter_names") or []
        width = max(canvas.winfo_width(), 800)
        height = max(canvas.winfo_height(), 380)
        margin_x, margin_y = 65, 45
        if not rows or len(names) < 2:
            canvas.create_text(width / 2, height / 2, text="Run the optimiser to create 8D diagnostic paths.")
            return
        xs = np.linspace(margin_x, width - margin_x, len(names))
        for x, name in zip(xs, names):
            canvas.create_line(x, margin_y, x, height - margin_y, fill="#777")
            canvas.create_text(x, height - margin_y + 20, text=name, angle=25, anchor="nw")
        objectives = np.array([float(row.get("objective", 0.0)) for row in rows])
        low, high = np.quantile(objectives, [0.05, 0.95]) if len(objectives) > 1 else (objectives[0], objectives[0] + 1)
        for row in reversed(rows[:120]):
            values = row.get("values") or {}
            points = []
            for x, name in zip(xs, names):
                value = min(max(float(values.get(name, 0.0)), 0.0), 1.0)
                y = height - margin_y - value * (height - 2 * margin_y)
                points.extend([float(x), float(y)])
            ratio = min(max((float(row.get("objective", low)) - low) / max(high - low, 1e-9), 0.0), 1.0)
            red = int(40 + 190 * ratio)
            blue = int(210 - 150 * ratio)
            canvas.create_line(*points, fill=f"#{red:02x}55{blue:02x}", width=2 if row.get("rank") == 1 else 1)
        canvas.create_text(margin_x, 18, anchor="w", text="Eight-dimensional parallel coordinates — lower-objective runs are darker/bluer")

    def draw_surface(self) -> None:
        canvas = self.surface_canvas
        canvas.delete("all")
        rows = self.surface_rows
        width = max(canvas.winfo_width(), 800)
        height = max(canvas.winfo_height(), 500)
        if not rows:
            canvas.create_text(width / 2, height / 2, text="Run the optimiser, then generate the 3D r-K-objective surface.")
            return
        k_values = np.array([float(row["k"]) for row in rows])
        r_values = np.array([float(row["r"]) for row in rows])
        z_values = np.array([float(row["objective_delta"]) for row in rows])
        z_cap = max(float(np.quantile(z_values[np.isfinite(z_values)], 0.90)), 1e-9)
        xn = (np.log(k_values) - np.log(k_values).min()) / max(float(np.ptp(np.log(k_values))), 1e-9)
        yn = (np.log(r_values) - np.log(r_values).min()) / max(float(np.ptp(np.log(r_values))), 1e-9)
        zn = np.clip(z_values / z_cap, 0.0, 1.0)
        angle = np.deg2rad(self.surface_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        scale = min(width * 0.36, height * 0.48)
        cx, cy = width * 0.50, height * 0.72
        projected = []
        for x, y, z, row in zip(xn, yn, zn, rows):
            rx = (x - 0.5) * cos_a - (y - 0.5) * sin_a
            ry = (x - 0.5) * sin_a + (y - 0.5) * cos_a
            px = cx + rx * scale * 1.6
            py = cy + ry * scale * 0.65 - z * scale * 0.95
            projected.append((py, px, z, row))
        for _py, px, z, row in sorted(projected, reverse=True):
            ratio = min(max(z, 0.0), 1.0)
            red = int(35 + 215 * ratio)
            green = int(155 - 90 * ratio)
            blue = int(220 - 180 * ratio)
            radius = 3 if ratio > 0.12 else 5
            canvas.create_oval(px - radius, _py - radius, px + radius, _py + radius, fill=f"#{red:02x}{green:02x}{blue:02x}", outline="")
        canvas.create_text(18, 18, anchor="nw", text="3D response surface: r × K × objective delta\nDrag horizontally to rotate. Blue/low points are near the optimum.")
        canvas.create_text(width * 0.16, height - 24, text="K axis")
        canvas.create_text(width * 0.84, height - 24, text="r axis")
        canvas.create_text(65, height * 0.35, text="Objective\ndelta")

    def _surface_drag_start(self, event) -> None:
        self._surface_last_x = event.x

    def _surface_drag(self, event) -> None:
        self.surface_angle += (event.x - self._surface_last_x) * 0.5
        self._surface_last_x = event.x
        self.draw_surface()

    def _populate_tree(self, tree: ttk.Treeview, rows: list[dict[str, Any]]) -> None:
        tree.delete(*tree.get_children())
        if not rows:
            tree["columns"] = ("message",)
            tree.heading("message", text="Message")
            tree.column("message", width=700)
            tree.insert("", END, values=("No rows available.",))
            return
        columns = []
        for row in rows:
            for key in row:
                if key not in columns and not isinstance(row[key], (dict, list, tuple)):
                    columns.append(key)
        columns = columns[:28]
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            tree.column(column, width=max(90, min(190, len(column) * 10)), anchor="w")
        for row in rows[:1000]:
            values = [self._format(row.get(column)) for column in columns]
            tree.insert("", END, values=values)

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.6g}"
        return "" if value is None else str(value)

    def export_results(self) -> None:
        if not self.optimizer_output and not self.fit_result:
            messagebox.showinfo(APP_TITLE, "Run at least one analysis before exporting.")
            return
        selected = filedialog.askdirectory(title="Choose a folder for the Quant Lab analysis package")
        if not selected:
            return
        dataset_summary = {}
        if self.dataset is not None:
            dataset_summary = {
                "dataset": self.dataset.name,
                "rows": int(len(self.dataset.frame)),
                "first_year": int(self.dataset.frame["year"].min()),
                "last_year": int(self.dataset.frame["year"].max()),
            }
        payload = {
            "summary": {
                **dataset_summary,
                "application": APP_TITLE,
                "modules_completed": [
                    name
                    for name, value in [
                        ("baseline_fit", self.fit_result),
                        ("global_optimizer", self.optimizer_output),
                        ("risk_frontier", self.risk_output),
                        ("stress_tests", self.stress_output),
                        ("sensitivity", self.sobol_output),
                        ("regime_shift", self.regime_output),
                        ("walk_forward", self.walk_forward_output),
                        ("optimizer_agreement", self.optimizer_agreement_output),
                        ("model_ensemble", self.model_ensemble_output),
                    ]
                    if value is not None
                ],
            },
            "baseline_fit": self.fit_result.__dict__ if self.fit_result else {},
            "optimizer": self.optimizer_output or {},
            "surface_3d": self.surface_rows,
            "risk_frontier": self.risk_output or {},
            "stress_tests": self.stress_output or {},
            "sobol": self.sobol_output or {},
            "regime_shift": self.regime_output or {},
            "walk_forward": self.walk_forward_output or {},
            "optimizer_agreement": self.optimizer_agreement_output or {},
            "model_ensemble": self.model_ensemble_output or {},
        }
        try:
            report = generate_quant_report(payload, Path(selected) / "Omega_FISH_Quant_Lab_Report")
            self.status.set(f"Exported analysis package to {report['folder']}")
            self.log(json.dumps({"exported_report": report}, indent=2, default=str))
            messagebox.showinfo(
                APP_TITLE,
                f"Analysis package created.\n\nHTML report:\n{report['html']}\n\nJSON and CSV tables are in the same folder.",
            )
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _launch_workspace(self, mode: str, source_script: str) -> None:
        try:
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable, "--mode", mode], cwd=str(Path(sys.executable).resolve().parent))
                return
            app = ROOT / source_script
            if not app.exists():
                raise FileNotFoundError(f"Could not find {app}")
            subprocess.Popen([sys.executable, str(app)], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _open_workspace(self, mode: str) -> None:
        try:
            subprocess.Popen([sys.executable, str(ROOT / "omega_desktop.py"), "--mode", mode], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def open_integrated_assessment(self) -> None:
        self._launch_workspace("integrated", "integrated_assessment_app.py")

    def open_existing_interface(self) -> None:
        self._launch_workspace("validation", "omega_complete_app.py")

    def log(self, text: str) -> None:
        self.log_text.insert(END, text.rstrip() + "\n\n")
        self.log_text.see(END)


def main() -> None:
    root = Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except Exception:
        pass
    QuantLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
