from __future__ import annotations

import csv
import json
import threading
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, Canvas, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Callable

import numpy as np
import pandas as pd

from stock_model.age_structured import (
    AgeFitSettings,
    AgeProjectionSettings,
    AgeStructuredResult,
    AgeStructuredSettings,
    SectorSettings,
    equilibrium_reference_points,
    fit_age_structured,
    life_history_arrays,
    project_age_structured,
    read_age_structured_file,
    read_composition_file,
    run_management_strategy_evaluation,
    sector_curves,
    simulate_age_structured,
    synthetic_age_structured_dataset,
)
from stock_model.data_io import StockDataset


APP_TITLE = "Omega FISH Model — Integrated Assessment"
ROOT = Path(__file__).resolve().parent


class IntegratedAssessmentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1540x940")
        self.root.minsize(1180, 760)
        self.dataset: StockDataset | None = None
        self.age_composition: pd.DataFrame | None = None
        self.length_composition: pd.DataFrame | None = None
        self.result: AgeStructuredResult | None = None
        self.simulation: dict[str, Any] | None = None
        self.projection: dict[str, Any] | None = None
        self.mse: dict[str, Any] | None = None
        self.status = StringVar(value="Load a time-series dataset or generate the synthetic demonstration.")

        self.max_age = StringVar(value="30")
        self.natural_mortality = StringVar(value="0.12")
        self.r0 = StringVar(value="1000000")
        self.steepness = StringVar(value="0.75")
        self.initial_depletion = StringVar(value="0.85")
        self.recruitment_sigma = StringVar(value="0.60")
        self.linf = StringVar(value="850")
        self.growth_k = StringVar(value="0.13")
        self.maturity_a50 = StringVar(value="5.0")
        self.survey_a50 = StringVar(value="4.0")
        self.discard_mortality = StringVar(value="0.50")
        self.minimum_length = StringVar(value="500")
        self.fit_population = StringVar(value="36")
        self.fit_generations = StringVar(value="24")
        self.projection_years = StringVar(value="20")
        self.projection_iterations = StringVar(value="400")
        self.projection_strategy = StringVar(value="hcr_40_10")
        self.fixed_catch = StringVar(value="250")
        self.fixed_f = StringVar(value="0.08")
        self.pstar = StringVar(value="0.45")
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=TOP, fill=X)
        ttk.Button(toolbar, text="Load time series", command=self.load_dataset).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Load age composition", command=self.load_age_composition).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Load length composition", command=self.load_length_composition).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Synthetic demonstration", command=self.load_synthetic).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Export package", command=self.export_package).pack(side=LEFT, padx=12)
        ttk.Label(toolbar, textvariable=self.status).pack(side=LEFT, padx=12)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill=BOTH, expand=True)
        controls = ttk.Frame(body, padding=10, width=310)
        body.add(controls, weight=0)
        main = ttk.Frame(body, padding=6)
        body.add(main, weight=1)

        ttk.Label(controls, text="Age-structured model controls", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 7))
        control_canvas = Canvas(controls, highlightthickness=0, width=290)
        control_scroll = ttk.Scrollbar(controls, orient="vertical", command=control_canvas.yview)
        control_inner = ttk.Frame(control_canvas)
        control_inner.bind("<Configure>", lambda _event: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0, 0), window=control_inner, anchor="nw", width=280)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        control_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        control_scroll.pack(side=RIGHT, fill=Y)

        self._section(control_inner, "Population and recruitment")
        self._entry(control_inner, "Maximum age / plus group", self.max_age)
        self._entry(control_inner, "Natural mortality M", self.natural_mortality)
        self._entry(control_inner, "Unfished recruitment R0", self.r0)
        self._entry(control_inner, "Steepness h", self.steepness)
        self._entry(control_inner, "Initial depletion", self.initial_depletion)
        self._entry(control_inner, "Recruitment sigma", self.recruitment_sigma)

        self._section(control_inner, "Growth, maturity and survey")
        self._entry(control_inner, "Asymptotic length L∞ (mm)", self.linf)
        self._entry(control_inner, "Growth k", self.growth_k)
        self._entry(control_inner, "Maturity age 50%", self.maturity_a50)
        self._entry(control_inner, "Survey selectivity age 50%", self.survey_a50)

        self._section(control_inner, "Retention and post-release mortality")
        self._entry(control_inner, "Retention length 50% (mm)", self.minimum_length)
        self._entry(control_inner, "Discard / release mortality", self.discard_mortality)

        self._section(control_inner, "Estimation")
        self._entry(control_inner, "Optimizer population", self.fit_population)
        self._entry(control_inner, "Optimizer generations", self.fit_generations)
        ttk.Button(control_inner, text="Run deterministic reconstruction", command=self.run_simulation).pack(fill=X, pady=3)
        ttk.Button(control_inner, text="Fit integrated model", command=self.run_fit).pack(fill=X, pady=3)
        ttk.Button(control_inner, text="Calculate equilibrium reference points", command=self.run_reference_points).pack(fill=X, pady=3)

        self._section(control_inner, "Projection and strategy testing")
        self._entry(control_inner, "Projection years", self.projection_years)
        self._entry(control_inner, "Projection simulations", self.projection_iterations)
        ttk.Label(control_inner, text="Projection strategy").pack(anchor="w", pady=(5, 1))
        ttk.Combobox(
            control_inner,
            textvariable=self.projection_strategy,
            values=["hcr_40_10", "fixed_f", "fixed_catch"],
            state="readonly",
        ).pack(fill=X)
        self._entry(control_inner, "Fixed catch", self.fixed_catch)
        self._entry(control_inner, "Fixed F", self.fixed_f)
        self._entry(control_inner, "P*", self.pstar)
        ttk.Button(control_inner, text="Run stochastic projection", command=self.run_projection).pack(fill=X, pady=3)
        ttk.Button(control_inner, text="Run management strategy evaluation", command=self.run_mse).pack(fill=X, pady=3)

        ttk.Label(
            control_inner,
            text=(
                "Foundation scope: ages, growth, weight, maturity, Beverton–Holt recruitment, sector selectivity, "
                "retention, dead discards, Baranov catch reconstruction, age/length compositions and stochastic projections."
            ),
            wraplength=270,
            justify="left",
        ).pack(anchor="w", pady=12)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=BOTH, expand=True)
        self.data_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)
        self.age_tab = ttk.Frame(self.notebook)
        self.curves_tab = ttk.Frame(self.notebook)
        self.sector_tab = ttk.Frame(self.notebook)
        self.composition_tab = ttk.Frame(self.notebook)
        self.diagnostics_tab = ttk.Frame(self.notebook)
        self.projection_tab = ttk.Frame(self.notebook)
        self.mse_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        for tab, title in [
            (self.data_tab, "Data"),
            (self.history_tab, "Biomass and F"),
            (self.age_tab, "Age Structure"),
            (self.curves_tab, "Selectivity and Retention"),
            (self.sector_tab, "Sector Catch and Discards"),
            (self.composition_tab, "Composition"),
            (self.diagnostics_tab, "Fit Diagnostics"),
            (self.projection_tab, "Projection"),
            (self.mse_tab, "Strategy Evaluation"),
            (self.log_tab, "Log"),
        ]:
            self.notebook.add(tab, text=title)

        self.data_tree = self._tree(self.data_tab)
        self.history_canvas = Canvas(self.history_tab, background="white")
        self.history_canvas.pack(fill=BOTH, expand=True)
        self.history_canvas.bind("<Configure>", lambda _event: self.draw_history())
        self.age_canvas = Canvas(self.age_tab, background="white")
        self.age_canvas.pack(fill=BOTH, expand=True)
        self.age_canvas.bind("<Configure>", lambda _event: self.draw_age_heatmap())
        self.curves_canvas = Canvas(self.curves_tab, background="white")
        self.curves_canvas.pack(fill=BOTH, expand=True)
        self.curves_canvas.bind("<Configure>", lambda _event: self.draw_curves())
        self.sector_tree = self._tree(self.sector_tab)
        composition_pane = ttk.Panedwindow(self.composition_tab, orient="vertical")
        composition_pane.pack(fill=BOTH, expand=True)
        age_frame = ttk.LabelFrame(composition_pane, text="Predicted age composition")
        length_frame = ttk.LabelFrame(composition_pane, text="Predicted length composition")
        composition_pane.add(age_frame, weight=1)
        composition_pane.add(length_frame, weight=1)
        self.age_comp_tree = self._tree(age_frame)
        self.length_comp_tree = self._tree(length_frame)
        self.diagnostics_tree = self._tree(self.diagnostics_tab)
        projection_pane = ttk.Panedwindow(self.projection_tab, orient="vertical")
        projection_pane.pack(fill=BOTH, expand=True)
        projection_chart_frame = ttk.Frame(projection_pane)
        projection_table_frame = ttk.Frame(projection_pane)
        projection_pane.add(projection_chart_frame, weight=1)
        projection_pane.add(projection_table_frame, weight=1)
        self.projection_canvas = Canvas(projection_chart_frame, background="white")
        self.projection_canvas.pack(fill=BOTH, expand=True)
        self.projection_canvas.bind("<Configure>", lambda _event: self.draw_projection())
        self.projection_tree = self._tree(projection_table_frame)
        self.mse_tree = self._tree(self.mse_tab)
        self.log_text = __import__("tkinter").Text(self.log_tab, wrap="word", font=("Consolas", 10))
        self.log_text.pack(fill=BOTH, expand=True)

    @staticmethod
    def _section(parent, text: str) -> None:
        ttk.Separator(parent).pack(fill=X, pady=(10, 6))
        ttk.Label(parent, text=text, font=("Segoe UI", 10, "bold")).pack(anchor="w")

    @staticmethod
    def _entry(parent, label: str, variable: StringVar) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(5, 1))
        ttk.Entry(parent, textvariable=variable).pack(fill=X)

    @staticmethod
    def _tree(parent) -> ttk.Treeview:
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

    def _settings(self) -> AgeStructuredSettings:
        discard = min(max(float(self.discard_mortality.get()), 0.0), 1.0)
        retention = float(self.minimum_length.get())
        sectors = (
            SectorSettings("commercial", "catch_commercial", 0.50, 5.0, 1.2, retention, 35.0, discard, 1.0),
            SectorSettings("charter", "catch_charter", 0.15, 4.5, 1.3, retention, 35.0, discard, 1.0),
            SectorSettings("recreational", "catch_recreational", 0.35, 4.0, 1.4, retention, 35.0, discard, 1.0),
        )
        return AgeStructuredSettings(
            max_age=max(int(self.max_age.get()), 2),
            natural_mortality=max(float(self.natural_mortality.get()), 0.001),
            r0=max(float(self.r0.get()), 1.0),
            steepness=min(max(float(self.steepness.get()), 0.2001), 0.999),
            recruitment_sigma=max(float(self.recruitment_sigma.get()), 0.0),
            initial_depletion=min(max(float(self.initial_depletion.get()), 0.01), 1.50),
            linf_mm=max(float(self.linf.get()), 1.0),
            growth_k=max(float(self.growth_k.get()), 0.001),
            maturity_a50=float(self.maturity_a50.get()),
            survey_selectivity_a50=float(self.survey_a50.get()),
            sectors=sectors,
        )

    def _fit_settings(self) -> AgeFitSettings:
        return AgeFitSettings(
            population=max(int(self.fit_population.get()), 12),
            generations=max(int(self.fit_generations.get()), 1),
            seed=8301,
        )

    def _projection_settings(self) -> AgeProjectionSettings:
        return AgeProjectionSettings(
            years=max(int(self.projection_years.get()), 1),
            iterations=max(int(self.projection_iterations.get()), 20),
            strategy=self.projection_strategy.get(),
            fixed_catch=max(float(self.fixed_catch.get()), 0.0),
            fixed_f=max(float(self.fixed_f.get()), 0.0),
            pstar=max(float(self.pstar.get()), 0.0),
            seed=9331,
        )

    def _require_dataset(self) -> StockDataset:
        if self.dataset is None:
            raise ValueError("Load a time-series dataset first.")
        return self.dataset

    def _require_result(self) -> AgeStructuredResult:
        if self.result is None:
            raise ValueError("Fit the integrated model first.")
        return self.result

    def load_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load time-series data",
            filetypes=[("Data files", "*.csv *.xlsx *.xlsm"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.dataset = read_age_structured_file(path)
            self.result = None
            self.simulation = None
            self.projection = None
            self.mse = None
            self._populate_tree(self.data_tree, self.dataset.frame.where(self.dataset.frame.notna(), "").to_dict(orient="records"))
            self.status.set(f"Loaded {self.dataset.name}: {len(self.dataset.frame)} years.")
            self.notebook.select(self.data_tab)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def load_age_composition(self) -> None:
        path = filedialog.askopenfilename(title="Load age composition", filetypes=[("Data files", "*.csv *.xlsx *.xlsm"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.age_composition = read_composition_file(path)
            if "age" not in self.age_composition:
                raise ValueError("Selected composition file does not contain an age column.")
            self.status.set(f"Loaded {len(self.age_composition)} age-composition rows.")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def load_length_composition(self) -> None:
        path = filedialog.askopenfilename(title="Load length composition", filetypes=[("Data files", "*.csv *.xlsx *.xlsm"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.length_composition = read_composition_file(path)
            if "length_mm" not in self.length_composition:
                raise ValueError("Selected composition file does not contain a length_mm column.")
            self.status.set(f"Loaded {len(self.length_composition)} length-composition rows.")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def load_synthetic(self) -> None:
        try:
            settings = replace(self._settings(), max_age=min(max(int(self.max_age.get()), 8), 20), r0=max(float(self.r0.get()), 350_000.0))
            self.dataset, self.age_composition = synthetic_age_structured_dataset(30, settings, 1234)
            self.length_composition = None
            self.result = None
            self.simulation = None
            self.projection = None
            self.mse = None
            self._populate_tree(self.data_tree, self.dataset.frame.where(self.dataset.frame.notna(), "").to_dict(orient="records"))
            self.status.set("Synthetic age-structured demonstration loaded with age compositions.")
            self.notebook.select(self.data_tab)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def _run_background(self, label: str, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        self.status.set(label)

        def target() -> None:
            try:
                result = work()
                self.root.after(0, lambda: self._complete(done, result, label))
            except Exception:
                trace = traceback.format_exc()
                self.root.after(0, lambda: self._failed(label, trace))

        threading.Thread(target=target, daemon=True).start()

    def _complete(self, done: Callable[[Any], None], result: Any, label: str) -> None:
        done(result)
        self.status.set(label.replace("Running", "Completed"))

    def _failed(self, label: str, trace: str) -> None:
        self.log(trace)
        self.status.set(f"Failed: {label}")
        messagebox.showerror(APP_TITLE, trace.splitlines()[-1] if trace.splitlines() else trace)

    def run_simulation(self) -> None:
        try:
            dataset = self._require_dataset()
            settings = self._settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background("Running deterministic reconstruction...", lambda: simulate_age_structured(dataset, settings), self._show_simulation)

    def _show_simulation(self, output: dict[str, Any]) -> None:
        self.simulation = output
        self.result = None
        self._populate_tree(self.sector_tree, output["sector_history"])
        self._populate_tree(self.age_comp_tree, output["predicted_age_composition"][:2000])
        self._populate_tree(self.length_comp_tree, [])
        self.draw_history()
        self.draw_age_heatmap()
        self.draw_curves()
        self.notebook.select(self.history_tab)
        self.log(json.dumps({"simulation_summary": {"b0": output["b0"], "catch_mismatch_total": output["catch_mismatch_total"]}}, indent=2))

    def run_fit(self) -> None:
        try:
            dataset = self._require_dataset()
            settings = self._settings()
            fit_settings = self._fit_settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background(
            "Running integrated age-structured fit...",
            lambda: fit_age_structured(dataset, settings, fit_settings, self.age_composition, self.length_composition),
            self._show_fit,
        )

    def _show_fit(self, result: AgeStructuredResult) -> None:
        self.result = result
        self.simulation = {
            "settings": result.settings,
            "history": result.history,
            "sector_history": result.sector_history,
            "age_structure": result.age_structure,
            "predicted_age_composition": result.predicted_age_composition,
            "predicted_length_composition": result.predicted_length_composition,
            "life_history": result.state["life_history"],
            "sector_curves": result.state["sector_curves"],
        }
        diagnostic_rows = [{"component": key, "value": value} for key, value in result.diagnostics["objective_components"].items()]
        diagnostic_rows.extend({"component": key, "value": value} for key, value in result.best.items())
        self._populate_tree(self.diagnostics_tree, diagnostic_rows)
        self._populate_tree(self.sector_tree, result.sector_history)
        self._populate_tree(self.age_comp_tree, result.predicted_age_composition[:2500])
        self._populate_tree(self.length_comp_tree, result.predicted_length_composition[:2500])
        self.draw_history()
        self.draw_age_heatmap()
        self.draw_curves()
        self.notebook.select(self.diagnostics_tab)
        self.log(json.dumps({"best": result.best, "diagnostics": result.diagnostics}, indent=2, default=str))

    def run_reference_points(self) -> None:
        try:
            settings = self._settings() if self.result is None else self._settings_from_result()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background("Running equilibrium reference-point grid...", lambda: equilibrium_reference_points(settings), self._show_reference_points)

    def _settings_from_result(self) -> AgeStructuredSettings:
        result = self._require_result()
        value = dict(result.settings)
        sectors = tuple(SectorSettings(**row) for row in value.pop("sectors"))
        return AgeStructuredSettings(sectors=sectors, **value)

    def _show_reference_points(self, output: dict[str, Any]) -> None:
        rows = [{"metric": key, "value": value} for key, value in output.items() if key != "grid"]
        self._populate_tree(self.diagnostics_tree, rows)
        self.notebook.select(self.diagnostics_tab)
        self.log(json.dumps({"reference_points": {key: value for key, value in output.items() if key != "grid"}}, indent=2))

    def run_projection(self) -> None:
        try:
            result = self._require_result()
            settings = self._projection_settings()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background("Running stochastic age-structured projection...", lambda: project_age_structured(result, settings), self._show_projection)

    def _show_projection(self, output: dict[str, Any]) -> None:
        self.projection = output
        self._populate_tree(self.projection_tree, output["projection"])
        self.draw_projection()
        self.notebook.select(self.projection_tab)
        self.log(json.dumps({"projection_risk": output["risk_summary"]}, indent=2))

    def run_mse(self) -> None:
        try:
            result = self._require_result()
            years = max(int(self.projection_years.get()), 1)
            iterations = max(int(self.projection_iterations.get()) // 2, 80)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        self._run_background("Running management strategy evaluation...", lambda: run_management_strategy_evaluation(result, years, iterations), self._show_mse)

    def _show_mse(self, output: dict[str, Any]) -> None:
        self.mse = output
        rows = []
        pareto_names = {row["strategy"] for row in output["pareto_front"]}
        for row in output["strategies"]:
            rows.append({**row, "pareto": row["strategy"] in pareto_names})
        self._populate_tree(self.mse_tree, rows)
        self.notebook.select(self.mse_tab)
        self.log(json.dumps({"mse_summary": output["summary"], "pareto_front": output["pareto_front"]}, indent=2))

    def _history_rows(self) -> list[dict[str, Any]]:
        if self.result is not None:
            return self.result.history
        if self.simulation is not None:
            return self.simulation.get("history", [])
        return []

    def draw_history(self) -> None:
        rows = self._history_rows()
        canvas = self.history_canvas
        canvas.delete("all")
        if not rows:
            canvas.create_text(20, 20, anchor="nw", text="Run a reconstruction or fit to display biomass, depletion and fishing mortality.")
            return
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        margin = 55
        years = np.array([row["year"] for row in rows], dtype=float)
        series = [
            ("depletion", "Depletion", "#136f63"),
            ("f_scalar", "Fishing mortality", "#9b2d21"),
        ]
        canvas.create_text(margin, 18, anchor="w", text="Stock status and fishing mortality", font=("Segoe UI", 12, "bold"))
        x = margin + (years - years.min()) / max(years.max() - years.min(), 1.0) * (width - 2 * margin)
        for key, label, colour in series:
            values = np.array([row[key] for row in rows], dtype=float)
            maximum = max(float(np.nanmax(values)), 1e-6)
            y = height - margin - values / maximum * (height - 2 * margin)
            points = [coordinate for pair in zip(x, y) for coordinate in pair]
            if len(points) >= 4:
                canvas.create_line(*points, fill=colour, width=2)
            canvas.create_text(width - margin - 150, 26 + 18 * series.index((key, label, colour)), anchor="w", text=f"{label} (scaled independently)", fill=colour)
        canvas.create_line(margin, height - margin, width - margin, height - margin, fill="#555")
        canvas.create_text(margin, height - 25, anchor="w", text=str(int(years.min())))
        canvas.create_text(width - margin, height - 25, anchor="e", text=str(int(years.max())))

    def draw_age_heatmap(self) -> None:
        canvas = self.age_canvas
        canvas.delete("all")
        rows = self.result.age_structure if self.result is not None else (self.simulation or {}).get("age_structure", [])
        if not rows:
            canvas.create_text(20, 20, anchor="nw", text="Age structure appears after a reconstruction or fit.")
            return
        years = sorted({int(row["year"]) for row in rows})
        ages = sorted({int(row["age"]) for row in rows})
        values = {(int(row["year"]), int(row["age"])): max(float(row["numbers"]), 0.0) for row in rows}
        logs = np.array([log_value for value in values.values() if value > 0 for log_value in [np.log10(value)]])
        low = float(np.min(logs)) if len(logs) else 0.0
        high = float(np.max(logs)) if len(logs) else 1.0
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 350)
        margin_x, margin_y = 70, 45
        cell_w = max((width - 2 * margin_x) / max(len(years), 1), 1.0)
        cell_h = max((height - 2 * margin_y) / max(len(ages), 1), 1.0)
        for ix, year in enumerate(years):
            for iy, age in enumerate(ages):
                value = values.get((year, age), 0.0)
                scaled = 0.0 if value <= 0 else (np.log10(value) - low) / max(high - low, 1e-9)
                red = int(245 - 155 * scaled)
                green = int(248 - 65 * scaled)
                blue = int(250 - 190 * scaled)
                colour = f"#{red:02x}{green:02x}{blue:02x}"
                x0 = margin_x + ix * cell_w
                y0 = margin_y + (len(ages) - 1 - iy) * cell_h
                canvas.create_rectangle(x0, y0, x0 + cell_w + 1, y0 + cell_h + 1, fill=colour, outline="")
        canvas.create_text(margin_x, 18, anchor="w", text="Numbers-at-age heatmap (log scale)", font=("Segoe UI", 12, "bold"))
        canvas.create_text(20, margin_y, anchor="nw", text=f"Age {max(ages)}")
        canvas.create_text(20, height - margin_y, anchor="sw", text="Age 0")
        canvas.create_text(margin_x, height - 18, anchor="w", text=str(min(years)))
        canvas.create_text(width - margin_x, height - 18, anchor="e", text=str(max(years)))

    def draw_curves(self) -> None:
        canvas = self.curves_canvas
        canvas.delete("all")
        try:
            settings = self._settings_from_result() if self.result is not None else self._settings()
        except Exception:
            return
        life = life_history_arrays(settings)
        curves = sector_curves(settings, life)
        ages = life["age"]
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 350)
        margin = 55
        x = margin + ages / max(float(ages.max()), 1.0) * (width - 2 * margin)
        colours = ["#136f63", "#275b85", "#a66a1f"]
        canvas.create_text(margin, 18, anchor="w", text="Sector selectivity and retention-at-age", font=("Segoe UI", 12, "bold"))
        for sector_index, sector in enumerate(settings.sectors):
            colour = colours[sector_index % len(colours)]
            selectivity = curves[sector.name]["selectivity"]
            retention = curves[sector.name]["retention"]
            y_sel = height - margin - selectivity * (height - 2 * margin)
            y_ret = height - margin - retention * (height - 2 * margin)
            canvas.create_line(*[coordinate for pair in zip(x, y_sel) for coordinate in pair], fill=colour, width=2)
            canvas.create_line(*[coordinate for pair in zip(x, y_ret) for coordinate in pair], fill=colour, width=1, dash=(4, 3))
            canvas.create_text(width - margin - 180, 28 + sector_index * 18, anchor="w", text=f"{sector.name}: solid selectivity, dashed retention", fill=colour)
        canvas.create_line(margin, height - margin, width - margin, height - margin, fill="#555")
        canvas.create_line(margin, margin, margin, height - margin, fill="#555")
        canvas.create_text(margin, height - 22, anchor="w", text="Age 0")
        canvas.create_text(width - margin, height - 22, anchor="e", text=f"Age {int(ages.max())}")

    def draw_projection(self) -> None:
        canvas = self.projection_canvas
        canvas.delete("all")
        rows = (self.projection or {}).get("projection", [])
        if not rows:
            canvas.create_text(20, 20, anchor="nw", text="Run a stochastic projection to display depletion uncertainty.")
            return
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 280)
        margin = 50
        years = np.array([row["year"] for row in rows], dtype=float)
        x = margin + (years - years.min()) / max(years.max() - years.min(), 1.0) * (width - 2 * margin)
        low = np.array([row["depletion_p10"] for row in rows])
        median = np.array([row["depletion_median"] for row in rows])
        high = np.array([row["depletion_p90"] for row in rows])
        ymax = max(float(np.max(high)), 0.5)
        scale = lambda values: height - margin - np.asarray(values) / ymax * (height - 2 * margin)
        polygon = list(zip(x, scale(low))) + list(zip(x[::-1], scale(high[::-1])))
        canvas.create_polygon(*[coordinate for pair in polygon for coordinate in pair], fill="#dbe9e4", outline="")
        canvas.create_line(*[coordinate for pair in zip(x, scale(median)) for coordinate in pair], fill="#136f63", width=2)
        for level, colour, label in [(0.40, "#a66a1f", "Target 0.40"), (0.10, "#9b2d21", "Limit 0.10")]:
            y = float(scale([level])[0])
            canvas.create_line(margin, y, width - margin, y, fill=colour, dash=(5, 3))
            canvas.create_text(width - margin, y - 3, anchor="se", text=label, fill=colour)
        canvas.create_text(margin, 18, anchor="w", text="Projected spawning-biomass depletion (P10–P90)", font=("Segoe UI", 12, "bold"))

    def _populate_tree(self, tree: ttk.Treeview, rows: list[dict[str, Any]]) -> None:
        tree.delete(*tree.get_children())
        if not rows:
            tree["columns"] = []
            return
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns and not isinstance(row[key], (dict, list, tuple, np.ndarray)):
                    columns.append(key)
        columns = columns[:30]
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            tree.column(column, width=max(90, min(190, len(column) * 10)), anchor="w")
        for row in rows[:3000]:
            values = [self._format(row.get(column)) for column in columns]
            tree.insert("", END, values=values)

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, float):
            return "" if not np.isfinite(value) else f"{value:.6g}"
        return "" if value is None else str(value)

    def export_package(self) -> None:
        if self.result is None and self.simulation is None:
            messagebox.showinfo(APP_TITLE, "Run a reconstruction or fit before exporting.")
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        target = Path(folder) / "Omega_FISH_Integrated_Assessment"
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "application": APP_TITLE,
            "dataset": self.dataset.name if self.dataset else "",
            "settings": asdict(self._settings()),
            "result": self.result.__dict__ if self.result else {},
            "simulation": self.simulation or {},
            "projection": self.projection or {},
            "management_strategy_evaluation": self.mse or {},
        }
        (target / "integrated_assessment.json").write_text(json.dumps(payload, indent=2, default=self._json_default), encoding="utf-8")
        tables = {
            "history.csv": self.result.history if self.result else (self.simulation or {}).get("history", []),
            "sector_history.csv": self.result.sector_history if self.result else (self.simulation or {}).get("sector_history", []),
            "age_structure.csv": self.result.age_structure if self.result else (self.simulation or {}).get("age_structure", []),
            "predicted_age_composition.csv": self.result.predicted_age_composition if self.result else (self.simulation or {}).get("predicted_age_composition", []),
            "predicted_length_composition.csv": self.result.predicted_length_composition if self.result else (self.simulation or {}).get("predicted_length_composition", []),
            "projection.csv": (self.projection or {}).get("projection", []),
            "strategy_evaluation.csv": (self.mse or {}).get("strategies", []),
        }
        for name, rows in tables.items():
            self._write_csv(target / name, rows)
        report = self._html_report(payload)
        (target / "INTEGRATED_ASSESSMENT_REPORT.html").write_text(report, encoding="utf-8")
        self.status.set(f"Exported integrated assessment package to {target}")
        messagebox.showinfo(APP_TITLE, f"Export complete.\n\n{target}")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = []
        for row in rows:
            for key, value in row.items():
                if key not in fields and not isinstance(value, (dict, list, tuple, np.ndarray)):
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    def _html_report(self, payload: dict[str, Any]) -> str:
        best = payload.get("result", {}).get("best", {}) if isinstance(payload.get("result"), dict) else {}
        projection_risk = payload.get("projection", {}).get("risk_summary", {}) if isinstance(payload.get("projection"), dict) else {}
        rows = "".join(f"<tr><th>{key}</th><td>{self._format(value)}</td></tr>" for key, value in best.items())
        risk_rows = "".join(f"<tr><th>{key}</th><td>{self._format(value)}</td></tr>" for key, value in projection_risk.items())
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{APP_TITLE}</title>
<style>body{{font-family:Arial;max-width:1100px;margin:30px auto;color:#172126}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ccd6d8;padding:7px;text-align:left}}th{{background:#eef3ef}}.note{{border-left:4px solid #a66a1f;padding:10px;background:#faf7ef}}</style></head>
<body><h1>{APP_TITLE}</h1><p>Dataset: {payload.get('dataset','')}</p>
<div class='note'>This release is an integrated age-structured foundation. It is designed for transparent testing and development and is not automatically equivalent to a completed peer-reviewed Stock Synthesis assessment.</div>
<h2>Best fit</h2><table>{rows}</table><h2>Projection risk</h2><table>{risk_rows}</table>
<h2>Implemented processes</h2><p>Ages and plus group; von Bertalanffy growth; weight-at-age; maturity; Beverton–Holt recruitment; sector selectivity; retention; discard mortality; Baranov catch equations; age and length compositions; equilibrium reference points; stochastic projections; management strategy evaluation.</p>
</body></html>"""

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        return str(value)

    def log(self, text: str) -> None:
        self.log_text.insert(END, text.rstrip() + "\n\n")
        self.log_text.see(END)


def main() -> None:
    root = Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except Exception:
        pass
    IntegratedAssessmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
