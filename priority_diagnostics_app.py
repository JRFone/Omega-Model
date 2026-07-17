from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, BooleanVar, Canvas, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Callable, Mapping

import pandas as pd

from stock_model.aspm_diagnostic import ASPMSettings, run_age_structured_aspm
from stock_model.core import ModelSettings, fit
from stock_model.data_io import read_stock_file
from stock_model.interactive_charts import ChartProfile, InteractiveChartFactory, SeriesSpec
from stock_model.interval_coverage import CoverageSettings, run_interval_coverage
from stock_model.likelihood_profiles import ProfileSettings, profile_likelihood
from stock_model.native_backend import native_status
from stock_model.native_benchmark import NativeBenchmarkSettings, write_native_benchmark
from chart_studio_app import NativeChartPreview
from ui.model_health import assess_model_health

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


class WrappedHealthTable(ttk.Frame):
    COLUMNS = (
        ("test", "Test", 1, 170),
        ("quick_verdict", "Quick verdict", 1, 180),
        ("accuracy_evidence", "Accuracy evidence", 2, 240),
        ("confounding_risk", "Confounding risk", 2, 240),
        ("reason", "Why", 2, 280),
        ("next_action", "Next action", 2, 280),
    )

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.canvas = Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollbar.pack(side=RIGHT, fill="y")
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.labels: list[tuple[object, int]] = []
        self.row_index = 1
        for column, (_key, title, weight, wrap) in enumerate(self.COLUMNS):
            self.inner.columnconfigure(column, weight=weight, uniform="health")
            label = ttk.Label(self.inner, text=title, style="HealthHeader.TLabel", anchor="center", padding=(6, 7))
            label.grid(row=0, column=column, sticky="nsew", padx=1, pady=1)
            self.labels.append((label, wrap))
        self.inner.bind("<Configure>", self._sync)
        self.canvas.bind("<Configure>", self._sync)

    def add_row(self, row: Mapping[str, str], severity: str) -> None:
        style = {"good": "HealthGood.TLabel", "warn": "HealthWarn.TLabel", "bad": "HealthBad.TLabel"}[severity]
        for column, (key, _title, _weight, wrap) in enumerate(self.COLUMNS):
            label = ttk.Label(
                self.inner,
                text=row.get(key, ""),
                style=style,
                anchor="nw",
                justify="left",
                wraplength=wrap,
                padding=(7, 7),
            )
            label.grid(row=self.row_index, column=column, sticky="nsew", padx=1, pady=1)
            label.bind("<MouseWheel>", self._wheel)
            self.labels.append((label, wrap))
        self.row_index += 1
        self._sync()

    def _sync(self, _event=None) -> None:
        try:
            width = max(600, self.canvas.winfo_width())
            self.canvas.itemconfigure(self.window, width=width)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            base = max(80, (width - 20) // 10)
            for label, original in self.labels:
                label.configure(wraplength=max(80, min(original, base * (2 if original >= 220 else 1))))
        except Exception:
            pass

    def _wheel(self, event):
        direction = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(direction * 3, "units")
        return "break"


class PriorityDiagnosticsApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.geometry("1320x850")
        self.root.minsize(1050, 700)
        self.dataset_path = StringVar()
        self.age_path = StringVar()
        self.length_path = StringVar()
        self.parameter = StringVar(value="initial_depletion")
        self.analysis_level = StringVar(value="standard")
        self.profile_points = IntVar(value=21)
        self.profile_multistarts = IntVar(value=3)
        self.workers = IntVar(value=max(1, min(8, (os.cpu_count() or 2) - 1)))
        self.coverage_replicates = IntVar(value=100)
        self.coverage_hessian = BooleanVar(value=True)
        self.coverage_profile = BooleanVar(value=False)
        self.coverage_bootstrap = BooleanVar(value=False)
        self.status = StringVar(value="Ready. Native acceleration status is shown below.")
        self.last_result: Mapping[str, Any] | None = None
        self.last_json: Path | None = None
        self.last_dashboard: Path | None = None
        self.health_rows: list[dict[str, str]] = []
        self._configure_style()
        self._build()
        self.refresh_native_status()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("PDHeader.TFrame", background="#0b1f33")
        style.configure("PDTitle.TLabel", background="#0b1f33", foreground="white", font=("Segoe UI", 22, "bold"))
        style.configure("PDSub.TLabel", background="#0b1f33", foreground="#c4d2df", font=("Segoe UI", 10))
        style.configure("PDStatus.TLabel", background="#e8eef5", foreground="#334155", padding=(10, 7))
        style.configure("HealthHeader.TLabel", background="#102a43", foreground="#ffffff", font=("Segoe UI", 9, "bold"))
        style.configure("HealthGood.TLabel", background="#dcfce7", foreground="#14532d")
        style.configure("HealthWarn.TLabel", background="#fef3c7", foreground="#78350f")
        style.configure("HealthBad.TLabel", background="#fee2e2", foreground="#7f1d1d")

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(22, 18), style="PDHeader.TFrame")
        header.pack(side=TOP, fill=X)
        ttk.Label(header, text="Omega Priority Diagnostics 1.3", style="PDTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Compiled C++ engine, fully refitted likelihood profiles, genuine age-structured ASPM and formal interval coverage",
            style="PDSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        source = ttk.LabelFrame(self.root, text="Assessment inputs", padding=10)
        source.pack(fill=X, padx=14, pady=(12, 6))
        preset = ttk.Frame(source)
        preset.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Label(preset, text="Analysis level").pack(side=LEFT)
        level = ttk.Combobox(preset, textvariable=self.analysis_level, values=("quick", "standard", "formal"), state="readonly", width=12)
        level.pack(side=LEFT, padx=8)
        level.bind("<<ComboboxSelected>>", lambda _event: self.apply_analysis_level())
        ttk.Label(preset, text="Quick is for inspection; formal selects larger profile and coverage workloads.").pack(side=LEFT, padx=8)
        self._file_row(source, 1, "Model-ready dataset", self.dataset_path, [("Data files", "*.csv *.xlsx *.xlsm"), ("All files", "*.*")])
        self._file_row(source, 2, "Age composition (optional)", self.age_path, [("CSV", "*.csv"), ("All files", "*.*")])
        self._file_row(source, 3, "Length composition (optional)", self.length_path, [("CSV", "*.csv"), ("All files", "*.*")])

        notebook = ttk.Notebook(self.root)
        self.notebook = notebook
        notebook.pack(fill=BOTH, expand=True, padx=14, pady=6)
        engine_tab = ttk.Frame(notebook, padding=12)
        profile_tab = ttk.Frame(notebook, padding=12)
        aspm_tab = ttk.Frame(notebook, padding=12)
        coverage_tab = ttk.Frame(notebook, padding=12)
        charts_tab = ttk.Frame(notebook, padding=8)
        health_tab = ttk.Frame(notebook, padding=12)
        self.charts_tab = charts_tab
        self.health_tab = health_tab
        results_tab = ttk.Frame(notebook, padding=12)
        notebook.add(engine_tab, text="Native engine")
        notebook.add(profile_tab, text="Likelihood profiles")
        notebook.add(aspm_tab, text="Age-structured ASPM")
        notebook.add(coverage_tab, text="Interval coverage")
        notebook.add(charts_tab, text="Diagnostic charts")
        notebook.add(health_tab, text="Quick model health")
        notebook.add(results_tab, text="Results and evidence")

        engine_controls = ttk.Frame(engine_tab)
        engine_controls.pack(fill=X)
        ttk.Button(engine_controls, text="Refresh engine status", command=self.refresh_native_status).pack(side=LEFT)
        ttk.Button(engine_controls, text="Build and test C++ backend", command=self.build_native).pack(side=LEFT, padx=6)
        ttk.Button(engine_controls, text="Run speed/parity benchmark", command=self.run_native_benchmark).pack(side=LEFT, padx=6)
        ttk.Label(engine_controls, text="The Python fallback remains available, but accelerated status is recorded in every run.").pack(side=LEFT, padx=12)
        self.engine_text = self._text(engine_tab)

        row = ttk.Frame(profile_tab)
        row.pack(fill=X, pady=6)
        ttk.Label(row, text="Profile parameter").pack(side=LEFT)
        ttk.Combobox(row, textvariable=self.parameter, values=("k", "r", "initial_depletion", "sigma"), state="readonly", width=22).pack(side=LEFT, padx=8)
        ttk.Label(row, text="Points").pack(side=LEFT)
        ttk.Spinbox(row, from_=7, to=61, textvariable=self.profile_points, width=6).pack(side=LEFT, padx=6)
        ttk.Label(row, text="Multistarts per point").pack(side=LEFT)
        ttk.Spinbox(row, from_=1, to=12, textvariable=self.profile_multistarts, width=6).pack(side=LEFT, padx=6)
        ttk.Label(row, text="Workers").pack(side=LEFT)
        ttk.Spinbox(row, from_=1, to=64, textvariable=self.workers, width=6).pack(side=LEFT, padx=6)
        ttk.Button(profile_tab, text="Run fully refitted profile", command=self.run_profile).pack(anchor="w", pady=8)
        ttk.Label(
            profile_tab,
            text="Every point fixes one parameter and re-optimises all remaining active parameters. Failed or non-stationary points remain visible.",
            wraplength=1000,
        ).pack(anchor="w")

        ttk.Label(
            aspm_tab,
            text="Retains ages, natural mortality, growth, maturity, weight-at-age, selectivity, retention and catch history while removing composition likelihoods.",
            wraplength=1000,
        ).pack(anchor="w", pady=6)
        aspm_controls = ttk.Frame(aspm_tab)
        aspm_controls.pack(fill=X, pady=8)
        ttk.Label(aspm_controls, text="Multistarts").pack(side=LEFT)
        ttk.Spinbox(aspm_controls, from_=1, to=12, textvariable=self.profile_multistarts, width=6).pack(side=LEFT, padx=6)
        ttk.Button(aspm_controls, text="Run ASPM, ASPM-R and index influence", command=self.run_aspm).pack(side=LEFT, padx=10)

        methods = ttk.LabelFrame(coverage_tab, text="Uncertainty methods", padding=10)
        methods.pack(fill=X, pady=6)
        ttk.Checkbutton(methods, text="Hessian / delta method", variable=self.coverage_hessian).pack(side=LEFT)
        ttk.Checkbutton(methods, text="Profile likelihood", variable=self.coverage_profile).pack(side=LEFT, padx=12)
        ttk.Checkbutton(methods, text="Parametric bootstrap", variable=self.coverage_bootstrap).pack(side=LEFT)
        coverage_controls = ttk.Frame(coverage_tab)
        coverage_controls.pack(fill=X, pady=10)
        ttk.Label(coverage_controls, text="Known-truth outer replicates").pack(side=LEFT)
        ttk.Spinbox(coverage_controls, from_=2, to=10000, textvariable=self.coverage_replicates, width=8).pack(side=LEFT, padx=6)
        ttk.Label(coverage_controls, text="Workers").pack(side=LEFT)
        ttk.Spinbox(coverage_controls, from_=1, to=64, textvariable=self.workers, width=6).pack(side=LEFT, padx=6)
        ttk.Button(coverage_controls, text="Run formal coverage test", command=self.run_coverage).pack(side=LEFT, padx=10)
        ttk.Label(
            coverage_tab,
            text="Failed fits and incomplete intervals count against coverage. Standard formal runs should use at least 500–1,000 outer replicates.",
            wraplength=1000,
        ).pack(anchor="w")

        ttk.Label(
            health_tab,
            text=(
                "Quick visual interpretation of completed tests. 'Accurate' is used only for known-truth recovery; "
                "a software PASS alone never establishes scientific accuracy."
            ),
            wraplength=1050,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        self.health_tree = self._health_tree(health_tab)

        self.chart_note = StringVar(value="Run a diagnostic to display its chart here.")
        ttk.Label(charts_tab, textvariable=self.chart_note, wraplength=1050, justify="left").pack(anchor="w", fill=X, pady=(0, 7))
        self.diagnostic_chart = NativeChartPreview(charts_tab)
        self.diagnostic_chart.pack(fill=BOTH, expand=True)
        self.diagnostic_chart.show_message("Run a likelihood profile, ASPM comparison, coverage test, or native benchmark to see its chart.")

        result_buttons = ttk.Frame(results_tab)
        result_buttons.pack(fill=X)
        ttk.Button(result_buttons, text="Open JSON evidence", command=self.open_json).pack(side=LEFT)
        ttk.Button(result_buttons, text="Open interactive dashboard", command=self.open_dashboard).pack(side=LEFT, padx=6)
        ttk.Button(result_buttons, text="Open reports folder", command=self.open_reports).pack(side=LEFT)
        self.results_text = self._text(results_tab)

        ttk.Label(self.root, textvariable=self.status, style="PDStatus.TLabel").pack(fill=X, padx=14, pady=(4, 12))

    def apply_analysis_level(self) -> None:
        level = self.analysis_level.get()
        if level == "quick":
            self.profile_points.set(9)
            self.profile_multistarts.set(2)
            self.coverage_replicates.set(20)
        elif level == "formal":
            self.profile_points.set(31)
            self.profile_multistarts.set(5)
            self.coverage_replicates.set(1000)
        else:
            self.profile_points.set(21)
            self.profile_multistarts.set(3)
            self.coverage_replicates.set(100)
        self.status.set(f"Applied {level} workload preset. All values remain editable.")

    def _file_row(self, parent, row: int, label: str, variable: StringVar, filetypes) -> None:
        ttk.Label(parent, text=label, width=28).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, filetypes)).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _text(parent):
        from tkinter import Text

        text = Text(parent, wrap="none", font=("Consolas", 9))
        text.pack(fill=BOTH, expand=True, pady=(10, 0))
        return text

    @staticmethod
    def _health_tree(parent):
        table = WrappedHealthTable(parent)
        table.pack(fill=BOTH, expand=True)
        return table

    def _browse(self, variable: StringVar, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def _dataset(self):
        path = Path(self.dataset_path.get()).expanduser()
        if not path.exists():
            raise FileNotFoundError("Select a model-ready dataset first.")
        return read_stock_file(path)

    def _run_background(self, label: str, task: Callable[[], tuple[Mapping[str, Any], Path, Path | None]]) -> None:
        self.status.set(label)
        outcome: Queue[tuple[str, Any]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                result, json_path, dashboard = task()
                self.last_result = result
                self.last_json = json_path
                self.last_dashboard = dashboard
                summary = result.get("summary") or {}
                payload = json.dumps({"summary": summary, "json": str(json_path), "dashboard": str(dashboard) if dashboard else None}, indent=2, default=str)
                health = assess_model_health(label, result)
                outcome.put(("ok", (payload, health, result, summary, json_path)))
            except Exception as exc:
                outcome.put(("error", (exc, traceback.format_exc())))

        def poll() -> None:
            try:
                state, value = outcome.get_nowait()
            except Empty:
                try:
                    if self.root.winfo_exists():
                        self.root.after(35, poll)
                except Exception:
                    pass
                return
            if state == "ok":
                payload, health, result, summary, json_path = value
                self._show_result(payload, health, result)
                self.status.set(f"Completed: {summary.get('status', 'finished')}. Evidence saved to {json_path}.")
            else:
                exc, detail = value
                self.status.set("Run failed. The error remains visible.")
                shell = getattr(self.root, "omega_shell", None)
                if shell is not None:
                    shell.log_error("Priority Diagnostics", exc, detail)
                messagebox.showerror("Omega Priority Diagnostics", f"{type(exc).__name__}: {exc}")

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(35, poll)

    def _show_result(
        self,
        text: str,
        health: Mapping[str, str] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self.results_text.delete("1.0", END)
        self.results_text.insert(END, text)
        chart_ready = False
        if result is not None:
            try:
                figure, x_title, y_title, note = self._diagnostic_figure(result, health)
                self.diagnostic_chart.show_figure(figure, health.get("test", "Diagnostic chart") if health else "Diagnostic chart", x_title, y_title)
                self.chart_note.set(note)
                chart_ready = True
            except Exception as exc:
                self.diagnostic_chart.show_message(f"The diagnostic completed, but its chart could not be built:\n{exc}")
                self.chart_note.set("The numerical result remains available in Results and evidence.")
        if health is not None:
            self.health_rows.append(dict(health))
            verdict = health.get("quick_verdict", "")
            tag = "bad" if any(word in verdict for word in ("FAILED", "INACCURATE", "SENSITIVE")) else "warn" if any(word in verdict for word in ("WARNING", "POSSIBLE", "QUESTIONABLE", "REVIEW", "WEAK")) else "good"
            self.health_tree.add_row(health, tag)
            self.notebook.select(self.charts_tab if chart_ready else self.health_tab)

    def _diagnostic_figure(
        self,
        result: Mapping[str, Any],
        health: Mapping[str, str] | None,
    ) -> tuple[Any, str, str, str]:
        factory = InteractiveChartFactory(ChartProfile(range_slider=False, default_height=600))
        title = health.get("test", "Diagnostic chart") if health else "Diagnostic chart"
        profile = result.get("profile") or []
        if profile:
            figure = factory.likelihood_profile(
                profile,
                parameter_key="fixed_value",
                objective_key="delta_nll",
                title=title,
                parameter_label=str((result.get("summary") or {}).get("parameter", "Parameter value")),
            )
            return figure, "Fixed parameter value", "Delta objective", "A flat curve or missing threshold crossing indicates weak parameter identifiability or confounding."
        coverage = result.get("coverage") or []
        if coverage:
            figure = factory.interval_coverage(
                coverage,
                nominal_key="nominal",
                empirical_key="empirical",
                parameter_key="parameter",
                title=title,
            )
            return figure, "Nominal interval coverage", "Observed coverage", "Points should remain close to the 1:1 calibration line; failed and incomplete intervals remain counted."
        variants = result.get("variants") or []
        full_model = result.get("full_model") or {}
        full_history = full_model.get("history") or []
        if full_history or any(row.get("history") for row in variants):
            series: list[SeriesSpec] = []
            if full_history:
                series.append(SeriesSpec("Full age-structured model", [row["year"] for row in full_history], [row["depletion"] for row in full_history], mode="lines+markers"))
            for variant in variants:
                history = variant.get("history") or []
                if history:
                    series.append(SeriesSpec(str(variant.get("name", "ASPM variant")), [row["year"] for row in history], [row["depletion"] for row in history], mode="lines"))
            figure = factory.time_series(series, title=title, x_title="Year", y_title="Relative spawning biomass / depletion")
            return figure, "Year", "Relative biomass", "Divergence between the full fit and ASPM variants shows dependence on compositions, recruitment, indices, or structure."
        summary = result.get("summary") or result
        numeric = []
        for key, value in summary.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)) and not isinstance(value, bool):
                numeric.append({"component": str(key).replace("_", " "), "value": float(value)})
        if numeric:
            figure = factory.likelihood_conflict(numeric[:16], component_key="component", value_key="value", preferred_key=None, title=title)
            return figure, "Recorded value", "Diagnostic measure", "Summary measures from the completed diagnostic. Use the evidence tab for units, thresholds, and full precision."
        raise ValueError("No numeric or trajectory values were available for this diagnostic.")

    def refresh_native_status(self) -> None:
        status = native_status()
        self.engine_text.delete("1.0", END)
        self.engine_text.insert(END, json.dumps(status, indent=2))
        self.status.set("C++ native backend available." if status.get("available") else "Native backend unavailable; Python fallback is active.")

    def build_native(self) -> None:
        self.status.set("Building and testing the C++ backend...")

        def worker() -> None:
            try:
                command = [sys.executable, str(Path(__file__).resolve().parent / "build_native_backend.py"), "--clean"]
                completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True, check=False)
                detail = (completed.stdout + "\n" + completed.stderr).strip()
                if completed.returncode != 0:
                    raise RuntimeError(detail[-5000:])
                self.root.after(0, self.refresh_native_status)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Native build", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def run_native_benchmark(self) -> None:
        def task():
            json_path, _html_path = self._report_paths("native_speed_parity_benchmark")
            level = self.analysis_level.get()
            candidates = 2000 if level == "quick" else 10000 if level == "standard" else 50000
            result = write_native_benchmark(json_path, NativeBenchmarkSettings(candidates=candidates, years=80, repeats=3))
            return {"summary": result}, json_path, None

        self._run_background("Running machine-specific native speed and parity benchmark...", task)

    def _report_paths(self, stem: str) -> tuple[Path, Path]:
        reports = ROOT / "reports" / "priority_diagnostics"
        reports.mkdir(parents=True, exist_ok=True)
        return reports / f"{stem}.json", reports / f"{stem}.html"

    def run_profile(self) -> None:
        def task():
            dataset = self._dataset()
            model_settings = ModelSettings(search_draws=160)
            fitted = fit(dataset, model_settings)
            result = profile_likelihood(
                dataset,
                model_settings,
                fitted,
                self.parameter.get(),
                ProfileSettings(
                    points=self.profile_points.get(),
                    multistarts=self.profile_multistarts.get(),
                    workers=self.workers.get(),
                    cache_dir=str(ROOT / "reports" / "cache"),
                ),
            )
            json_path, html_path = self._report_paths(f"profile_{self.parameter.get()}")
            json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            rows = [dict(row, value=row.get("fixed_value")) for row in result["profile"]]
            figure = InteractiveChartFactory(ChartProfile()).likelihood_profile(
                rows,
                parameter_key="value",
                objective_key="objective",
                component_keys=list((rows[0].get("components") or {}).keys()) if rows else None,
                title=f"Fully refitted likelihood profile — {self.parameter.get()}",
                parameter_label=self.parameter.get(),
            )
            InteractiveChartFactory(ChartProfile()).write_html(figure, html_path)
            return result, json_path, html_path

        self._run_background("Running fully refitted likelihood profile...", task)

    def run_aspm(self) -> None:
        def task():
            dataset = self._dataset()
            age = pd.read_csv(self.age_path.get()) if self.age_path.get() else None
            length = pd.read_csv(self.length_path.get()) if self.length_path.get() else None
            result = run_age_structured_aspm(
                dataset,
                age_composition=age,
                length_composition=length,
                settings=ASPMSettings(multistarts=self.profile_multistarts.get()),
            )
            json_path, html_path = self._report_paths("age_structured_aspm")
            json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            full = (result.get("full_model") or {}).get("history") or []
            series = [SeriesSpec("Full integrated age model", [row["year"] for row in full], [row["depletion"] for row in full])]
            for variant in result.get("variants") or []:
                history = variant.get("history") or []
                if history:
                    series.append(SeriesSpec(str(variant.get("name")), [row["year"] for row in history], [row["depletion"] for row in history]))
            figure = InteractiveChartFactory(ChartProfile()).time_series(series, title="Age-structured ASPM and ASPM-R", y_title="Spawning depletion")
            InteractiveChartFactory(ChartProfile()).write_html(figure, html_path)
            return result, json_path, html_path

        self._run_background("Running genuine age-structured ASPM variants...", task)

    def run_coverage(self) -> None:
        def task():
            dataset = self._dataset()
            settings = ModelSettings(search_draws=160)
            truth_fit = fit(dataset, settings)
            methods = []
            if self.coverage_hessian.get(): methods.append("hessian")
            if self.coverage_profile.get(): methods.append("profile")
            if self.coverage_bootstrap.get(): methods.append("parametric_bootstrap")
            if not methods:
                raise ValueError("Select at least one uncertainty method.")
            result = run_interval_coverage(
                dataset,
                settings,
                truth_fit,
                CoverageSettings(
                    replicates=self.coverage_replicates.get(),
                    methods=tuple(methods),
                    workers=self.workers.get(),
                    search_draws=120 if self.analysis_level.get() == "quick" else 160 if self.analysis_level.get() == "standard" else 300,
                    profile_points=9 if self.analysis_level.get() == "quick" else 13 if self.analysis_level.get() == "standard" else 21,
                    profile_multistarts=2 if self.analysis_level.get() == "quick" else 3 if self.analysis_level.get() == "standard" else 5,
                    bootstrap_replicates=20 if self.analysis_level.get() == "quick" else 80 if self.analysis_level.get() == "standard" else 250,
                    native_threads_per_worker=1,
                ),
            )
            json_path, html_path = self._report_paths("formal_interval_coverage")
            json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            rows = [dict(row, parameter=f"{row.get('method')} — {row.get('parameter')}") for row in result.get("coverage") or []]
            figure = InteractiveChartFactory(ChartProfile()).interval_coverage(rows, title="Formal known-truth interval coverage")
            InteractiveChartFactory(ChartProfile()).write_html(figure, html_path)
            return result, json_path, html_path

        self._run_background("Running formal known-truth interval coverage...", task)

    def open_json(self) -> None:
        self._open(self.last_json)

    def open_dashboard(self) -> None:
        self._open(self.last_dashboard)

    def open_reports(self) -> None:
        path = ROOT / "reports" / "priority_diagnostics"
        path.mkdir(parents=True, exist_ok=True)
        self._open(path)

    @staticmethod
    def _open(path: Path | None) -> None:
        if path is None or not path.exists():
            messagebox.showinfo("Omega Priority Diagnostics", "No matching output has been generated yet.")
            return
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(path.as_uri())


def main() -> None:
    root = Tk()
    root.title("Omega FISH — Priority Diagnostics")
    PriorityDiagnosticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
