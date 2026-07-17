from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, BooleanVar, Canvas, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stock_model.core import ModelSettings
from stock_model.data_io import read_stock_file
from stock_model.age_structured import read_composition_file
from stock_model.expert_workflow import ExpertWorkflowSettings, WorkflowOverride, run_expert_workflow
from stock_model.interactive_charts import ChartProfile, ChartProfileStore, InteractiveChartFactory, SeriesSpec
from chart_studio_app import NativeChartPreview


APP_TITLE = "Omega FISH Model — Automatic Expert Workflow"
ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "expert_workflow"
PROFILE_STORE = ChartProfileStore(Path.home() / ".omega_fish" / "chart_profiles.json")
DEMO_FILE = ROOT / "Data_Sets" / "Data_set_Age_Structured_Demo" / "model_ready_timeseries.csv"


class MetricCard(ttk.Frame):
    def __init__(self, parent, title: str) -> None:
        super().__init__(parent, padding=(14, 10), style="Card.TFrame")
        self.value = StringVar(value="—")
        self.subtitle = StringVar(value="")
        ttk.Label(self, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(self, textvariable=self.value, style="CardValue.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Label(self, textvariable=self.subtitle, style="CardSub.TLabel", wraplength=230).pack(anchor="w")


class ScrollFrame(ttk.Frame):
    def __init__(self, parent, *, width: int = 330) -> None:
        super().__init__(parent, width=width, style="Sidebar.TFrame")
        self.pack_propagate(False)
        self.canvas = Canvas(self, width=width - 18, highlightthickness=0, background="#102a43")
        self.canvas.omega_role = "workspace_controls"  # type: ignore[attr-defined]
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Sidebar.TFrame", padding=(18, 16))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.window, width=event.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.bind_all("<MouseWheel>", self._wheel, add="+")
        self.bind_all("<Button-4>", self._wheel, add="+")
        self.bind_all("<Button-5>", self._wheel, add="+")

    def _wheel(self, event):
        try:
            current = getattr(event, "widget", None)
            inside = False
            while current is not None:
                if current in {self, self.canvas, self.inner}:
                    inside = True
                    break
                current = getattr(current, "master", None)
            pointer_x = self.winfo_pointerx()
            pointer_y = self.winfo_pointery()
            inside = inside or (
                self.canvas.winfo_rootx() <= pointer_x < self.canvas.winfo_rootx() + self.canvas.winfo_width()
                and self.canvas.winfo_rooty() <= pointer_y < self.canvas.winfo_rooty() + self.canvas.winfo_height()
            )
            if not inside:
                return None
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
            self.canvas.yview_scroll(direction * 3, "units")
            return "break"
        except Exception:
            return None


class ExpertWorkflowApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1580x960")
        self.root.minsize(1180, 720)
        self.dataset_path = StringVar(value=str(DEMO_FILE))
        self.mode = StringVar(value="automatic")
        self.speed = StringVar(value="quick")
        self.model = StringVar(value="schaefer")
        self.skip_steps = StringVar(value="")
        self.override_reason = StringVar(value="")
        self.chart_profile = StringVar(value="Omega default")
        self.status = StringVar(value="Ready. Automatic mode runs every implemented expert diagnostic and records failures.")
        self.report_chart_name = StringVar(value="")
        self.result: dict[str, Any] | None = None
        self.dashboard_path: Path | None = None
        self.result_path: Path | None = None
        self.report_figures: dict[str, Any] = {}
        self.report_table_details: dict[str, str] = {}
        self._configure_style()
        self._build()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Header.TFrame", background="#0b1f33")
        style.configure("Sidebar.TFrame", background="#102a43")
        style.configure("HeaderTitle.TLabel", background="#0b1f33", foreground="white", font=("Segoe UI", 24, "bold"))
        style.configure("HeaderSub.TLabel", background="#0b1f33", foreground="#c8d6e5", font=("Segoe UI", 10))
        style.configure("SideTitle.TLabel", background="#102a43", foreground="white", font=("Segoe UI", 10, "bold"))
        style.configure("SideText.TLabel", background="#102a43", foreground="#d7e3ee", font=("Segoe UI", 9))
        style.configure("Card.TFrame", background="white", borderwidth=1, relief="solid")
        style.configure("CardTitle.TLabel", background="white", foreground="#68758a", font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background="white", foreground="#102a43", font=("Segoe UI", 20, "bold"))
        style.configure("CardSub.TLabel", background="white", foreground="#68758a", font=("Segoe UI", 8))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 9))
        style.configure("Status.TLabel", background="#e8eef5", foreground="#334155", padding=(10, 7))
        style.configure("Treeview", rowheight=27, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame")
        shell.pack(fill=BOTH, expand=True)
        header = ttk.Frame(shell, style="Header.TFrame", padding=(24, 18))
        header.pack(side=TOP, fill=X)
        ttk.Label(header, text="Automatic Expert Workflow", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Convergence, optimizer agreement, jitter, boundaries, residuals, profiles, retrospectives, hindcasts, ASPM, "
                "influence, weighting, simulation recovery, interval coverage, structural ensembles and closed-loop MSE."
            ),
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(shell, style="App.TFrame")
        body.pack(fill=BOTH, expand=True)
        sidebar_scroll = ScrollFrame(body, width=330)
        sidebar_scroll.pack(side=LEFT, fill=Y)
        self.control_scroll = sidebar_scroll
        sidebar = sidebar_scroll.inner

        self._side_label(sidebar, "Dataset")
        ttk.Entry(sidebar, textvariable=self.dataset_path).pack(fill=X, pady=(4, 5))
        ttk.Button(sidebar, text="Choose CSV or Excel", command=self.choose_dataset).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Use built-in demonstration", command=lambda: self.dataset_path.set(str(DEMO_FILE))).pack(fill=X, pady=2)

        ttk.Separator(sidebar).pack(fill=X, pady=13)
        self._side_label(sidebar, "Assessment structure")
        ttk.Combobox(sidebar, textvariable=self.model, values=("schaefer", "fox", "pella"), state="readonly").pack(fill=X, pady=(4, 7))
        self._side_label(sidebar, "Workflow mode")
        ttk.Combobox(sidebar, textvariable=self.mode, values=("automatic", "exploration"), state="readonly").pack(fill=X, pady=(4, 7))
        ttk.Label(
            sidebar,
            text="Automatic runs all gates. Exploration permits skips and overrides but keeps them visible in the evidence record.",
            style="SideText.TLabel",
            wraplength=285,
        ).pack(anchor="w")
        self._side_label(sidebar, "Analysis depth")
        ttk.Combobox(sidebar, textvariable=self.speed, values=("quick", "standard", "deep"), state="readonly").pack(fill=X, pady=(4, 7))
        ttk.Label(
            sidebar,
            text="Quick is interactive. Standard increases repetitions. Deep is intended for overnight or high-performance runs.",
            style="SideText.TLabel",
            wraplength=285,
        ).pack(anchor="w")

        ttk.Separator(sidebar).pack(fill=X, pady=13)
        self._side_label(sidebar, "Exploration overrides")
        ttk.Label(sidebar, text="Step names to skip — comma separated", style="SideText.TLabel").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.skip_steps).pack(fill=X, pady=(3, 6))
        ttk.Label(sidebar, text="Reason for override", style="SideText.TLabel").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.override_reason).pack(fill=X, pady=(3, 6))
        ttk.Label(
            sidebar,
            text="Skips only apply in exploration mode. Omega does not erase the skipped check from the final report.",
            style="SideText.TLabel",
            wraplength=285,
        ).pack(anchor="w")

        ttk.Separator(sidebar).pack(fill=X, pady=13)
        self._side_label(sidebar, "Interactive chart profile")
        profiles = PROFILE_STORE.load_all()
        ttk.Combobox(sidebar, textvariable=self.chart_profile, values=tuple(profiles), state="readonly").pack(fill=X, pady=(4, 7))
        ttk.Button(sidebar, text="Open Chart Studio", command=self.open_chart_studio).pack(fill=X, pady=2)

        ttk.Separator(sidebar).pack(fill=X, pady=13)
        ttk.Button(sidebar, text="RUN COMPLETE WORKFLOW", style="Primary.TButton", command=self.run).pack(fill=X, pady=3)
        ttk.Button(sidebar, text="Open interactive dashboard", command=self.open_dashboard).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Export workflow JSON", command=self.export_json).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Open output folder", command=self.open_output_folder).pack(fill=X, pady=2)

        main = ttk.Frame(body, style="App.TFrame", padding=(16, 14))
        main.pack(side=RIGHT, fill=BOTH, expand=True)
        cards = ttk.Frame(main, style="App.TFrame")
        cards.pack(fill=X)
        for column in range(5):
            cards.columnconfigure(column, weight=1)
        self.overall_card = MetricCard(cards, "Workflow status")
        self.grade_card = MetricCard(cards, "Reliability grade")
        self.steps_card = MetricCard(cards, "Checks completed")
        self.failure_card = MetricCard(cards, "Required failures")
        self.depletion_card = MetricCard(cards, "Terminal depletion")
        for index, card in enumerate((self.overall_card, self.grade_card, self.steps_card, self.failure_card, self.depletion_card)):
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 0 if index == 4 else 6))

        notebook = ttk.Notebook(main)
        notebook.pack(fill=BOTH, expand=True, pady=(14, 0))
        self.notebook = notebook
        steps_tab = ttk.Frame(notebook, padding=8)
        reliability_tab = ttk.Frame(notebook, padding=8)
        summary_tab = ttk.Frame(notebook, padding=8)
        report_tab = ttk.Frame(notebook, padding=8)
        json_tab = ttk.Frame(notebook, padding=8)
        notebook.add(steps_tab, text="Diagnostic Gates")
        notebook.add(reliability_tab, text="Reliability Evidence")
        notebook.add(summary_tab, text="Major Results")
        notebook.add(report_tab, text="Complete Report")
        notebook.add(json_tab, text="Raw Evidence JSON")
        self.report_tab = report_tab

        self.steps_tree = self._tree(steps_tab, ("name", "status", "required", "message", "error"))
        self.reliability_tree = self._tree(reliability_tab, ("diagnostic", "status", "value", "criterion", "impact", "why"))
        self.summary_tree = self._tree(summary_tab, ("section", "metric", "value"))
        self.json_text = self._text(json_tab)

        report_actions = ttk.Frame(report_tab)
        report_actions.pack(fill=X, pady=(0, 7))
        ttk.Label(
            report_actions,
            text="Completed-run report: scientific verdicts, charts, evidence tables and saved files",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=LEFT)
        ttk.Button(report_actions, text="Open interactive dashboard", command=self.open_dashboard).pack(side=RIGHT, padx=(6, 0))
        ttk.Button(report_actions, text="Open report folder", command=self.open_output_folder).pack(side=RIGHT)

        self.report_notebook = ttk.Notebook(report_tab)
        self.report_notebook.pack(fill=BOTH, expand=True)
        report_overview_tab = ttk.Frame(self.report_notebook, padding=8)
        report_charts_tab = ttk.Frame(self.report_notebook, padding=8)
        report_tables_tab = ttk.Frame(self.report_notebook, padding=8)
        report_files_tab = ttk.Frame(self.report_notebook, padding=8)
        self.report_notebook.add(report_overview_tab, text="Overview")
        self.report_notebook.add(report_charts_tab, text="Charts")
        self.report_notebook.add(report_tables_tab, text="Evidence Tables")
        self.report_notebook.add(report_files_tab, text="Files")

        self.report_overview_text = self._text(report_overview_tab)
        self.report_overview_text.configure(wrap="word")

        chart_bar = ttk.Frame(report_charts_tab)
        chart_bar.pack(fill=X, pady=(0, 7))
        ttk.Label(chart_bar, text="Report chart").pack(side=LEFT)
        self.report_chart_picker = ttk.Combobox(chart_bar, textvariable=self.report_chart_name, state="readonly", width=44)
        self.report_chart_picker.pack(side=LEFT, padx=(8, 0))
        self.report_chart_picker.bind("<<ComboboxSelected>>", lambda _event: self._show_report_chart())
        ttk.Button(chart_bar, text="Open all charts interactively", command=self.open_dashboard).pack(side=RIGHT)
        self.report_chart_preview = NativeChartPreview(report_charts_tab)
        self.report_chart_preview.pack(fill=BOTH, expand=True)
        self.report_chart_preview.show_message("Run the complete workflow to build the report charts.")

        report_table_split = ttk.Panedwindow(report_tables_tab, orient="vertical")
        report_table_split.pack(fill=BOTH, expand=True)
        report_table_frame = ttk.Frame(report_table_split)
        report_detail_frame = ttk.Frame(report_table_split)
        report_table_split.add(report_table_frame, weight=4)
        report_table_split.add(report_detail_frame, weight=1)
        self.report_table = self._tree(report_table_frame, ("category", "item", "status", "value"))
        self.report_table.bind("<<TreeviewSelect>>", self._show_report_table_detail)
        self.report_detail_text = self._text(report_detail_frame)
        self.report_detail_text.configure(wrap="word", height=5)

        self.report_files_text = self._text(report_files_tab)
        self.report_files_text.configure(wrap="word")

        ttk.Label(shell, textvariable=self.status, style="Status.TLabel").pack(fill=X)

    @staticmethod
    def _side_label(parent, text: str) -> None:
        ttk.Label(parent, text=text, style="SideTitle.TLabel").pack(anchor="w")

    @staticmethod
    def _tree(parent, columns: tuple[str, ...]):
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            tree.column(column, width=150 if column not in {"message", "error", "why"} else 340, stretch=True)
        return tree

    @staticmethod
    def _text(parent):
        from tkinter import Text

        text = Text(parent, wrap="none", font=("Consolas", 9))
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        return text

    def choose_dataset(self) -> None:
        filename = filedialog.askopenfilename(
            title="Choose Omega dataset",
            filetypes=(("Data files", "*.csv *.xlsx *.xlsm"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm"), ("All files", "*.*")),
        )
        if filename:
            self.dataset_path.set(filename)

    def run(self) -> None:
        if self.mode.get() == "automatic" and self.skip_steps.get().strip():
            messagebox.showinfo(APP_TITLE, "Automatic mode ignores step skips. Choose exploration mode to record and use them.")
        self.status.set("Starting complete expert workflow...")
        self._clear()
        events: Queue[tuple[str, Any]] = Queue()

        def progress(message: str) -> None:
            events.put(("progress", message))

        def worker() -> None:
            try:
                dataset = read_stock_file(self.dataset_path.get())
                skipped = ()
                overrides: tuple[WorkflowOverride, ...] = ()
                if self.mode.get() == "exploration":
                    skipped = tuple(value.strip() for value in self.skip_steps.get().split(",") if value.strip())
                    if skipped:
                        overrides = tuple(
                            WorkflowOverride("skip_step", value, self.override_reason.get().strip() or "Exploratory alternative")
                            for value in skipped
                        )
                dataset_file = Path(self.dataset_path.get())
                age_path = dataset_file.parent / "age_composition.csv"
                length_path = dataset_file.parent / "length_composition.csv"
                age_composition = read_composition_file(age_path) if age_path.exists() else None
                length_composition = read_composition_file(length_path) if length_path.exists() else None
                result = run_expert_workflow(
                    dataset,
                    ModelSettings(model=self.model.get()),
                    ExpertWorkflowSettings(
                        mode=self.mode.get(),
                        speed=self.speed.get(),
                        skipped_steps=skipped,
                        overrides=overrides,
                        cache_directory=str(REPORT_ROOT / "cache"),
                    ),
                    progress=progress,
                    age_composition=age_composition,
                    length_composition=length_composition,
                )
                REPORT_ROOT.mkdir(parents=True, exist_ok=True)
                result_path = REPORT_ROOT / "expert_workflow_latest.json"
                result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                dashboard = self._build_dashboard(result, REPORT_ROOT / "expert_workflow_dashboard.html")
                events.put(("done", (result, dashboard, result_path)))
            except Exception as exc:
                detail = traceback.format_exc()
                events.put(("error", (exc, detail)))

        def poll() -> None:
            terminal = False
            while True:
                try:
                    state, value = events.get_nowait()
                except Empty:
                    break
                if state == "progress":
                    self.status.set(f"Running: {value}")
                elif state == "done":
                    result, dashboard, result_path = value
                    self.result = result
                    self.dashboard_path = dashboard
                    self.result_path = result_path
                    self._show_result(result)
                    self.status.set(f"Workflow complete. Evidence: {result_path}")
                    terminal = True
                else:
                    exc, detail = value
                    self.status.set("Workflow failed.")
                    shell = getattr(self.root, "omega_shell", None)
                    if shell is not None:
                        shell.log_error("Automatic Expert Workflow", exc, detail)
                    messagebox.showerror(APP_TITLE, f"{exc}\n\n{detail[-2200:]}")
                    terminal = True
            if not terminal:
                try:
                    if self.root.winfo_exists():
                        self.root.after(40, poll)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(40, poll)

    def _chart_profile_value(self) -> ChartProfile:
        return PROFILE_STORE.load_all().get(self.chart_profile.get(), ChartProfile())

    def _build_dashboard(self, result: Mapping[str, Any], output: Path) -> Path:
        factory = InteractiveChartFactory(self._chart_profile_value())
        figures: dict[str, Any] = {}
        base = result.get("base") or {}
        history = base.get("history") or []
        if history:
            figures["Assessment trajectory"] = factory.time_series(
                [
                    SeriesSpec("Biomass", [row["year"] for row in history], [row["biomass"] for row in history]),
                    SeriesSpec("Depletion", [row["year"] for row in history], [row["depletion"] for row in history], yaxis="y2"),
                ],
                title="Base assessment trajectory",
                y_title="Biomass / depletion",
            )
        results = result.get("results") or {}
        jitter = results.get("jitter") or {}
        if jitter.get("runs"):
            figures["Jitter"] = factory.jitter_distribution(jitter["runs"], title="Jitter and multi-start stability")
        optimizers = results.get("optimizer_agreement") or {}
        if optimizers.get("runs"):
            figures["Optimizers"] = factory.optimizer_agreement(optimizers["runs"], label_key="algorithm", title="Independent optimizer agreement")
        residuals = results.get("residuals") or {}
        heatmap = residuals.get("heatmap") or {}
        if heatmap.get("matrix"):
            figures["Residual heatmap"] = factory.residual_heatmap(
                heatmap["matrix"], x_labels=heatmap.get("column_labels"), y_labels=heatmap.get("row_labels"), title="Residual patterns through time"
            )
        profiles = results.get("profiles") or {}
        profile_sets = profiles.get("parameters") or {}
        preferred_profile = profile_sets.get("initial_depletion") or next(iter(profile_sets.values()), None)
        if preferred_profile and preferred_profile.get("profile"):
            preferred = str(preferred_profile.get("parameter") or "parameter")
            selected = [dict(row, value=row.get("fixed_value")) for row in preferred_profile["profile"]]
            figures["Likelihood profile"] = factory.likelihood_profile(
                selected,
                parameter_key="value",
                objective_key="objective",
                component_keys=list((selected[0].get("components") or {}).keys()) if selected else None,
                title=f"Fully refitted likelihood profile — {preferred}",
                parameter_label=preferred,
            )
        component_profiles = results.get("likelihood_component_profiles") or {}
        component_rows = component_profiles.get("profiles") or []
        if component_rows:
            preferred = next((row["parameter"] for row in component_rows if row.get("parameter") == "initial_depletion"), component_rows[0].get("parameter"))
            selected = [row for row in component_rows if row.get("parameter") == preferred]
            component_keys = [key for key in selected[0] if key not in {"parameter", "value", "objective"}] if selected else []
            figures["Component conflict"] = factory.likelihood_profile(
                selected,
                parameter_key="value",
                objective_key="objective",
                component_keys=component_keys,
                title=f"Likelihood-component preferences — {preferred}",
                parameter_label=str(preferred),
            )
        retrospective = results.get("retrospective") or {}
        if retrospective.get("full"):
            figures["Retrospective"] = factory.retrospective(
                retrospective["full"],
                [row["series"] for row in retrospective.get("peels") or []],
                title="Retrospective depletion",
                y_title="Depletion",
                mohn_rho=(retrospective.get("summary") or {}).get("mohn_rho"),
            )
        hindcast = results.get("hindcast") or {}
        if hindcast.get("chart_rows"):
            figures["Hindcast"] = factory.hindcast(hindcast["chart_rows"], title="Walk-forward prediction", mase=(hindcast.get("summary") or {}).get("index_mase"))
        aspm = results.get("aspm") or {}
        full_aspm = (aspm.get("full_model") or {}).get("history") or []
        aspm_variants = [row for row in (aspm.get("variants") or []) if row.get("history")]
        if full_aspm and aspm_variants:
            series = [SeriesSpec("Full integrated age model", [row["year"] for row in full_aspm], [row["depletion"] for row in full_aspm])]
            for variant in aspm_variants:
                series.append(SeriesSpec(str(variant.get("name")), [row["year"] for row in variant["history"]], [row["depletion"] for row in variant["history"]]))
            figures["ASPM"] = factory.time_series(
                series,
                title="Age-structured ASPM and ASPM-R driver diagnostic",
                y_title="Spawning depletion",
            )
        components = results.get("likelihood_components") or []
        if components:
            figures["Likelihood components"] = factory.likelihood_conflict(components, component_key="component", value_key="objective", title="Objective and penalty components")
        removal = results.get("data_removal") or {}
        if removal.get("scenarios"):
            figures["Data influence"] = factory.likelihood_conflict(removal["scenarios"], component_key="omitted", value_key="absolute_change", title="Data-removal influence on terminal depletion")
        weighting = results.get("weighting") or {}
        if weighting.get("scenarios"):
            figures["Weight sensitivity"] = factory.optimizer_agreement(
                weighting["scenarios"], x_key="index_weight", y_key="terminal_depletion", label_key="scenario", title="Data-weight sensitivity"
            )
        composition = results.get("composition_weighting") or {}
        if composition.get("scenarios"):
            figures["Composition weights"] = factory.optimizer_agreement(
                composition["scenarios"], x_key="age_comp_weight", y_key="terminal_depletion", label_key="scenario", title="Age and length composition reweighting"
            )
        ensemble = results.get("ensemble") or {}
        combined = ensemble.get("combined_projection") or []
        if combined:
            years = [row["year"] for row in combined]
            figures["Structural ensemble"] = factory.ensemble_fan(
                years,
                [row["candidate_weighted_depletion"] for row in combined],
                [row["minimum_model_depletion"] for row in combined],
                [row["maximum_model_depletion"] for row in combined],
                title="Schaefer, Fox and Pella structural ensemble",
                y_title="Projected depletion",
            )
        simulation = results.get("simulation_recovery") or {}
        if simulation.get("coverage"):
            coverage_rows = [dict(row, parameter=f"{row.get('method')} — {row.get('parameter')}") for row in simulation["coverage"]]
            figures["Interval coverage"] = factory.interval_coverage(coverage_rows, title="Formal known-truth interval coverage")
        mse = results.get("mse") or {}
        mse_rows = mse.get("summary") if isinstance(mse.get("summary"), list) else []
        if mse_rows:
            figures["Closed-loop MSE"] = factory.mse_tradeoff(
                mse_rows,
                x_key="median_annual_catch",
                y_key="prob_terminal_above_limit",
                color_key="median_catch_cv",
                size_key="mean_closure_frequency",
                label_key="procedure",
                title="Management procedure trade-offs",
            )
        self.report_figures = figures
        return factory.write_dashboard(
            figures,
            output,
            title="Omega FISH automatic expert workflow",
            metadata={
                "Workflow status": (result.get("summary") or {}).get("status"),
                "Mode": (result.get("summary") or {}).get("mode"),
                "Depth": (result.get("summary") or {}).get("speed"),
                "Reliability grade": (result.get("summary") or {}).get("reliability_grade"),
                "Overrides": len(result.get("overrides") or []),
                "Charts": len(figures),
            },
        )

    def _show_result(self, result: Mapping[str, Any]) -> None:
        summary = result.get("summary") or {}
        self.overall_card.value.set(str(summary.get("status", "—")))
        self.grade_card.value.set(str(summary.get("reliability_grade", "—")))
        self.steps_card.value.set(f"{summary.get('steps_completed', 0)} / {summary.get('steps', 0)}")
        self.failure_card.value.set(str(summary.get("required_failures", 0)))
        depletion = summary.get("terminal_depletion")
        self.depletion_card.value.set(f"{float(depletion):.3f}" if depletion is not None else "—")
        self.overall_card.subtitle.set(f"Mode: {summary.get('mode')} — {summary.get('speed')}")
        self.grade_card.subtitle.set("Evidence-based diagnostic summary; not peer-review certification")
        self.steps_card.subtitle.set(f"Warnings: {summary.get('warnings', 0)}; skipped: {summary.get('skipped', 0)}")
        self.failure_card.subtitle.set("Failures remain visible; exploration is not blocked")
        self.depletion_card.subtitle.set("Current base production-model result")

        for tree in (self.steps_tree, self.reliability_tree, self.summary_tree):
            tree.delete(*tree.get_children())
        for step in result.get("steps") or []:
            self.steps_tree.insert("", END, values=(step.get("name"), step.get("status"), step.get("required"), step.get("message"), step.get("error") or ""))
        reliability = ((result.get("results") or {}).get("reliability") or {})
        for item in reliability.get("items") or []:
            self.reliability_tree.insert("", END, values=(item.get("name"), item.get("status"), item.get("value"), item.get("criterion"), item.get("impact"), item.get("why")))
        for section, payload in (result.get("results") or {}).items():
            if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
                for key, value in payload["summary"].items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        self.summary_tree.insert("", END, values=(section, key, value))
        self.json_text.delete("1.0", END)
        self.json_text.insert("1.0", json.dumps(result, indent=2, default=str))
        self._populate_complete_report(result)

    def _populate_complete_report(self, result: Mapping[str, Any]) -> None:
        summary = result.get("summary") or {}
        results = result.get("results") or {}
        self.report_overview_text.delete("1.0", END)
        lines = [
            "OMEGA AUTOMATIC EXPERT WORKFLOW â€” COMPLETED RUN\n",
            f"Dataset: {self.dataset_path.get()}",
            f"Overall status: {summary.get('status', 'not reported')}",
            f"Reliability grade: {summary.get('reliability_grade', 'not reported')}",
            f"Checks completed: {summary.get('steps_completed', 0)} of {summary.get('steps', 0)}",
            f"Required failures: {summary.get('required_failures', 0)}",
            f"Warnings: {summary.get('warnings', 0)}",
            f"Skipped checks: {summary.get('skipped', 0)}",
            f"Terminal depletion: {summary.get('terminal_depletion', 'not reported')}",
            "",
            "How to use this report:",
            "â€¢ Start with Required failures and Reliability grade.",
            "â€¢ Inspect every warning or failure in Evidence Tables; select a row for the full explanation and next evidence.",
            "â€¢ Use Charts to inspect fit, residual patterns, profiles, retrospective bias, interval coverage and MSE trade-offs.",
            "â€¢ Open the interactive dashboard to zoom, compare series and export figures.",
            "",
            "Scientific limitation:",
            "A completed workflow is an evidence record, not proof that the stock assessment is scientifically correct. "
            "Review input provenance, model assumptions, diagnostics and management context before using results for decisions.",
        ]
        self.report_overview_text.insert("1.0", "\n".join(lines))

        self.report_table.delete(*self.report_table.get_children())
        self.report_table_details.clear()

        def insert(category: str, item: str, status: Any, value: Any, detail: str) -> None:
            row = self.report_table.insert("", END, values=(category, item, status, value))
            self.report_table_details[row] = detail

        for key, value in summary.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                insert("Run summary", str(key).replace("_", " ").title(), "", value, f"Recorded workflow summary value: {value}")
        for step in result.get("steps") or []:
            detail = "\n".join(
                part for part in (
                    str(step.get("message") or ""),
                    f"Required: {step.get('required')}",
                    f"Error: {step.get('error')}" if step.get("error") else "",
                ) if part
            )
            insert("Diagnostic gate", str(step.get("name") or "Unnamed check"), step.get("status"), step.get("required"), detail)
        reliability = (results.get("reliability") or {})
        for item in reliability.get("items") or []:
            detail = "\n".join(
                part for part in (
                    f"Criterion: {item.get('criterion')}" if item.get("criterion") is not None else "",
                    f"Impact: {item.get('impact')}" if item.get("impact") is not None else "",
                    str(item.get("why") or ""),
                ) if part
            )
            insert("Reliability evidence", str(item.get("name") or "Evidence"), item.get("status"), item.get("value"), detail)
        for section, payload in results.items():
            if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
                for key, value in payload["summary"].items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        insert(str(section).replace("_", " ").title(), str(key).replace("_", " ").title(), "", value, f"{section}.{key} = {value}")

        chart_names = tuple(self.report_figures)
        self.report_chart_picker.configure(values=chart_names)
        if chart_names:
            self.report_chart_name.set(chart_names[0])
            self._show_report_chart()
        else:
            self.report_chart_name.set("")
            self.report_chart_preview.show_message("The run completed without chart-ready result series. Review Evidence Tables and Raw Evidence JSON.")

        self.report_files_text.delete("1.0", END)
        file_lines = [
            f"Source dataset\n{self.dataset_path.get()}",
            f"\nWorkflow evidence JSON\n{self.result_path or (REPORT_ROOT / 'expert_workflow_latest.json')}",
            f"\nInteractive chart dashboard\n{self.dashboard_path or (REPORT_ROOT / 'expert_workflow_dashboard.html')}",
            f"\nReport folder\n{REPORT_ROOT}",
            f"\nCharts generated\n{len(chart_names)}",
            "\nThe JSON preserves the complete numerical evidence. The dashboard contains the chart-ready portions of that evidence.",
        ]
        self.report_files_text.insert("1.0", "\n".join(file_lines))
        self.notebook.select(self.report_tab)

    def _show_report_chart(self) -> None:
        name = self.report_chart_name.get()
        figure = self.report_figures.get(name)
        if figure is None:
            self.report_chart_preview.show_message("Choose a completed-run chart from the list.")
            return
        layout = getattr(figure, "layout", None)
        title = str(getattr(getattr(layout, "title", None), "text", None) or name)
        x_title = str(getattr(getattr(getattr(layout, "xaxis", None), "title", None), "text", None) or "")
        y_title = str(getattr(getattr(getattr(layout, "yaxis", None), "title", None), "text", None) or "")
        self.report_chart_preview.show_figure(figure, title, x_title, y_title)

    def _show_report_table_detail(self, _event=None) -> None:
        selected = self.report_table.selection()
        detail = self.report_table_details.get(selected[0], "Select an evidence row to see its full explanation.") if selected else "Select an evidence row to see its full explanation."
        self.report_detail_text.delete("1.0", END)
        self.report_detail_text.insert("1.0", detail)

    def _clear(self) -> None:
        for card in (self.overall_card, self.grade_card, self.steps_card, self.failure_card, self.depletion_card):
            card.value.set("—")
            card.subtitle.set("")
        for tree in (self.steps_tree, self.reliability_tree, self.summary_tree, self.report_table):
            tree.delete(*tree.get_children())
        self.json_text.delete("1.0", END)
        self.report_overview_text.delete("1.0", END)
        self.report_files_text.delete("1.0", END)
        self.report_detail_text.delete("1.0", END)
        self.report_chart_name.set("")
        self.report_chart_picker.configure(values=())
        self.report_chart_preview.show_message("Run the complete workflow to build the report charts.")

    def open_dashboard(self) -> None:
        if self.dashboard_path and self.dashboard_path.exists():
            webbrowser.open(self.dashboard_path.as_uri())
        else:
            messagebox.showinfo(APP_TITLE, "Run the workflow first.")

    def export_json(self) -> None:
        if not self.result:
            messagebox.showinfo(APP_TITLE, "Run the workflow first.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON", "*.json"),), initialfile="omega_expert_workflow.json")
        if filename:
            Path(filename).write_text(json.dumps(self.result, indent=2, default=str), encoding="utf-8")
            self.status.set(f"Saved workflow evidence to {filename}")

    def open_chart_studio(self) -> None:
        import subprocess
        import sys

        subprocess.Popen([sys.executable, str(ROOT / "chart_studio_app.py")], cwd=str(ROOT))

    @staticmethod
    def open_output_folder() -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(REPORT_ROOT)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(REPORT_ROOT.as_uri())


def main() -> None:
    root = Tk()
    ExpertWorkflowApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
