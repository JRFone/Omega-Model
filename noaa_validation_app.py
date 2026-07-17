from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, Canvas, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Mapping, Sequence

from stock_model.ss3_validation import (
    NOAA_MODEL_CATALOG,
    NOAA_REPOSITORY_URL,
    NOAA_VALIDATION_COMMIT,
    capability_matrix,
    competitive_scorecard,
    download_noaa_model,
    download_latest_ss3_executable,
    validate_model_directory,
    write_validation_report,
)


APP_TITLE = "Omega FISH Model — NOAA / Stock Synthesis Validation Lab"
ROOT = Path(__file__).resolve().parent
FIXTURE_ROOT = ROOT / "validation_data" / "noaa_ss3"
CACHE_ROOT = ROOT / "validation_cache" / "noaa_ss3"
REPORT_ROOT = ROOT / "reports" / "noaa_validation"


def comparison_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Present every NOAA reference check beside Omega's independently calculated value."""

    rows: list[dict[str, Any]] = []
    for check in payload.get("checks", []):
        expected = check.get("expected")
        actual = check.get("actual")
        difference = check.get("difference")
        tolerance = check.get("tolerance")
        if difference is None:
            difference = "exact match" if check.get("status") == "PASS" else "different"
        if tolerance is None:
            tolerance = "exact"
        comparison_type = "structural reference" if isinstance(expected, bool) else "NOAA reference value"
        rows.append(
            {
                "comparison": check.get("name", "Unnamed check"),
                "NOAA_reference": expected,
                "Omega_result": actual,
                "difference": difference,
                "allowed_difference": tolerance,
                "verdict": check.get("status", "UNKNOWN"),
                "quick_visual": "MATCHES REFERENCE" if check.get("status") == "PASS" else "MISMATCH — INSPECT",
                "evidence": check.get("detail") or f"{comparison_type}; {check.get('category', 'general')}",
            }
        )
    return rows


class MetricCard(ttk.Frame):
    def __init__(self, parent, title: str, value: str = "—", subtitle: str = "") -> None:
        super().__init__(parent, padding=(16, 12), style="Card.TFrame")
        self.title_var = StringVar(value=title)
        self.value_var = StringVar(value=value)
        self.subtitle_var = StringVar(value=subtitle)
        ttk.Label(self, textvariable=self.title_var, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(self, textvariable=self.value_var, style="CardValue.TLabel").pack(anchor="w", pady=(4, 1))
        ttk.Label(self, textvariable=self.subtitle_var, style="CardSubtitle.TLabel", wraplength=220).pack(anchor="w")


class BarSummary(Canvas):
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, height=170, background="#ffffff", highlightthickness=0, **kwargs)
        self.bind("<Configure>", lambda _event: self.redraw())
        self.values: dict[str, float] = {}

    def set_values(self, values: Mapping[str, float]) -> None:
        self.values = {str(key): float(value) for key, value in values.items()}
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 420)
        height = max(self.winfo_height(), 160)
        if not self.values:
            self.create_text(width / 2, height / 2, text="Run a validation to populate this chart.", fill="#68758a", font=("Segoe UI", 10))
            return
        maximum = max(max(self.values.values()), 1.0)
        left = 46
        right = 24
        bottom = height - 32
        top = 18
        available = width - left - right
        bar_gap = 20
        bar_width = max(36, (available - bar_gap * (len(self.values) - 1)) / len(self.values))
        palette = {"PASS": "#15803d", "FAIL": "#b91c1c", "PARITY": "#2563eb", "GAP": "#d97706", "PARTIAL": "#7c3aed"}
        for index, (label, value) in enumerate(self.values.items()):
            x0 = left + index * (bar_width + bar_gap)
            x1 = x0 + bar_width
            y1 = bottom
            y0 = bottom - (bottom - top) * value / maximum
            self.create_rectangle(x0, y0, x1, y1, fill=palette.get(label.upper(), "#3b82f6"), outline="")
            self.create_text((x0 + x1) / 2, y0 - 10, text=f"{value:g}", fill="#172033", font=("Segoe UI", 10, "bold"))
            self.create_text((x0 + x1) / 2, bottom + 16, text=label, fill="#4b5563", font=("Segoe UI", 9))


class NOAAValidationApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1580x960")
        self.root.minsize(1180, 720)
        self.status = StringVar(value="Ready. The embedded NOAA Simple fixture can be validated offline.")
        self.model_var = StringVar(value="Simple")
        self.folder_var = StringVar(value=str(FIXTURE_ROOT / "Simple"))
        self.executable_var = StringVar(value="")
        self.result: dict[str, Any] | None = None
        self._configure_style()
        self._build()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Sidebar.TFrame", background="#102a43")
        style.configure("Header.TFrame", background="#0b1f33")
        style.configure("HeaderTitle.TLabel", background="#0b1f33", foreground="#ffffff", font=("Segoe UI", 24, "bold"))
        style.configure("HeaderSub.TLabel", background="#0b1f33", foreground="#b9c9d8", font=("Segoe UI", 10))
        style.configure("SideTitle.TLabel", background="#102a43", foreground="#ffffff", font=("Segoe UI", 11, "bold"))
        style.configure("SideText.TLabel", background="#102a43", foreground="#d7e3ee", font=("Segoe UI", 9))
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#68758a", font=("Segoe UI", 9, "bold"))
        style.configure("CardValue.TLabel", background="#ffffff", foreground="#102a43", font=("Segoe UI", 22, "bold"))
        style.configure("CardSubtitle.TLabel", background="#ffffff", foreground="#68758a", font=("Segoe UI", 8))
        style.configure("Section.TLabel", background="#f4f7fb", foreground="#102a43", font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", background="#e8eef5", foreground="#334155", padding=(10, 7))
        style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), padding=(10, 8))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame")
        shell.pack(fill=BOTH, expand=True)

        header = ttk.Frame(shell, style="Header.TFrame", padding=(24, 18))
        header.pack(side=TOP, fill=X)
        ttk.Label(header, text="NOAA / Stock Synthesis Validation Lab", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Pinned NOAA test models, deterministic equation checks, native SS3 execution, feature-parity tracking, "
                "and evidence-backed gap reporting."
            ),
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Panedwindow(shell, orient="horizontal")
        body.pack(fill=BOTH, expand=True)
        self.workspace_pane = body

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", padding=(18, 18), width=310)
        ttk.Label(sidebar, text="1. NOAA model data", style="SideTitle.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="Official NOAA SS3 test model", style="SideText.TLabel").pack(anchor="w", pady=(2, 6))
        model_box = ttk.Combobox(sidebar, textvariable=self.model_var, values=tuple(NOAA_MODEL_CATALOG), state="readonly")
        model_box.pack(fill=X)
        model_box.bind("<<ComboboxSelected>>", self._model_changed)

        ttk.Separator(sidebar).pack(fill=X, pady=14)
        ttk.Label(sidebar, text="Model folder", style="SideTitle.TLabel").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.folder_var).pack(fill=X, pady=(5, 5))
        ttk.Button(sidebar, text="Choose local SS3 folder", command=self.choose_folder).pack(fill=X, pady=2)
        self.download_model_button = ttk.Button(sidebar, text="Download selected NOAA model", command=self.download_selected)
        self.download_model_button.pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Use embedded Simple fixture", command=self.use_fixture).pack(fill=X, pady=2)

        ttk.Separator(sidebar).pack(fill=X, pady=14)
        ttk.Label(sidebar, text="2. Official SS3 program", style="SideTitle.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text="Optional. Select it to run NOAA's program as well as Omega's checks.", style="SideText.TLabel", wraplength=270).pack(anchor="w", pady=(2, 5))
        ttk.Entry(sidebar, textvariable=self.executable_var).pack(fill=X, pady=(0, 5))
        ttk.Button(sidebar, text="Choose SS3 executable", command=self.choose_executable).pack(fill=X, pady=2)
        self.download_executable_button = ttk.Button(sidebar, text="Download official SS3 executable", command=self.download_ss3_executable)
        self.download_executable_button.pack(fill=X, pady=2)

        ttk.Separator(sidebar).pack(fill=X, pady=14)
        ttk.Label(sidebar, text="3. Run and compare", style="SideTitle.TLabel").pack(anchor="w")
        ttk.Label(
            sidebar,
            text="Runs the selected NOAA data through Omega and opens every reference answer beside Omega's result.",
            style="SideText.TLabel",
            wraplength=270,
        ).pack(anchor="w", pady=(2, 6))
        self.run_compare_button = ttk.Button(sidebar, text="RUN NOAA DATA + COMPARE", style="Primary.TButton", command=self.run_validation)
        self.run_compare_button.pack(fill=X, pady=(0, 5))
        ttk.Button(sidebar, text="Export HTML + JSON report", command=self.export_report).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Open NOAA source repository", command=lambda: webbrowser.open(NOAA_REPOSITORY_URL)).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Open report folder", command=self.open_report_folder).pack(fill=X, pady=2)
        ttk.Button(sidebar, text="Open Interactive Chart Studio", command=self.open_chart_studio).pack(fill=X, pady=2)

        ttk.Separator(sidebar).pack(fill=X, pady=14)
        ttk.Label(sidebar, text="Pinned source", style="SideTitle.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text=f"Commit\n{NOAA_VALIDATION_COMMIT}", style="SideText.TLabel", wraplength=270).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            sidebar,
            text="NOAA test-model data are for software testing only and may be altered. They are not stock-status evidence.",
            style="SideText.TLabel",
            wraplength=270,
        ).pack(anchor="w", pady=(12, 0))

        main = ttk.Frame(body, style="App.TFrame", padding=(18, 14))
        body.add(sidebar, weight=0)
        body.add(main, weight=1)
        self.root.after(120, lambda: self._set_sidebar_width(310))

        cards = ttk.Frame(main, style="App.TFrame")
        cards.pack(fill=X)
        for column in range(4):
            cards.columnconfigure(column, weight=1)
        self.status_card = MetricCard(cards, "Validation status", "NOT RUN", "Run the pinned fixture or a downloaded model.")
        self.status_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.check_card = MetricCard(cards, "Deterministic checks", "—", "Parser and equation checks")
        self.check_card.grid(row=0, column=1, sticky="nsew", padx=8)
        self.parity_card = MetricCard(cards, "Feature parity", "—", "Implemented vs NOAA model features")
        self.parity_card.grid(row=0, column=2, sticky="nsew", padx=8)
        self.native_card = MetricCard(cards, "Official SS3 run", "NOT RUN", "Select NOAA's SS3 executable for a direct run")
        self.native_card.grid(row=0, column=3, sticky="nsew", padx=(8, 0))

        ttk.Label(main, text="Validation workspace", style="Section.TLabel").pack(anchor="w", pady=(16, 8))
        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill=BOTH, expand=True)
        self.comparison_tab = ttk.Frame(self.tabs, padding=8)
        self.overview_tab = ttk.Frame(self.tabs, padding=12)
        self.checks_tab = ttk.Frame(self.tabs, padding=8)
        self.capability_tab = ttk.Frame(self.tabs, padding=8)
        self.structure_tab = ttk.Frame(self.tabs, padding=8)
        self.scorecard_tab = ttk.Frame(self.tabs, padding=8)
        self.native_tab = ttk.Frame(self.tabs, padding=8)
        self.raw_tab = ttk.Frame(self.tabs, padding=8)
        for frame, title in [
            (self.comparison_tab, "NOAA vs Omega"),
            (self.overview_tab, "Overview"),
            (self.checks_tab, "Checks"),
            (self.capability_tab, "Feature parity"),
            (self.structure_tab, "Model structure"),
            (self.scorecard_tab, "Better-than-SS scorecard"),
            (self.native_tab, "Native SS3"),
            (self.raw_tab, "Raw result"),
        ]:
            self.tabs.add(frame, text=title)

        self.comparison_intro = StringVar(
            value="Click RUN NOAA DATA + COMPARE. Each NOAA reference answer will appear beside Omega's result, difference, tolerance, and verdict."
        )
        ttk.Label(
            self.comparison_tab,
            textvariable=self.comparison_intro,
            wraplength=1050,
            justify="left",
        ).pack(anchor="w", fill=X, padx=4, pady=(2, 10))
        self.comparison_tree = self._tree(self.comparison_tab)
        self.chart = BarSummary(self.overview_tab)
        self.chart.pack(fill=X, pady=(0, 12))
        self.overview_tree = self._tree(self.overview_tab)
        self.checks_tree = self._tree(self.checks_tab)
        self.capability_tree = self._tree(self.capability_tab)
        self.structure_tree = self._tree(self.structure_tab)
        self.scorecard_tree = self._tree(self.scorecard_tab)
        self._fill(self.scorecard_tree, competitive_scorecard())
        self.native_tree = self._tree(self.native_tab)
        from tkinter import Text

        self.raw_text = Text(self.raw_tab, wrap="none", font=("Consolas", 9), background="#0f172a", foreground="#dbeafe", insertbackground="#ffffff")
        self.raw_text.pack(fill=BOTH, expand=True)

        ttk.Label(shell, textvariable=self.status, style="Status.TLabel").pack(side="bottom", fill=X)

    def _set_sidebar_width(self, width: int) -> None:
        try:
            maximum = max(240, self.workspace_pane.winfo_width() - 620)
            self.workspace_pane.sashpos(0, max(240, min(width, maximum)))
        except Exception:
            pass

    @staticmethod
    def _tree(parent) -> ttk.Treeview:
        holder = ttk.Frame(parent)
        holder.pack(fill=BOTH, expand=True)
        tree = ttk.Treeview(holder, show="headings")
        yscroll = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        yscroll.pack(side=RIGHT, fill=Y)
        xscroll.pack(side="bottom", fill=X)
        return tree

    @staticmethod
    def _fill(tree: ttk.Treeview, rows: Sequence[Mapping[str, Any]]) -> None:
        tree.delete(*tree.get_children())
        if not rows:
            tree["columns"] = ()
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column.replace("_", " ").title())
            width = 110 if column in {"status", "parity", "tier"} else 170
            tree.column(column, width=width, minwidth=80, stretch=True)
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.8g}"
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, default=str)[:1000]
                values.append(value)
            tree.insert("", END, values=values)

    def _model_changed(self, _event=None) -> None:
        model = self.model_var.get()
        cached = CACHE_ROOT / model
        fixture = FIXTURE_ROOT / model
        if cached.exists():
            self.folder_var.set(str(cached))
        elif fixture.exists():
            self.folder_var.set(str(fixture))
        else:
            self.folder_var.set(str(cached))
        self.parity_card.value_var.set("NOT RUN")
        self.status.set(f"Selected NOAA model {model}.")

    def choose_folder(self) -> None:
        value = filedialog.askdirectory(title="Choose an SS3 model folder")
        if value:
            self.folder_var.set(value)
            self.model_var.set(Path(value).name)

    def choose_executable(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose the Stock Synthesis executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if value:
            self.executable_var.set(value)

    def use_fixture(self) -> None:
        self.model_var.set("Simple")
        self.folder_var.set(str(FIXTURE_ROOT / "Simple"))
        self.status.set("Using the embedded pinned NOAA Simple fixture.")

    def _background(self, label: str, function, on_success) -> None:
        self.status.set(label)

        def worker() -> None:
            try:
                value = function()
                self.root.after(0, lambda: on_success(value))
            except Exception:
                error = traceback.format_exc()
                self.root.after(0, lambda: self.status.set("Operation failed."))
                self.root.after(0, lambda: messagebox.showerror("Omega FISH validation error", error))

        threading.Thread(target=worker, daemon=True).start()

    def download_selected(self) -> None:
        model = self.model_var.get()
        destination = CACHE_ROOT / model

        def complete(manifest: dict[str, Any]) -> None:
            self.folder_var.set(str(destination))
            self.status.set(f"Downloaded {len(manifest['downloaded_files'])} NOAA files for {model}.")
            messagebox.showinfo("NOAA model download", f"Downloaded {model} to:\n{destination}")

        self._background(
            f"Downloading official NOAA model {model} from pinned commit...",
            lambda: download_noaa_model(model, destination),
            complete,
        )

    def download_ss3_executable(self) -> None:
        destination = ROOT / "tools" / "ss3"

        def complete(manifest: dict[str, Any]) -> None:
            self.executable_var.set(manifest["executable"])
            self.status.set(f"Downloaded official SS3 release {manifest.get('release_tag')}; executable selected.")
            messagebox.showinfo(
                "Stock Synthesis download",
                f"Downloaded from the official NOAA/NMFS repository.\n\nExecutable:\n{manifest['executable']}\n\nThe binary has not been run yet.",
            )

        self._background(
            "Downloading the latest official NOAA/NMFS Stock Synthesis executable...",
            lambda: download_latest_ss3_executable(destination, platform_name="windows"),
            complete,
        )

    def run_validation(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        executable = self.executable_var.get().strip() or None
        model = self.model_var.get() or folder.name

        def complete(result) -> None:
            self.result = asdict(result)
            self._apply_result(self.result)

        self._background(
            f"Running NOAA data and comparing Omega results for {model}...",
            lambda: validate_model_directory(folder, model_name=model, native_executable=executable),
            complete,
        )

    def _apply_result(self, payload: dict[str, Any]) -> None:
        summary = payload.get("summary", {})
        status = summary.get("validation_status", "UNKNOWN")
        passed = int(summary.get("checks_passed", 0))
        total = int(summary.get("checks_total", 0))
        parity = int(summary.get("capabilities_at_parity", 0))
        parity_total = int(summary.get("capabilities_total", 0))
        native = payload.get("native_ss3")

        self.status_card.value_var.set(status)
        self.status_card.subtitle_var.set(summary.get("claim_limit", ""))
        self.check_card.value_var.set(f"{passed}/{total}")
        self.check_card.subtitle_var.set("Pinned parser and deterministic-equation checks")
        self.parity_card.value_var.set(f"{parity}/{parity_total}")
        self.parity_card.subtitle_var.set("Only fully implemented features count as parity")
        if native:
            self.native_card.value_var.set(native.get("status", "UNKNOWN"))
            self.native_card.subtitle_var.set(f"Return code {native.get('return_code')}; Report.sso: {native.get('report_created')}")
        else:
            self.native_card.value_var.set("NOT RUN")
            self.native_card.subtitle_var.set("Omega comparison ran; choose NOAA's SS3 executable for a direct SS3 run too")

        rows = comparison_rows(payload)
        self._fill(self.comparison_tree, rows)
        native_text = (
            f"Official NOAA SS3 executable: {native.get('status', 'UNKNOWN')} (return code {native.get('return_code')}). "
            if native
            else "Official NOAA SS3 executable: not run. The table still compares the pinned NOAA reference values with Omega's answers. "
        )
        self.comparison_intro.set(
            native_text
            + f"Omega matched {passed} of {total} selected parser and deterministic-equation checks. "
            + "This is visible, reproducible evidence, but it is not a claim of full SS3 numerical equivalence."
        )
        self.tabs.select(self.comparison_tab)

        failures = total - passed
        partial = sum(row.get("omega_status") == "partial" for row in payload.get("capability_matrix", []))
        gaps = sum(row.get("omega_status") == "not_implemented" for row in payload.get("capability_matrix", []))
        self.chart.set_values({"PASS": passed, "FAIL": failures, "PARITY": parity, "PARTIAL": partial, "GAP": gaps})

        overview = [
            {"item": "Model", "value": payload.get("model_name")},
            {"item": "Source repository", "value": payload.get("source_repository")},
            {"item": "Pinned commit", "value": payload.get("source_commit")},
            {"item": "Source mode", "value": payload.get("source_mode")},
            {"item": "Validation status", "value": status},
            {"item": "Checks passed", "value": f"{passed}/{total}"},
            {"item": "Feature parity", "value": f"{parity}/{parity_total}"},
            {"item": "Claim boundary", "value": summary.get("claim_limit")},
        ]
        self._fill(self.overview_tree, overview)
        self._fill(self.checks_tree, payload.get("checks", []))
        self._fill(self.capability_tree, payload.get("capability_matrix", []))

        data = payload.get("data", {})
        control = payload.get("control", {})
        structure = [
            {"section": "data", "field": "years", "value": f"{data.get('start_year')}–{data.get('end_year')}"},
            {"section": "data", "field": "seasons", "value": data.get("seasons")},
            {"section": "data", "field": "sexes", "value": data.get("sexes")},
            {"section": "data", "field": "maximum age", "value": data.get("max_age")},
            {"section": "data", "field": "areas", "value": data.get("areas")},
            {"section": "data", "field": "fleets", "value": len(data.get("fleets", []))},
            {"section": "data", "field": "catch rows", "value": len(data.get("catches", []))},
            {"section": "data", "field": "index rows", "value": len(data.get("indices", []))},
            {"section": "control", "field": "natural mortality type", "value": control.get("natural_mortality_type")},
            {"section": "control", "field": "growth model", "value": control.get("growth_model")},
            {"section": "control", "field": "maturity option", "value": control.get("maturity_option")},
            {"section": "control", "field": "stock-recruit option", "value": control.get("stock_recruit_option")},
            {"section": "control", "field": "F method", "value": control.get("fishing_mortality_method")},
            {"section": "control", "field": "estimated parameters detected", "value": len(control.get("parameters", {}))},
            {"section": "control", "field": "recruitment deviations", "value": len(control.get("recruitment_deviations", []))},
        ]
        self._fill(self.structure_tree, structure)
        self._fill(self.scorecard_tree, competitive_scorecard())
        native_rows = []
        if native:
            for key, value in native.items():
                native_rows.append({"field": key, "value": value})
        self._fill(self.native_tree, native_rows)
        self.raw_text.delete("1.0", END)
        self.raw_text.insert("1.0", json.dumps(payload, indent=2, default=str))
        self.status.set(f"Comparison complete: {status}; {passed}/{total} NOAA reference checks matched. The NOAA vs Omega tab is open.")

    def export_report(self) -> None:
        if not self.result:
            messagebox.showinfo("Omega FISH", "Run a validation before exporting.")
            return
        from stock_model.ss3_validation import NOAAValidationResult

        payload = dict(self.result)
        # Reconstruct the dataclass only at the outer layer; nested values are plain serialisable mappings.
        result = NOAAValidationResult(**payload)
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        outputs = write_validation_report(result, REPORT_ROOT)
        self.status.set(f"Validation report exported to {outputs['html']}")
        try:
            os.startfile(outputs["html"])  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(Path(outputs["html"]).resolve().as_uri())

    def open_chart_studio(self) -> None:
        try:
            subprocess.Popen([sys.executable, str(ROOT / "omega_desktop.py"), "--mode", "charts"], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def open_report_folder(self) -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(REPORT_ROOT)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(REPORT_ROOT.resolve().as_uri())


def main() -> None:
    root = Tk()
    NOAAValidationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
