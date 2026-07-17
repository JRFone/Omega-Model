from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, BooleanVar, Canvas, Menu, PanedWindow, StringVar, Tk, messagebox
from tkinter import ttk
from typing import Any, Callable

from ui.datasets import DatasetEntry, DatasetLibrary, materialize_omega_timeseries
from ui.themes import ThemeManager
from ui.tutorial import TutorialController, TutorialTarget


APP_NAME = "Omega FISH Model"
APP_VERSION = "1.4.1"
SOURCE_ROOT = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT
DATA_ROOT = SOURCE_ROOT / "Data_Sets"
PROFILE_DIR = Path(os.environ.get("APPDATA", Path.home())) / "OmegaFISH"
PROFILE_PATH = PROFILE_DIR / "desktop_profile.json"


WORKSPACES: dict[str, tuple[str, str]] = {
    "integrated": (
        "Integrated Assessment",
        "Fit and inspect the age-structured assessment, compositions, biomass, fishing mortality, projections, and sector results.",
    ),
    "parameters": (
        "Visual Parameter Lab",
        "Move realistic model-parameter sliders and see the biomass scenario, observations, reference points, and warnings update immediately.",
    ),
    "truthmse": (
        "Biomass Evidence & Advanced MSE",
        "Combine competing biomass evidence, test separate operating truths, and compare management procedures.",
    ),
    "priority": (
        "Priority Diagnostics",
        "Run refitted likelihood profiles, age-structured ASPM, interval coverage, and native-engine parity tests.",
    ),
    "expert": (
        "Automatic Expert Workflow",
        "Run the wider convergence, residual, retrospective, hindcast, influence, recovery, and reliability workflow.",
    ),
    "charts": (
        "Interactive Chart Studio",
        "Create zoomable, pannable, configurable Plotly charts and export interactive evidence dashboards.",
    ),
    "quant": (
        "Quant Lab",
        "Explore optimisers, high-dimensional diagnostics, stress tests, model ensembles, and risk frontiers.",
    ),
    "noaa": (
        "NOAA / SS3 Validation",
        "Inspect official test configurations, parse SS3 files, run parity checks, and preserve validation evidence.",
    ),
    "validation": (
        "Validation & Legacy MSE",
        "Run deterministic benchmarks, tagging examples, CPUE standardisation, reliability checks, and the legacy MSE.",
    ),
}


def enable_windows_dpi_awareness() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))
    return base / relative


def apply_window_identity(root: Tk, title: str) -> None:
    root.title(title)
    icon = resource_path("assets/omega_fish.ico")
    if icon.exists():
        try:
            root.iconbitmap(default=str(icon))
        except Exception:
            pass


def create_workspace(mode: str, root) -> Any:
    if mode == "integrated":
        from integrated_assessment_app import IntegratedAssessmentApp

        return IntegratedAssessmentApp(root)
    if mode == "parameters":
        from visual_parameter_lab_app import VisualParameterLabApp

        return VisualParameterLabApp(root)
    if mode == "quant":
        from quant_lab_app import QuantLabApp

        return QuantLabApp(root)
    if mode == "validation":
        from omega_complete_app import CompleteAssessmentApp

        return CompleteAssessmentApp(root)
    if mode == "noaa":
        from noaa_validation_app import NOAAValidationApp

        return NOAAValidationApp(root)
    if mode == "expert":
        from expert_workflow_app import ExpertWorkflowApp

        return ExpertWorkflowApp(root)
    if mode == "charts":
        from chart_studio_app import ChartStudioApp

        return ChartStudioApp(root)
    if mode == "priority":
        from priority_diagnostics_app import PriorityDiagnosticsApp

        return PriorityDiagnosticsApp(root)
    if mode == "truthmse":
        from mse_truth_lab_app import MSETruthLabApp

        return MSETruthLabApp(root)
    raise ValueError(f"Unknown workspace: {mode}")


def run_mode(mode: str) -> None:
    """Run one explicitly detached workspace in its own window."""

    root = Tk()
    title = WORKSPACES[mode][0]
    apply_window_identity(root, f"{APP_NAME} — {title}")
    create_workspace(mode, root)
    root.mainloop()


class EmbeddedWorkspace(ttk.Frame):
    """Frame compatible with legacy workspace constructors that expect ``Tk``."""

    def title(self, *_args: object) -> None:
        return None

    def geometry(self, *_args: object) -> None:
        return None

    def minsize(self, *_args: object) -> None:
        return None


class AdjustablePanedWindow(PanedWindow):
    """Classic Tk splitter with the ttk-compatible sashpos convenience method."""

    def sashpos(self, index: int, newpos: int | None = None) -> int:
        if newpos is None:
            x, y = self.sash_coord(index)
            return int(x if str(self.cget("orient")) == "horizontal" else y)
        if str(self.cget("orient")) == "horizontal":
            self.sash_place(index, int(newpos), 1)
        else:
            self.sash_place(index, 1, int(newpos))
        return int(newpos)


class DatasetLibraryView(ttk.Frame):
    def __init__(self, parent, shell: "OmegaShell") -> None:
        super().__init__(parent, padding=16)
        self.shell = shell
        self.search = StringVar()
        self.difficulty = StringVar(value="All levels")
        self.coverage = StringVar(value="All datasets")
        self.details = StringVar(value="Select a dataset to see its purpose and available inputs.")
        self.entries: list[DatasetEntry] = []
        self.filtered: list[DatasetEntry] = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        title = ttk.Frame(self)
        title.pack(fill=X)
        ttk.Label(title, text="Dataset Library", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            title,
            text="Choose data without browsing through technical folders. Omega never edits an original dataset when it runs a model.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 12))

        filters = ttk.Frame(self)
        filters.pack(fill=X, pady=(0, 10))
        filters.columnconfigure(1, weight=1)
        ttk.Label(filters, text="Search").grid(row=0, column=0, sticky="w")
        entry = ttk.Entry(filters, textvariable=self.search, width=34)
        entry.grid(row=0, column=1, sticky="ew", padx=(6, 14))
        entry.bind("<KeyRelease>", lambda _event: self.apply_filter())
        ttk.Label(filters, text="Difficulty").grid(row=0, column=2, sticky="w")
        level = ttk.Combobox(
            filters,
            textvariable=self.difficulty,
            values=("All levels", "Beginner", "Intermediate", "Advanced"),
            state="readonly",
            width=14,
        )
        level.grid(row=0, column=3, sticky="w", padx=6)
        level.bind("<<ComboboxSelected>>", lambda _event: self.apply_filter())
        ttk.Label(filters, text="Coverage").grid(row=0, column=4, sticky="w", padx=(10, 0))
        coverage = ttk.Combobox(
            filters,
            textvariable=self.coverage,
            values=("All datasets", "Full Omega dataset", "Full NOAA/SS3 model", "Partial inputs"),
            state="readonly",
            width=20,
        )
        coverage.grid(row=0, column=5, sticky="w", padx=6)
        coverage.bind("<<ComboboxSelected>>", lambda _event: self.apply_filter())
        actions = ttk.Frame(filters)
        actions.grid(row=1, column=0, columnspan=6, sticky="e", pady=(7, 0))
        ttk.Button(actions, text="Download / update NOAA data", command=self.shell.refresh_noaa_library).pack(side=LEFT, padx=6)
        ttk.Button(actions, text="Refresh library", command=self.refresh).pack(side=LEFT)

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill=BOTH, expand=True)
        table_frame = ttk.Frame(pane)
        detail_frame = ttk.Frame(pane, padding=(14, 0, 0, 0))
        pane.add(table_frame, weight=3)
        pane.add(detail_frame, weight=2)

        columns = ("name", "source", "difficulty", "coverage", "type", "data")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "name": "Dataset",
            "source": "Source",
            "difficulty": "Level",
            "coverage": "Coverage",
            "type": "Model type",
            "data": "Available data",
        }
        widths = {"name": 190, "source": 75, "difficulty": 80, "coverage": 145, "type": 125, "data": 190}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], minwidth=70, stretch=True)
        ybar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        xbar = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        ybar.pack(side=RIGHT, fill=Y)
        xbar.pack(side="bottom", fill=X)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._show_selected())
        self.tree.bind("<Double-1>", lambda _event: self.load_selected())

        ttk.Label(detail_frame, text="Dataset preview", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(detail_frame, textvariable=self.details, wraplength=340, justify="left").pack(anchor="w", fill=X, pady=(8, 14))
        self.load_button = ttk.Button(detail_frame, text="LOAD DATASET", command=self.load_selected, style="Accent.TButton")
        self.load_button.pack(fill=X, pady=4)
        ttk.Button(detail_frame, text="Open dataset folder", command=self.open_selected_folder).pack(fill=X, pady=4)
        self.beginner_button = ttk.Button(detail_frame, text="Load beginner dataset", command=self.shell.load_beginner_dataset)
        self.beginner_button.pack(fill=X, pady=4)
        ttk.Label(
            detail_frame,
            text="Stock Synthesis folders can be inspected and validated in the NOAA workspace. CSV model-ready datasets can also be passed directly into Integrated Assessment.",
            style="Muted.TLabel",
            wraplength=340,
            justify="left",
        ).pack(anchor="w", pady=(16, 0))

    def refresh(self) -> None:
        self.entries = self.shell.dataset_library.scan()
        self.apply_filter()

    def apply_filter(self) -> None:
        query = self.search.get().strip().lower()
        level = self.difficulty.get()
        coverage = self.coverage.get()
        self.filtered = [
            item
            for item in self.entries
            if (not query or query in " ".join((item.display_name, item.source, item.description, item.coverage, " ".join(item.data_types))).lower())
            and (level == "All levels" or item.difficulty.lower() == level.lower())
            and (coverage == "All datasets" or item.coverage == coverage)
        ]
        self.tree.delete(*self.tree.get_children())
        for index, item in enumerate(self.filtered):
            self.tree.insert(
                "",
                END,
                iid=str(index),
                values=(item.display_name, item.source, item.difficulty, item.coverage, item.model_type, ", ".join(item.data_types) or "Files only"),
            )
        if self.filtered:
            self.tree.selection_set("0")
            self._show_selected()
        else:
            self.details.set("No datasets match the current filters.")

    def selected(self) -> DatasetEntry | None:
        selected = self.tree.selection()
        if not selected:
            return None
        index = int(selected[0])
        return self.filtered[index] if 0 <= index < len(self.filtered) else None

    def _show_selected(self) -> None:
        item = self.selected()
        if item is None:
            return
        file_text = str(item.primary_file.name) if item.primary_file else "SS3 / auxiliary files"
        self.details.set(
            f"{item.display_name}\n\n{item.description}\n\n"
            f"Source: {item.source}\nLevel: {item.difficulty}\nCoverage: {item.coverage}\nModel: {item.model_type}\n"
            f"Workspace support: {item.workspace_coverage}\n"
            f"Inputs: {', '.join(item.data_types) or 'not yet catalogued'}\nPrimary file: {file_text}\n\n"
            + (f"Recommended tools: {', '.join(item.recommended_tools)}\n\n" if item.recommended_tools else "")
            + (f"Expected behaviour: {item.expected_behavior}\n\n" if item.expected_behavior else "")
            + "Loading selects this dataset for Omega. Original files remain unchanged."
        )

    def load_selected(self) -> None:
        item = self.selected()
        if item is not None:
            self.shell.set_active_dataset(item)

    def open_selected_folder(self) -> None:
        item = self.selected()
        if item is not None:
            self.shell.open_path(item.root)


class SettingsView(ttk.Frame):
    """Persistent, user-facing application settings."""

    def __init__(self, parent, shell: "OmegaShell") -> None:
        super().__init__(parent, padding=22, style="Shell.TFrame")
        self.shell = shell
        self.default_workload = StringVar(value=str(shell.profile.get("default_workload", "Standard")).title())
        self.sidebar_width = StringVar(value=str(shell.profile.get("sidebar_width", 225)))
        self.sidebar_position = StringVar(value=str(shell.profile.get("sidebar_position", "Left")).title())
        self.confirm_dataset_switch = BooleanVar(value=bool(shell.profile.get("confirm_dataset_switch", True)))
        self.adapt_ss3 = BooleanVar(value=bool(shell.profile.get("adapt_ss3_for_omega", True)))
        self.auto_load_charts = BooleanVar(value=bool(shell.profile.get("auto_load_dataset_in_charts", True)))
        self._build()

    def _build(self) -> None:
        ttk.Label(self, text="Settings", font=("Segoe UI", 24, "bold")).pack(anchor="w")
        ttk.Label(
            self,
            text="These settings are saved for the next launch and applied to workspaces in this window.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 16))

        appearance = ttk.LabelFrame(self, text="Appearance and layout", padding=14)
        appearance.pack(fill=X, pady=6)
        self._choice(appearance, 0, "Theme", self.shell.theme, ("Dark", "Light", "System", "High Contrast"))
        self._choice(appearance, 1, "Display density", self.shell.density, ("Large Text", "Comfortable", "Compact"))
        self._choice(appearance, 2, "Sidebar position", self.sidebar_position, ("Left", "Top", "Right", "Bottom"))
        ttk.Label(appearance, text="Sidebar size (170–480 pixels)").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Spinbox(appearance, from_=170, to=480, textvariable=self.sidebar_width, width=12).grid(row=3, column=1, sticky="w", padx=8)

        analysis = ttk.LabelFrame(self, text="Analysis defaults", padding=14)
        analysis.pack(fill=X, pady=6)
        self._choice(analysis, 0, "Default workload", self.default_workload, ("Quick", "Standard", "Formal"))
        ttk.Label(
            analysis,
            text="Standard is the realistic general-purpose default. Quick is for inspection; Formal can be much slower and is not automatically proof of scientific adequacy.",
            style="Muted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        datasets = ttk.LabelFrame(self, text="Dataset behaviour", padding=14)
        datasets.pack(fill=X, pady=6)
        ttk.Checkbutton(datasets, text="Confirm before changing datasets when results may be unsaved", variable=self.confirm_dataset_switch).pack(anchor="w", pady=3)
        ttk.Checkbutton(datasets, text="Adapt NOAA/SS3 annual catch and index data for Omega workspaces", variable=self.adapt_ss3).pack(anchor="w", pady=3)
        ttk.Checkbutton(datasets, text="Automatically preview the selected dataset in Chart Studio", variable=self.auto_load_charts).pack(anchor="w", pady=3)
        ttk.Label(
            datasets,
            text="The SS3 adapter never invents biomass or composition observations. NOAA structural and native comparisons remain in NOAA / SS3 Validation.",
            style="Muted.TLabel",
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        buttons = ttk.Frame(self, style="Shell.TFrame")
        buttons.pack(fill=X, pady=(16, 0))
        ttk.Button(buttons, text="SAVE AND APPLY", command=self.save, style="Accent.TButton").pack(side=LEFT)
        ttk.Button(buttons, text="Restore safe defaults", command=self.restore_defaults).pack(side=LEFT, padx=8)

    @staticmethod
    def _choice(parent, row: int, label: str, variable: StringVar, values: tuple[str, ...]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=22).grid(row=row, column=1, sticky="w", padx=8)

    def save(self) -> None:
        try:
            width = max(170, min(480, int(self.sidebar_width.get())))
        except ValueError:
            messagebox.showerror(APP_NAME, "Sidebar width must be a whole number from 170 to 480.")
            return
        self.sidebar_width.set(str(width))
        self.shell.profile.update(
            {
                "theme": self.shell.theme.get(),
                "density": self.shell.density.get(),
                "sidebar_width": width,
                "sidebar_position": self.sidebar_position.get(),
                "default_workload": self.default_workload.get(),
                "confirm_dataset_switch": self.confirm_dataset_switch.get(),
                "adapt_ss3_for_omega": self.adapt_ss3.get(),
                "auto_load_dataset_in_charts": self.auto_load_charts.get(),
            }
        )
        self.shell._save_profile()
        self.shell.apply_appearance(save=False)
        self.shell._apply_sidebar_position()
        self.shell._apply_preferences_to_open_workspaces()
        self.shell.status.set("Settings saved and applied.")

    def restore_defaults(self) -> None:
        self.shell.reset_standard_defaults()
        self.default_workload.set("Standard")
        self.sidebar_width.set("225")
        self.sidebar_position.set("Left")
        self.confirm_dataset_switch.set(True)
        self.adapt_ss3.set(True)
        self.auto_load_charts.set(True)


class ErrorLogView(ttk.Frame):
    def __init__(self, parent, shell: "OmegaShell") -> None:
        super().__init__(parent, padding=18)
        self.shell = shell
        title = ttk.Frame(self)
        title.pack(fill=X, pady=(0, 10))
        ttk.Label(title, text="Error Log", font=("Segoe UI", 20, "bold")).pack(side=LEFT)
        ttk.Button(title, text="CLEAR LOG", command=shell.clear_error_log).pack(side=RIGHT)
        ttk.Label(
            self,
            text="Uncaught interface errors are recorded here with a timestamp and traceback. Clearing this list does not delete model results or reports.",
            style="Muted.TLabel",
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", fill=X, pady=(0, 10))
        from tkinter import Text

        body = ttk.Frame(self)
        body.pack(fill=BOTH, expand=True)
        self.text = Text(body, wrap="word", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        self.text.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.refresh()

    def refresh(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", END)
        self.text.insert(END, "\n\n".join(self.shell.error_entries) if self.shell.error_entries else "No errors have been recorded in this session.\n")
        self.text.configure(state="disabled")


class OmegaShell:
    def __init__(self, root: Tk) -> None:
        self.root = root
        apply_window_identity(root, f"{APP_NAME} {APP_VERSION}")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = max(980, min(1500, screen_width - 70))
        window_height = max(680, min(920, screen_height - 110))
        root.geometry(f"{window_width}x{window_height}")
        root.minsize(min(1050, max(900, screen_width - 120)), min(700, max(620, screen_height - 150)))
        self.theme_manager = ThemeManager(root)
        self.dataset_library = DatasetLibrary(DATA_ROOT)
        self.profile = self._load_profile()
        self.theme = StringVar(value=str(self.profile.get("theme", "Dark")).title())
        self.density = StringVar(value=str(self.profile.get("density", "Comfortable")).title())
        self.page_title = StringVar(value="Home")
        self.dataset_text = StringVar(value="No dataset selected")
        self.quick_dataset = StringVar(value="Choose a dataset…")
        self.dataset_choices: dict[str, DatasetEntry] = {}
        self.status = StringVar(value="Ready. Workspaces open inside this window; Detach is optional.")
        self.engine_text = StringVar(value="Checking engine…")
        self.active_dataset: DatasetEntry | None = None
        self.current_mode = "home"
        self.current_app: Any | None = None
        self.frames: dict[str, tuple[ttk.Frame, Any | None]] = {}
        self.tutorial_targets: dict[str, object] = {}
        self.error_entries: list[str] = []
        self.error_log_text = StringVar(value="Error Log (0)")
        self.processing_text = StringVar(value="")
        self._loading_visible = False
        self.history: list[str] = []
        self.history_index = -1
        self.root.report_callback_exception = self._report_callback_exception
        self._build_shell()
        self.tutorial = TutorialController(self, root)
        self._restore_active_dataset()
        self.navigate("home")
        self._refresh_engine_status()

    def _build_shell(self) -> None:
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, padding=(12, 9), style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Button(header, text="←", width=4, command=self.back, style="Tool.TButton").pack(side=LEFT)
        ttk.Button(header, text="→", width=4, command=self.forward, style="Tool.TButton").pack(side=LEFT, padx=4)
        home_button = ttk.Button(header, text="Home", command=lambda: self.navigate("home"), style="Tool.TButton")
        home_button.pack(side=LEFT)
        self.tutorial_targets["nav_home"] = home_button
        ttk.Label(header, textvariable=self.page_title, style="Header.TLabel", font=("Segoe UI", 15, "bold")).pack(side=LEFT, padx=16)
        change_dataset_button = ttk.Button(header, text="BROWSE DATASETS", command=lambda: self.navigate("datasets"), style="Accent.TButton")
        change_dataset_button.pack(side=RIGHT)
        self.tutorial_targets["change_dataset"] = change_dataset_button
        ttk.Button(header, text="RESET DEFAULTS", command=self.reset_standard_defaults, style="Tool.TButton").pack(side=RIGHT, padx=6)
        self.full_auto_button = ttk.Button(header, text="FULL AUTO RUN", command=self.run_full_auto, style="Accent.TButton")
        self.full_auto_button.pack(side=RIGHT, padx=6)
        self.dataset_picker = ttk.Combobox(header, textvariable=self.quick_dataset, state="readonly", width=42)
        self.dataset_picker.pack(side=RIGHT, padx=8)
        self.dataset_picker.bind("<<ComboboxSelected>>", self._quick_dataset_selected)
        ttk.Label(header, text="DATASET", style="Header.TLabel").pack(side=RIGHT, padx=(10, 0))
        ttk.Label(header, textvariable=self.engine_text, style="Header.TLabel").pack(side=RIGHT, padx=10)
        self._refresh_dataset_picker()

        self.main_pane = AdjustablePanedWindow(
            self.root,
            orient="horizontal",
            sashwidth=7,
            sashrelief="raised",
            showhandle=True,
            opaqueresize=True,
            borderwidth=0,
        )
        self.main_pane.grid(row=1, column=0, sticky="nsew")
        sidebar_shell = ttk.Frame(self.main_pane, width=225, style="Sidebar.TFrame")
        self.sidebar_shell = sidebar_shell
        sidebar_shell.grid_propagate(False)
        sidebar_canvas = Canvas(sidebar_shell, width=208, highlightthickness=0)
        sidebar_canvas.omega_role = "sidebar"  # type: ignore[attr-defined]
        sidebar_scroll = ttk.Scrollbar(sidebar_shell, orient="vertical", command=sidebar_canvas.yview)
        sidebar = ttk.Frame(sidebar_canvas, padding=8, style="Sidebar.TFrame")
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")
        sidebar.bind("<Configure>", lambda _event: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))
        dataset_label = ttk.Label(sidebar, textvariable=self.dataset_text, wraplength=205, justify="left")

        def resize_sidebar(event) -> None:
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)
            dataset_label.configure(wraplength=max(120, event.width - 24))

        sidebar_canvas.bind("<Configure>", resize_sidebar)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        sidebar_scroll.pack(side=RIGHT, fill=Y)
        self.sidebar_canvas = sidebar_canvas

        def sidebar_mousewheel(event):
            try:
                current = getattr(event, "widget", None)
                inside_widget = False
                while current is not None:
                    if current in {sidebar_canvas, sidebar_shell, sidebar}:
                        inside_widget = True
                        break
                    current = getattr(current, "master", None)
                pointer_x = self.root.winfo_pointerx()
                pointer_y = self.root.winfo_pointery()
                left = sidebar_canvas.winfo_rootx()
                top = sidebar_canvas.winfo_rooty()
                inside = inside_widget or (
                    left <= pointer_x < left + sidebar_canvas.winfo_width()
                    and top <= pointer_y < top + sidebar_canvas.winfo_height()
                )
                if not inside:
                    return None
                direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
                sidebar_canvas.yview_scroll(direction * 3, "units")
                return "break"
            except Exception:
                return None

        self._sidebar_mousewheel = sidebar_mousewheel
        self.root.bind_all("<MouseWheel>", sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-4>", sidebar_mousewheel, add="+")
        self.root.bind_all("<Button-5>", sidebar_mousewheel, add="+")
        ttk.Label(sidebar, text="OMEGA", font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=5, pady=(4, 10))
        ttk.Label(sidebar, text="CURRENT DATASET", style="Muted.TLabel").pack(anchor="w", padx=5)
        dataset_label.pack(anchor="w", padx=5, pady=(1, 3))
        ttk.Label(sidebar, textvariable=self.engine_text, style="Muted.TLabel").pack(anchor="w", padx=5, pady=(0, 8))
        nav = [
            ("Home", "home"),
            ("Dataset Library", "datasets"),
            ("Integrated Assessment", "integrated"),
            ("Visual Parameter Lab", "parameters"),
            ("Biomass & MSE", "truthmse"),
            ("Priority Diagnostics", "priority"),
            ("Expert Workflow", "expert"),
            ("Chart Studio", "charts"),
            ("Quant Lab", "quant"),
            ("NOAA / SS3", "noaa"),
            ("Validation", "validation"),
        ]
        for label, mode in nav:
            button = ttk.Button(sidebar, text=label, command=lambda selected=mode: self.navigate(selected))
            button.pack(fill=X, pady=2)
            self.tutorial_targets.setdefault(f"nav_{mode}", button)
        ttk.Separator(sidebar).pack(fill=X, pady=10)
        ttk.Button(sidebar, text="WATCH A COMPLETE MODEL", command=lambda: self.start_tutorial(True), style="Accent.TButton").pack(fill=X, pady=3)
        ttk.Button(sidebar, text="Guided practice", command=lambda: self.start_tutorial(False)).pack(fill=X, pady=3)
        ttk.Button(sidebar, text="Detach workspace", command=self.detach_current).pack(fill=X, pady=(12, 3))
        ttk.Label(sidebar, text="Theme", style="Muted.TLabel").pack(anchor="w", pady=(12, 2))
        theme = ttk.Combobox(sidebar, textvariable=self.theme, values=("Dark", "Light", "System", "High Contrast"), state="readonly")
        theme.pack(fill=X)
        theme.bind("<<ComboboxSelected>>", lambda _event: self.apply_appearance())
        ttk.Label(sidebar, text="Display density", style="Muted.TLabel").pack(anchor="w", pady=(8, 2))
        density = ttk.Combobox(sidebar, textvariable=self.density, values=("Large Text", "Comfortable", "Compact"), state="readonly")
        density.pack(fill=X)
        density.bind("<<ComboboxSelected>>", lambda _event: self.apply_appearance())
        ttk.Separator(sidebar).pack(fill=X, pady=(12, 8))
        ttk.Button(sidebar, textvariable=self.error_log_text, command=lambda: self.navigate("errors")).pack(fill=X, pady=(0, 4))
        settings_button = ttk.Button(sidebar, text="Settings", command=lambda: self.navigate("settings"))
        settings_button.pack(fill=X, pady=(0, 4))
        self.tutorial_targets["nav_settings"] = settings_button

        self.content_shell = ttk.Frame(self.main_pane, style="Shell.TFrame")
        self.content_shell.rowconfigure(0, weight=1)
        self.content_shell.columnconfigure(0, weight=1)
        self.content_canvas = Canvas(self.content_shell, highlightthickness=0)
        self.content_ybar = ttk.Scrollbar(self.content_shell, orient="vertical", command=self.content_canvas.yview)
        self.content_xbar = ttk.Scrollbar(self.content_shell, orient="horizontal", command=self.content_canvas.xview)
        self.content_canvas.configure(yscrollcommand=self.content_ybar.set, xscrollcommand=self.content_xbar.set)
        self.content_canvas.grid(row=0, column=0, sticky="nsew")
        self.content_ybar.grid(row=0, column=1, sticky="ns")
        self.content_xbar.grid(row=1, column=0, sticky="ew")
        self.content = ttk.Frame(self.content_canvas, style="Shell.TFrame")
        self.content_window = self.content_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _event: self._sync_content_scroller())
        self.content_canvas.bind("<Configure>", lambda _event: self._sync_content_scroller())
        self.main_pane.add(sidebar_shell, minsize=170, stretch="never")
        self.main_pane.add(self.content_shell, minsize=520, stretch="always")
        self.main_pane.bind("<ButtonRelease-1>", self._remember_sidebar_width)
        self.root.after(120, self._apply_sidebar_position)
        status = ttk.Frame(self.root, padding=(10, 5), style="Card.TFrame")
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status, style="Status.TLabel").grid(row=0, column=0, sticky="ew")
        ttk.Label(status, text=f"Version {APP_VERSION}", style="Status.TLabel").grid(row=0, column=1, sticky="e")
        self.processing_label = ttk.Label(status, textvariable=self.processing_text, style="Status.TLabel")
        self.processing_bar = ttk.Progressbar(status, mode="indeterminate")
        self.processing_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.processing_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.processing_label.grid_remove()
        self.processing_bar.grid_remove()
        self.root.bind_all("<MouseWheel>", self._main_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._main_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._main_mousewheel, add="+")
        self.root.bind_all("<Button-3>", self._show_context_menu, add="+")
        self.apply_appearance()
        self.root.after(200, self._poll_processing)

    def _poll_processing(self) -> None:
        """Mirror active workspace work in a persistent bottom loading bar."""

        try:
            app = self.current_app
            app_status = str(app.status.get()) if app is not None and hasattr(app, "status") else ""
            busy_value = getattr(app, "busy", False) if app is not None else False
            busy = bool(busy_value.get()) if hasattr(busy_value, "get") else bool(busy_value)
            combined = app_status or self.status.get()
            working_prefixes = (
                "running", "starting", "preparing", "loading", "downloading", "building",
                "fitting", "optimizing", "optimising", "analyzing", "analysing", "generating",
            )
            working = busy or combined.strip().lower().startswith(working_prefixes)
            if working and not self._loading_visible:
                self.processing_label.grid()
                self.processing_bar.grid()
                self.processing_bar.start(12)
                self._loading_visible = True
            elif not working and self._loading_visible:
                self.processing_bar.stop()
                self.processing_label.grid_remove()
                self.processing_bar.grid_remove()
                self._loading_visible = False
            if working:
                self.processing_text.set(combined)
            if self.root.winfo_exists():
                self.root.after(200, self._poll_processing)
        except Exception:
            return

    def _show_context_menu(self, event) -> str | None:
        """Show actions for the control and workspace under the pointer."""

        widget = getattr(event, "widget", None)
        if widget is None:
            return None
        menu = Menu(self.root, tearoff=False)
        widget_class = str(widget.winfo_class())
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}:
            if widget_class not in {"TCombobox"}:
                menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"))
            menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
            if widget_class not in {"TCombobox"}:
                menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))
            menu.add_separator()
            menu.add_command(label="Select all", command=lambda: widget.event_generate("<<SelectAll>>"))
        elif widget_class == "Treeview":
            selected = widget.selection()
            if selected:
                values = widget.item(selected[0], "values")
                menu.add_command(label="Copy selected row", command=lambda: self._copy_to_clipboard("\t".join(map(str, values))))
            menu.add_command(label="Copy current status", command=lambda: self._copy_to_clipboard(self.status.get()))
        else:
            app = self.current_app

            def add(label: str, method: str) -> None:
                action = getattr(app, method, None) if app is not None else None
                if callable(action):
                    menu.add_command(label=label, command=action)

            if self.current_mode == "parameters":
                add("Reset realistic defaults", "reset_realistic_defaults")
                add("Choose another dataset", "choose_dataset")
            elif self.current_mode == "charts":
                add("Update live chart", "update_live_preview")
                add("Open interactive chart", "preview")
                add("Reset chart profile", "reset_profile")
            elif self.current_mode == "quant":
                add("Run global optimization", "run_optimizer")
                add("Run optimization grid", "run_surface")
                add("Export Quant results", "export_results")
            elif self.current_mode == "integrated":
                add("Run assessment fit", "run_fit")
                add("Run projection", "run_projection")
                add("Export assessment package", "export_package")
            elif self.current_mode == "priority":
                add("Run likelihood profile", "run_profile")
                add("Run ASPM diagnostic", "run_aspm")
                add("Run coverage diagnostic", "run_coverage")
                add("Open diagnostic reports", "open_reports")
            elif self.current_mode == "expert":
                add("Run complete expert workflow", "run")
                add("Open evidence dashboard", "open_dashboard")
                add("Open Chart Studio", "open_chart_studio")
            elif self.current_mode == "noaa":
                add("Run NOAA comparison", "run_validation")
                add("Export validation report", "export_report")
                add("Open report folder", "open_report_folder")
            elif self.current_mode == "datasets":
                view = self.frames.get("datasets", (None, None))[0]
                if isinstance(view, DatasetLibraryView):
                    menu.add_command(label="Refresh dataset list", command=view.refresh)
                    menu.add_command(label="Load selected dataset", command=view.load_selected)
                    menu.add_command(label="Open selected dataset folder", command=view.open_selected_folder)
            elif self.current_mode == "home":
                menu.add_command(label="Watch complete model tutorial", command=lambda: self.start_tutorial(True))
                menu.add_command(label="Browse datasets", command=lambda: self.navigate("datasets"))
                menu.add_command(label="Run full automatic workflow", command=self.run_full_auto)
            if menu.index("end") is not None:
                menu.add_separator()
            menu.add_command(label="Open Settings", command=lambda: self.navigate("settings"))
            menu.add_command(label="Open Error Log", command=lambda: self.navigate("errors"))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def _copy_to_clipboard(self, value: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(value)

    def _sync_content_scroller(self) -> None:
        try:
            self.content_canvas.update_idletasks()
            viewport_width = max(1, self.content_canvas.winfo_width())
            viewport_height = max(1, self.content_canvas.winfo_height())
            requested_width = max(1, self.content.winfo_reqwidth())
            requested_height = max(1, self.content.winfo_reqheight())
            width = max(viewport_width, requested_width)
            height = max(viewport_height, requested_height)
            self.content_canvas.itemconfigure(self.content_window, width=width, height=height)
            self.content_canvas.configure(scrollregion=(0, 0, width, height))
            if requested_height > viewport_height + 2:
                self.content_ybar.grid()
            else:
                self.content_ybar.grid_remove()
            if requested_width > viewport_width + 2:
                self.content_xbar.grid()
            else:
                self.content_xbar.grid_remove()
        except Exception:
            pass

    def _main_mousewheel(self, event):
        try:
            widget = getattr(event, "widget", None)
            current = widget
            while current is not None:
                if current is not self.content_canvas and (
                    current.winfo_class() in {"Treeview", "Text", "Listbox", "Canvas"}
                    or getattr(current, "omega_role", "") in {"controls", "workspace_controls", "sidebar"}
                ):
                    return None
                current = getattr(current, "master", None)
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
            left = self.content_canvas.winfo_rootx()
            top = self.content_canvas.winfo_rooty()
            if not (left <= x < left + self.content_canvas.winfo_width() and top <= y < top + self.content_canvas.winfo_height()):
                return None
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
            self.content_canvas.yview_scroll(direction * 3, "units")
            return "break"
        except Exception:
            return None

    def _apply_sidebar_position(self) -> None:
        position = str(self.profile.get("sidebar_position", "Left")).title()
        if position not in {"Left", "Top", "Right", "Bottom"}:
            position = "Left"
        try:
            for pane in tuple(self.main_pane.panes()):
                self.main_pane.forget(pane)
            self.main_pane.configure(orient="horizontal" if position in {"Left", "Right"} else "vertical")
            if position in {"Left", "Top"}:
                self.main_pane.add(self.sidebar_shell, minsize=170, stretch="never")
                self.main_pane.add(self.content_shell, minsize=520 if position == "Left" else 340, stretch="always")
            else:
                self.main_pane.add(self.content_shell, minsize=520 if position == "Right" else 340, stretch="always")
                self.main_pane.add(self.sidebar_shell, minsize=170, stretch="never")
            self.profile["sidebar_position"] = position
            self.root.after(60, self._restore_sidebar_width)
        except Exception as exc:
            self.status.set(f"Sidebar position could not be changed: {exc}")

    def _restore_sidebar_width(self) -> None:
        try:
            requested = int(self.profile.get("sidebar_width", 225))
        except (TypeError, ValueError):
            requested = 225
        try:
            position = str(self.profile.get("sidebar_position", "Left")).title()
            horizontal = position in {"Left", "Right"}
            total = self.main_pane.winfo_width() if horizontal else self.main_pane.winfo_height()
            reserve = 520 if horizontal else 340
            maximum = max(170, total - reserve)
            size = max(170, min(requested, maximum))
            sash = size if position in {"Left", "Top"} else total - size
            self.main_pane.sashpos(0, sash)
        except Exception:
            pass

    def _remember_sidebar_width(self, _event=None) -> None:
        try:
            position = str(self.profile.get("sidebar_position", "Left")).title()
            horizontal = position in {"Left", "Right"}
            total = self.main_pane.winfo_width() if horizontal else self.main_pane.winfo_height()
            reserve = 520 if horizontal else 340
            maximum = max(170, total - reserve)
            sash = int(self.main_pane.sashpos(0))
            size = sash if position in {"Left", "Top"} else total - sash
            size = max(170, min(size, maximum))
            self.main_pane.sashpos(0, size if position in {"Left", "Top"} else total - size)
            self.profile["sidebar_width"] = size
            self._save_profile()
        except Exception:
            pass

    def _refresh_dataset_picker(self) -> None:
        entries = self.dataset_library.scan()
        choices: dict[str, DatasetEntry] = {}
        for entry in entries:
            base = f"{entry.source} — {entry.display_name} — {entry.coverage}"
            label = base
            suffix = 2
            while label in choices:
                label = f"{base} ({suffix})"
                suffix += 1
            choices[label] = entry
        self.dataset_choices = choices
        self.dataset_picker.configure(values=tuple(choices))
        if self.active_dataset is not None:
            selected = next((label for label, entry in choices.items() if entry.identifier == self.active_dataset.identifier), None)
            if selected:
                self.quick_dataset.set(selected)

    def _quick_dataset_selected(self, _event=None) -> None:
        entry = self.dataset_choices.get(self.quick_dataset.get())
        if entry is None:
            return
        self.set_active_dataset(entry)
        if entry.model_type.lower() == "stock synthesis":
            self.navigate("noaa")
            app = self.current_app
            if app is not None:
                app.model_var.set(entry.root.name)
                app.folder_var.set(str(entry.root))
                app.status.set(f"Ready to run and compare the full NOAA/SS3 model: {entry.display_name}.")
        else:
            self.status.set(f"Selected {entry.display_name}. Open any compatible Omega workspace to use it.")

    def _home_view(self) -> ttk.Frame:
        frame = ttk.Frame(self.content, padding=22, style="Shell.TFrame")
        ttk.Label(frame, text="What would you like to do?", font=("Segoe UI", 24, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Start with the watchable tutorial, choose a dataset, or open an assessment workspace. Everything stays in this window unless you select Detach.",
            style="Muted.TLabel",
            wraplength=1050,
        ).pack(anchor="w", pady=(4, 18))
        quick = ttk.Frame(frame, style="Shell.TFrame")
        quick.pack(fill=X, pady=(0, 14))
        ttk.Button(quick, text="WATCH OMEGA RUN A COMPLETE MODEL", command=lambda: self.start_tutorial(True), style="Accent.TButton").pack(side=LEFT, padx=(0, 8))
        ttk.Button(quick, text="RUN MY FIRST MODEL", command=self.run_first_model).pack(side=LEFT, padx=8)
        ttk.Button(quick, text="CHANGE DATASET", command=lambda: self.navigate("datasets")).pack(side=LEFT, padx=8)

        grid = ttk.Frame(frame, style="Shell.TFrame")
        grid.pack(fill=BOTH, expand=True)
        for column in range(3):
            grid.columnconfigure(column, weight=1)
        for row in range(5):
            grid.rowconfigure(row, weight=1)
        cards = [
            ("Integrated Assessment", "• Biomass and depletion\n• Fishing mortality and recruitment\n• Age/length composition fit\n• Projections and sector results", "integrated"),
            ("Visual Parameter Lab", "• Natural and fishing mortality\n• Recruitment strength and variability\n• Growth, carrying capacity and depletion\n• Catchability and observation error", "parameters"),
            ("Best-supported Biomass", "• Compare biomass evidence\n• Separate operating-model truths\n• Management strategy evaluation\n• Risk and trade-off summaries", "truthmse"),
            ("Diagnostic Control", "• Refit likelihood profiles\n• ASPM comparison\n• Interval coverage\n• Native-engine parity", "priority"),
            ("Automatic Expert Workflow", "• Convergence and residual checks\n• Retrospective and hindcast tests\n• Influence and recovery checks\n• Reliability verdicts", "expert"),
            ("Interactive Charts", "• Time series and uncertainty\n• Diagnostic and residual plots\n• Optimization parameter grids\n• Interactive HTML dashboards", "charts"),
            ("Quant Lab", "• Global and multi-optimizer analysis\n• Parameter grids and identifiability\n• Stress and sensitivity tests\n• Ensembles and risk frontiers", "quant"),
            ("NOAA Validation", "• Official SS3 configurations\n• Parser and parity checks\n• Reproducible answer comparison\n• Preserved validation evidence", "noaa"),
            ("Validation & MSE", "• Deterministic benchmarks\n• CPUE standardisation\n• Tagging and reliability examples\n• Legacy MSE checks", "validation"),
            ("Settings", "• Sidebar position and display density\n• Standard workload defaults\n• Dataset adapter preferences\n• Restore realistic defaults", "settings"),
        ]
        for index, (title, description, mode) in enumerate(cards):
            card = ttk.Frame(grid, padding=16, style="Card.TFrame")
            card.grid(row=index // 3, column=index % 3, sticky="nsew", padx=7, pady=7)
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            ttk.Label(card, text=description, style="CardText.TLabel", wraplength=320, justify="left").pack(anchor="w", fill=X, expand=True, pady=(7, 12))
            ttk.Button(card, text="Open", command=lambda selected=mode: self.navigate(selected)).pack(anchor="e")

        note = ttk.Frame(grid, padding=16, style="Card.TFrame")
        note.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=7, pady=7)
        ttk.Label(note, text="Scientific boundary", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(
            note,
            text="Omega estimates the biomass best supported by supplied data and assumptions. It cannot reveal assumption-free true biomass, and a completed run is not equivalent to independent scientific validation.",
            style="CardText.TLabel",
            wraplength=1050,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))
        return frame

    def _new_frame(self, mode: str) -> tuple[ttk.Frame, Any | None]:
        if mode == "home":
            return self._home_view(), None
        if mode == "datasets":
            return DatasetLibraryView(self.content, self), None
        if mode == "settings":
            return SettingsView(self.content, self), None
        if mode == "errors":
            return ErrorLogView(self.content, self), None
        if mode in WORKSPACES:
            frame = EmbeddedWorkspace(self.content)
            frame.omega_shell = self  # type: ignore[attr-defined]
            app = create_workspace(mode, frame)
            return frame, app
        raise ValueError(f"Unknown page: {mode}")

    def navigate(self, mode: str, *, record: bool = True) -> None:
        if mode == self.current_mode and mode in self.frames:
            return
        if self.current_mode in self.frames:
            self.frames[self.current_mode][0].pack_forget()
        if mode not in self.frames:
            self.frames[mode] = self._new_frame(mode)
        frame, app = self.frames[mode]
        frame.pack(fill=BOTH, expand=True)
        self.content_canvas.xview_moveto(0.0)
        self.content_canvas.yview_moveto(0.0)
        self.current_mode = mode
        self.current_app = app
        self.page_title.set(
            "Dataset Library"
            if mode == "datasets"
            else "Settings"
            if mode == "settings"
            else "Error Log"
            if mode == "errors"
            else "Home"
            if mode == "home"
            else WORKSPACES[mode][0]
        )
        if record:
            self.history = self.history[: self.history_index + 1]
            self.history.append(mode)
            self.history_index = len(self.history) - 1
        if app is not None:
            self._inject_active_dataset(app)
        self.apply_appearance(save=False)
        self.root.after_idle(self._sync_content_scroller)
        self.status.set(f"Opened {self.page_title.get()} inside the main Omega window.")
        if hasattr(self, "tutorial"):
            self.tutorial.panel.lift()

    def log_error(self, context: str, error: BaseException, detail: str | None = None) -> None:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        trace = detail or "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.error_entries.append(f"[{stamp}] {context}\n{trace.strip()}")
        self.error_log_text.set(f"Error Log ({len(self.error_entries)})")
        view = self.frames.get("errors")
        if view and isinstance(view[0], ErrorLogView):
            view[0].refresh()

    def clear_error_log(self) -> None:
        self.error_entries.clear()
        self.error_log_text.set("Error Log (0)")
        view = self.frames.get("errors")
        if view and isinstance(view[0], ErrorLogView):
            view[0].refresh()
        self.status.set("Session error log cleared. Model results and reports were not changed.")

    def _report_callback_exception(self, exc_type, exc_value, exc_traceback) -> None:
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        self.log_error("User-interface callback", exc_value, detail)
        self.status.set(f"An interface error was recorded: {exc_value}")
        try:
            messagebox.showerror(APP_NAME, f"{exc_value}\n\nThe details were added to Error Log in the sidebar.")
        except Exception:
            pass

    def back(self) -> None:
        if self.history_index > 0:
            self.history_index -= 1
            self.navigate(self.history[self.history_index], record=False)

    def forward(self) -> None:
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.navigate(self.history[self.history_index], record=False)

    def detach_current(self) -> None:
        if self.current_mode not in WORKSPACES:
            messagebox.showinfo(APP_NAME, "Open a model workspace before selecting Detach.")
            return
        try:
            command = [sys.executable, str(SOURCE_ROOT / "omega_desktop.py"), "--mode", self.current_mode]
            subprocess.Popen(command, cwd=str(APP_DIR))
            self.status.set(f"Detached {WORKSPACES[self.current_mode][0]}. The main window remains open.")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def apply_appearance(self, *, save: bool = True) -> None:
        self.theme_manager.apply(self.theme.get(), self.density.get())
        if save:
            self.profile["theme"] = self.theme.get()
            self.profile["density"] = self.density.get()
            self._save_profile()

    def set_active_dataset(self, entry: DatasetEntry) -> None:
        if (
            self.active_dataset is not None
            and self.active_dataset.identifier != entry.identifier
            and self._has_unsaved_results()
            and bool(self.profile.get("confirm_dataset_switch", True))
        ):
            proceed = messagebox.askyesno(
                "Change dataset",
                "The current workspace contains results that may not have been exported. Change dataset and clear incompatible results?",
            )
            if not proceed:
                return
        self.active_dataset = entry
        self.dataset_text.set(entry.display_name)
        selected = next((label for label, candidate in self.dataset_choices.items() if candidate.identifier == entry.identifier), None)
        if selected:
            self.quick_dataset.set(selected)
        self.profile["active_dataset"] = entry.identifier
        self._save_profile()
        for _mode, (_frame, app) in self.frames.items():
            if app is not None:
                self._inject_active_dataset(app, clear_results=True)
        self.status.set(f"Current dataset: {entry.display_name}. Original files remain unchanged.")

    def load_beginner_dataset(self) -> DatasetEntry | None:
        entries = self.dataset_library.scan()
        compatible = [item for item in entries if item.difficulty.lower() == "beginner" and item.primary_file is not None]
        if not compatible:
            messagebox.showwarning(APP_NAME, "No model-ready beginner dataset was found in Data_Sets.")
            return None
        preferred = next((item for item in compatible if "demo" in item.display_name.lower()), compatible[0])
        self.set_active_dataset(preferred)
        return preferred

    def run_first_model(self) -> None:
        if self.active_dataset is None:
            self.load_beginner_dataset()
        self.navigate("integrated")

    def run_full_auto(self) -> None:
        """Load available inputs and start every applicable expert check."""

        if self.active_dataset is None:
            entries = self.dataset_library.scan()
            preferred = next((item for item in entries if item.identifier == "omega-diagnostics-reference"), None)
            preferred = preferred or next((item for item in entries if item.primary_file is not None), None)
            if preferred is None:
                messagebox.showwarning(APP_NAME, "No model-ready dataset is available. Open Dataset Library and add or select a dataset first.")
                return
            self.set_active_dataset(preferred)
        self.navigate("expert")
        app = self.current_app
        if app is None:
            messagebox.showerror(APP_NAME, "The Automatic Expert Workflow could not be opened.")
            return
        self._inject_active_dataset(app)
        workload = str(self.profile.get("default_workload", "Standard")).lower()
        app.mode.set("automatic")
        app.speed.set("deep" if workload == "formal" else workload if workload in {"quick", "standard"} else "standard")
        app.skip_steps.set("")
        app.override_reason.set("")
        dataset_name = self.active_dataset.display_name if self.active_dataset is not None else "selected dataset"
        app.status.set(f"Preparing all applicable checks for {dataset_name}...")
        self.status.set(
            f"Full automatic analysis started for {dataset_name}. Omega will run each check supported by the available inputs and keep failures visible."
        )
        app.run()

    def refresh_noaa_library(self) -> None:
        if not messagebox.askyesno(
            "Download official NOAA test data",
            "Download or refresh the official NOAA Stock Synthesis test-model and user-example libraries? This may take several minutes and uses local disk space.",
        ):
            return
        self.status.set("Downloading official NOAA test data and building the catalogue…")

        def worker() -> None:
            command = [sys.executable, str(SOURCE_ROOT / "tools" / "download_noaa_test_data.py"), "--refresh"]
            completed = subprocess.run(command, cwd=str(SOURCE_ROOT), capture_output=True, text=True)

            def finish() -> None:
                if completed.returncode == 0:
                    try:
                        payload = json.loads(completed.stdout)
                        summary = f"NOAA library ready: {payload.get('models', 0)} models and {payload.get('files', 0)} files."
                    except json.JSONDecodeError:
                        summary = "NOAA library downloaded and catalogued."
                    self.status.set(summary)
                    view = self.frames.get("datasets")
                    if view and isinstance(view[0], DatasetLibraryView):
                        view[0].refresh()
                    self._refresh_dataset_picker()
                    messagebox.showinfo("NOAA Test Model Library", summary)
                else:
                    detail = (completed.stderr or completed.stdout or "Unknown download error")[-3000:]
                    self.status.set("NOAA download failed. Existing datasets were not removed.")
                    messagebox.showerror("NOAA download failed", detail)

            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _inject_active_dataset(self, app: Any, *, clear_results: bool = False) -> None:
        entry = self.active_dataset
        if entry is None:
            return
        self._apply_workspace_preferences(app)
        if getattr(app, "_omega_dataset_id", None) == entry.identifier and not clear_results:
            return
        app._omega_dataset_id = entry.identifier
        class_name = app.__class__.__name__
        if class_name == "NOAAValidationApp":
            if entry.model_type.lower() == "stock synthesis" and (entry.root / "starter.ss").exists():
                app.model_var.set(entry.root.name)
                app.folder_var.set(str(entry.root))
                app.status.set(f"Loaded full NOAA/SS3 source model: {entry.display_name}. Run NOAA DATA + COMPARE when ready.")
            else:
                app.status.set("The active Omega dataset is available to Omega analysis workspaces. NOAA validation requires a Stock Synthesis starter/data/control folder.")
            return
        if class_name == "CompleteAssessmentApp":
            app.status.set("Validation & Legacy MSE uses controlled built-in simulations; the selected dataset remains active for all data-driven workspaces.")
            return
        if entry.primary_file is None and not bool(self.profile.get("adapt_ss3_for_omega", True)):
            if hasattr(app, "status"):
                app.status.set("This is an SS3 dataset. Enable the transparent SS3 annual-data adapter in Settings to use it here.")
            return
        try:
            primary_file, source_note = materialize_omega_timeseries(entry, PROFILE_DIR / "derived_datasets")
        except Exception as exc:
            if hasattr(app, "status"):
                app.status.set(f"Dataset could not be prepared for this workspace: {exc}")
            return
        path_text = str(primary_file)
        if hasattr(app, "dataset_path"):
            app.dataset_path.set(path_text)
        if hasattr(app, "age_path"):
            app.age_path.set(str(entry.age_composition) if entry.age_composition is not None and entry.primary_file is not None else "")
        if hasattr(app, "length_path"):
            app.length_path.set(str(entry.length_composition) if entry.length_composition is not None and entry.primary_file is not None else "")
        if class_name == "IntegratedAssessmentApp":
            try:
                from stock_model.age_structured import read_age_structured_file, read_composition_file

                app.dataset = read_age_structured_file(primary_file)
                app.age_composition = read_composition_file(entry.age_composition) if entry.age_composition and entry.primary_file is not None else None
                app.length_composition = read_composition_file(entry.length_composition) if entry.length_composition and entry.primary_file is not None else None
                if clear_results:
                    app.result = None
                    app.simulation = None
                    app.projection = None
                    app.mse = None
                app._populate_tree(app.data_tree, app.dataset.frame.where(app.dataset.frame.notna(), "").to_dict(orient="records"))
                app._show_loaded_compositions()
                app.status.set(f"Loaded from Dataset Library: {entry.display_name} ({source_note}). Building the deterministic baseline...")
                app.notebook.select(app.data_tab)
                app.run_simulation()
            except Exception as exc:
                self.status.set(f"Dataset selected, but Integrated Assessment could not read it: {exc}")
        elif class_name == "VisualParameterLabApp":
            app.load_dataset_path(primary_file, f"Dataset Library: {entry.display_name} ({source_note})")
        elif class_name == "QuantLabApp":
            try:
                from stock_model.data_io import read_stock_file

                app.dataset = read_stock_file(primary_file)
                if clear_results:
                    for name in (
                        "fit_result",
                        "optimizer_output",
                        "walk_forward_output",
                        "optimizer_agreement_output",
                        "model_ensemble_output",
                        "risk_output",
                        "stress_output",
                        "sobol_output",
                        "regime_output",
                        "full_output",
                    ):
                        if hasattr(app, name):
                            setattr(app, name, None)
                if hasattr(app, "data_tree"):
                    app._populate_tree(app.data_tree, app.dataset.frame.head(500).where(app.dataset.frame.notna(), "").to_dict(orient="records"))
                if hasattr(app, "status"):
                    app.status.set(f"Loaded from Dataset Library: {entry.display_name} ({source_note}).")
            except Exception as exc:
                self.status.set(f"Dataset selected, but Quant Lab could not read it: {exc}")
        elif class_name == "ChartStudioApp":
            if not bool(self.profile.get("auto_load_dataset_in_charts", True)):
                app.status.set("Dataset preview is disabled in Settings. Use Load CSV / JSON to choose chart data.")
                return
            try:
                import pandas as pd

                chart_file = entry.root / "all_functions_chart_data.csv"
                chart_source = chart_file if chart_file.exists() else primary_file
                app._set_frame(pd.read_csv(chart_source), chart_source)
                app.title.set(f"{entry.display_name} — {'complete analysis examples' if chart_file.exists() else 'input data'}")
                app.status.set(f"Previewing {entry.display_name} ({source_note}). Choose numeric columns and open the interactive preview.")
            except Exception as exc:
                app.status.set(f"The selected dataset could not be previewed: {exc}")
        elif hasattr(app, "status"):
            app.status.set(f"Loaded from Dataset Library: {entry.display_name} ({source_note}).")

    def _apply_workspace_preferences(self, app: Any, *, force: bool = False) -> None:
        if getattr(app, "_omega_preferences_applied", False) and not force:
            return
        workload = str(self.profile.get("default_workload", "Standard")).lower()
        if hasattr(app, "analysis_level"):
            try:
                app.analysis_level.set(workload)
                if hasattr(app, "apply_analysis_level"):
                    app.apply_analysis_level()
                elif hasattr(app, "apply_preset"):
                    app.apply_preset()
            except Exception:
                pass
        if app.__class__.__name__ == "ExpertWorkflowApp" and hasattr(app, "speed"):
            app.speed.set("deep" if workload == "formal" else workload)
            app.mode.set("automatic")
            app.model.set("schaefer")
        elif app.__class__.__name__ == "IntegratedAssessmentApp":
            settings = {
                "quick": ("12", "1", "80"),
                "standard": ("36", "24", "400"),
                "formal": ("72", "60", "1000"),
            }[workload if workload in {"quick", "standard", "formal"} else "standard"]
            app.fit_population.set(settings[0])
            app.fit_generations.set(settings[1])
            app.projection_iterations.set(settings[2])
            app.projection_years.set("20" if workload != "formal" else "30")
        elif app.__class__.__name__ == "VisualParameterLabApp":
            app.reset_realistic_defaults()
        elif app.__class__.__name__ == "QuantLabApp":
            settings = {
                "quick": ("24", "8", "120", "100"),
                "standard": ("48", "35", "300", "300"),
                "formal": ("96", "80", "1000", "1000"),
            }[workload if workload in {"quick", "standard", "formal"} else "standard"]
            app.population.set(settings[0])
            app.generations.set(settings[1])
            app.search_draws.set(settings[2])
            app.projection_iterations.set(settings[3])
            app.projection_years.set("20" if workload != "formal" else "30")
            app.model.set("schaefer")
            app.algorithm.set("differential_evolution")
        app._omega_preferences_applied = True

    def _apply_preferences_to_open_workspaces(self) -> None:
        for _mode, (_frame, app) in self.frames.items():
            if app is not None:
                self._apply_workspace_preferences(app, force=True)

    def reset_standard_defaults(self) -> None:
        self.theme.set("Dark")
        self.density.set("Comfortable")
        self.profile.update(
            {
                "theme": "Dark",
                "density": "Comfortable",
                "sidebar_width": 225,
                "sidebar_position": "Left",
                "default_workload": "Standard",
                "confirm_dataset_switch": True,
                "adapt_ss3_for_omega": True,
                "auto_load_dataset_in_charts": True,
            }
        )
        self._save_profile()
        self.apply_appearance(save=False)
        self._apply_sidebar_position()
        self._apply_preferences_to_open_workspaces()
        for _mode, (_frame, app) in self.frames.items():
            if app is not None:
                self._inject_active_dataset(app)
        settings_frame = self.frames.get("settings")
        if settings_frame and isinstance(settings_frame[0], SettingsView):
            view = settings_frame[0]
            view.default_workload.set("Standard")
            view.sidebar_width.set("225")
            view.sidebar_position.set("Left")
            view.confirm_dataset_switch.set(True)
            view.adapt_ss3.set(True)
            view.auto_load_charts.set(True)
        self.status.set("Standard realistic defaults restored for all workspaces. The active dataset was not changed.")

    def _has_unsaved_results(self) -> bool:
        app = self.current_app
        if app is None:
            return False
        return any(getattr(app, name, None) is not None for name in ("result", "output", "simulation", "projection", "mse", "baseline", "last_json"))

    def start_tutorial(self, automatic: bool) -> None:
        self.tutorial.start(automatic=automatic)

    def prepare_tutorial_target(self, action: str) -> TutorialTarget | None:
        navigation = {
            "home": ("nav_home", "Click Home"),
            "datasets": ("nav_datasets", "Click Dataset Library"),
            "integrated": ("nav_integrated", "Click Integrated Assessment"),
            "priority": ("nav_priority", "Click Priority Diagnostics"),
            "mse": ("nav_truthmse", "Click Biomass & MSE"),
        }
        if action in navigation:
            key, instruction = navigation[action]
            widget = self.tutorial_targets.get(key)
            return TutorialTarget(widget, instruction) if widget is not None else None
        if action == "load_beginner":
            self.navigate("datasets")
            view = self.frames["datasets"][0]
            if not isinstance(view, DatasetLibraryView):
                return None
            return TutorialTarget(view.beginner_button, "Click Load beginner dataset")
        if action == "configure_quick_fit":
            return None
        if action == "run_fit":
            app = self._integrated_app()
            app.result = None
            return TutorialTarget(app.run_fit_button, "Click Fit integrated model")
        if action in {"show_biomass", "show_diagnostics"}:
            app = self._integrated_app()
            tab = app.history_tab if action == "show_biomass" else app.diagnostics_tab
            label = "Click the Biomass and F tab" if action == "show_biomass" else "Click the Fit Diagnostics tab"
            return TutorialTarget(
                app.notebook,
                label,
                bounds=lambda notebook=app.notebook, selected_tab=tab: self._notebook_tab_bounds(notebook, selected_tab),
            )
        return None

    def verify_tutorial_action(self, action: str, finished: Callable[[bool, str], None]) -> None:
        if action == "home":
            finished(self.current_mode == "home", "Home displayed." if self.current_mode == "home" else "Click the highlighted Home control.")
        elif action == "datasets":
            finished(self.current_mode == "datasets", "Dataset Library displayed." if self.current_mode == "datasets" else "Click Dataset Library to continue.")
        elif action == "load_beginner":
            loaded = self.active_dataset is not None and self.active_dataset.difficulty.lower() == "beginner"
            detail = f"Loaded {self.active_dataset.display_name}." if loaded and self.active_dataset else "The beginner dataset has not been loaded yet."
            finished(loaded, detail)
        elif action == "integrated":
            finished(self.current_mode == "integrated", "Integrated Assessment displayed." if self.current_mode == "integrated" else "Click Integrated Assessment to continue.")
        elif action == "run_fit":
            self._wait_for_fit(self._integrated_app(), finished, attempts=0)
        elif action in {"show_biomass", "show_diagnostics"}:
            app = self._integrated_app()
            expected = app.history_tab if action == "show_biomass" else app.diagnostics_tab
            selected = app.notebook.select() == str(expected)
            detail = "Biomass and fishing mortality are displayed." if action == "show_biomass" else "Fit diagnostics are displayed."
            finished(selected, detail if selected else "Click the highlighted tab, not another tab.")
        elif action == "priority":
            finished(self.current_mode == "priority", "Priority Diagnostics displayed." if self.current_mode == "priority" else "Click Priority Diagnostics to continue.")
        elif action == "mse":
            finished(self.current_mode == "truthmse", "Biomass & MSE displayed." if self.current_mode == "truthmse" else "Click Biomass & MSE to continue.")
        else:
            finished(False, f"The guide cannot verify action: {action}")

    @staticmethod
    def _notebook_tab_bounds(notebook, tab) -> tuple[int, int, int, int]:
        notebook.update_idletasks()
        x, y, width, height = notebook.bbox(tab)
        return notebook.winfo_rootx() + x, notebook.winfo_rooty() + y, width, height

    def perform_tutorial_action(self, action: str, finished: Callable[[bool, str], None]) -> None:
        try:
            if action == "home":
                self.navigate("home")
                self.root.after(250, lambda: finished(True, "Home dashboard displayed."))
            elif action == "datasets":
                self.navigate("datasets")
                self.root.after(250, lambda: finished(True, "Dataset Library displayed."))
            elif action == "load_beginner":
                entry = self.load_beginner_dataset()
                finished(entry is not None, f"Loaded {entry.display_name}." if entry else "No compatible beginner dataset was found.")
            elif action == "integrated":
                self.navigate("integrated")
                self.root.after(350, lambda: finished(True, "Integrated Assessment opened with the active dataset."))
            elif action == "configure_quick_fit":
                app = self._integrated_app()
                app.fit_population.set("12")
                app.fit_generations.set("1")
                app.max_age.set("10")
                app.projection_iterations.set("80")
                app.tutorial_quick_fit = True
                finished(True, "Short teaching fit selected: fixed biology, maximum age 10, and one estimated depletion parameter.")
            elif action == "run_fit":
                app = self._integrated_app()
                app.result = None
                app.tutorial_quick_fit = False
                from dataclasses import replace
                from stock_model.age_structured import AgeFitSettings, fit_age_structured

                dataset = app._require_dataset()
                settings = replace(app._settings(), max_age=10)
                teaching_fit = AgeFitSettings(
                    population=12,
                    generations=1,
                    local_rounds=1,
                    estimate_natural_mortality=False,
                    estimate_steepness=False,
                    estimate_initial_depletion=True,
                    estimate_survey_selectivity=False,
                    estimate_recruitment_sigma=False,
                    seed=8301,
                )
                app._run_background(
                    "Running live teaching fit...",
                    lambda: fit_age_structured(dataset, settings, teaching_fit, None, None),
                    app._show_fit,
                )
                self._wait_for_fit(app, finished, attempts=0)
            elif action == "show_biomass":
                app = self._integrated_app()
                if app.result is None:
                    finished(False, "The model fit has not completed, so biomass cannot be shown.")
                else:
                    app.notebook.select(app.history_tab)
                    finished(True, "Biomass, depletion, and fishing mortality are displayed from the completed fit.")
            elif action == "show_diagnostics":
                app = self._integrated_app()
                app.notebook.select(app.diagnostics_tab)
                finished(True, "Fit diagnostics displayed. Treat warnings as evidence to investigate, not decorations.")
            elif action == "priority":
                self.navigate("priority")
                self.root.after(350, lambda: finished(True, "Priority diagnostics workspace displayed."))
            elif action == "mse":
                self.navigate("truthmse")
                self.root.after(350, lambda: finished(True, "Advanced MSE workspace displayed. No long MSE run was started automatically."))
            else:
                finished(False, f"Unknown tutorial action: {action}")
        except Exception as exc:
            finished(False, f"Tutorial paused because the live action failed: {exc}")

    def _integrated_app(self):
        self.navigate("integrated")
        app = self.frames["integrated"][1]
        if app is None:
            raise RuntimeError("Integrated Assessment is unavailable.")
        if app.dataset is None:
            entry = self.load_beginner_dataset()
            if entry is None:
                raise RuntimeError("No beginner dataset is available.")
            self._inject_active_dataset(app)
        return app

    def _wait_for_fit(self, app: Any, finished: Callable[[bool, str], None], attempts: int) -> None:
        if app.result is not None:
            finished(True, "The live age-structured fit completed. Omega retained the fitted parameters and diagnostics.")
            return
        status = str(app.status.get())
        if status.lower().startswith("failed"):
            finished(False, status)
            return
        if attempts >= 120:
            finished(False, "The teaching fit did not finish within one minute. The tutorial paused without hiding the run.")
            return
        self.root.after(500, lambda: self._wait_for_fit(app, finished, attempts + 1))

    def _refresh_engine_status(self) -> None:
        try:
            from stock_model.native_backend import native_status

            info = native_status()
            self.engine_text.set("C++ engine" if info.get("available") else "Python fallback")
        except Exception:
            self.engine_text.set("Engine status unavailable")

    def _restore_active_dataset(self) -> None:
        identifier = self.profile.get("active_dataset")
        entries = self.dataset_library.scan()
        selected = next((item for item in entries if item.identifier == identifier), None)
        if selected is None:
            selected = next((item for item in entries if item.difficulty.lower() == "beginner" and item.primary_file is not None), None)
        if selected is not None:
            self.active_dataset = selected
            self.dataset_text.set(selected.display_name)
            picker_label = next((label for label, entry in self.dataset_choices.items() if entry.identifier == selected.identifier), None)
            if picker_label:
                self.quick_dataset.set(picker_label)

    @staticmethod
    def open_path(path: Path) -> None:
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(path.as_uri())

    @staticmethod
    def _load_profile() -> dict[str, Any]:
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_profile(self) -> None:
        try:
            PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            PROFILE_PATH.write_text(json.dumps(self.profile, indent=2), encoding="utf-8")
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=tuple(WORKSPACES))
    args, _unknown = parser.parse_known_args()
    if args.mode:
        enable_windows_dpi_awareness()
        run_mode(args.mode)
        return
    enable_windows_dpi_awareness()
    root = Tk()
    OmegaShell(root)
    root.mainloop()


if __name__ == "__main__":
    main()
