from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    BooleanVar,
    Canvas,
    DoubleVar,
    IntVar,
    Listbox,
    StringVar,
    Tk,
    filedialog,
    messagebox,
)
from tkinter import ttk
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stock_model.interactive_charts import (
    ChartProfile,
    ChartProfileStore,
    InteractiveChartFactory,
    SeriesSpec,
)


APP_TITLE = "Omega FISH Model — Interactive Chart Studio"
ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / "reports" / "interactive_charts"
PROFILE_STORE = ChartProfileStore(Path.home() / ".omega_fish" / "chart_profiles.json")

CHART_TYPES = (
    "Time series / overlays",
    "Residual heatmap",
    "Jitter distribution",
    "Optimizer agreement",
    "Optimization surface / grid",
    "Likelihood-component conflict",
    "Likelihood profile",
    "Retrospective analysis",
    "Hindcast prediction",
    "Structural ensemble fan",
    "Closed-loop MSE trade-off",
    "Interval coverage",
)


class ScrollFrame(ttk.Frame):
    def __init__(self, parent, *, width: int = 340) -> None:
        super().__init__(parent)
        canvas = Canvas(self, width=width, highlightthickness=0, background="#102a43")
        canvas.omega_role = "workspace_controls"  # type: ignore[attr-defined]
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas, style="Sidebar.TFrame", padding=(16, 14))
        window = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.canvas = canvas
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
            pointer_x, pointer_y = self.winfo_pointerx(), self.winfo_pointery()
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


class NativeChartPreview(ttk.Frame):
    """Lightweight in-window preview of the Plotly figure built by Chart Studio."""

    PALETTE = ("#42bff5", "#ffb85c", "#5ed59a", "#ba9cff", "#f27d90", "#92a9bd")

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.canvas = Canvas(self, highlightthickness=1, highlightbackground="#41566f", background="#081522")
        self.canvas.pack(fill=BOTH, expand=True)
        self.figure = None
        self.title = ""
        self.x_title = ""
        self.y_title = ""
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def show_figure(self, figure, title: str, x_title: str, y_title: str) -> None:
        self.figure = figure
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.redraw()

    def show_message(self, message: str) -> None:
        self.figure = None
        self.canvas.delete("all")
        self.canvas.create_text(
            max(20, self.canvas.winfo_width() // 2),
            max(20, self.canvas.winfo_height() // 2),
            text=message,
            width=max(260, self.canvas.winfo_width() - 80),
            justify="center",
            fill="#dce8f2",
            font=("Segoe UI", 11),
        )

    def redraw(self) -> None:
        if self.figure is None:
            return
        traces = [trace for trace in self.figure.data if getattr(trace, "visible", True) is not False]
        if not traces:
            self.show_message("This chart has no drawable values. Check the selected columns.")
            return
        heatmap = next((trace for trace in traces if getattr(trace, "type", "") == "heatmap"), None)
        if heatmap is not None:
            self._draw_heatmap(heatmap, traces)
            return
        bar = next((trace for trace in traces if getattr(trace, "type", "") == "bar"), None)
        if bar is not None:
            self._draw_bar(bar)
            return
        self._draw_series(traces)

    def _base(self) -> tuple[int, int, int, int, int, int]:
        canvas = self.canvas
        width = max(520, canvas.winfo_width())
        height = max(320, canvas.winfo_height())
        canvas.delete("all")
        left, right, top, bottom = 74, 26, 50, 58
        canvas.create_text(left, 15, anchor="nw", text=self.title, fill="#dce8f2", font=("Segoe UI", 12, "bold"))
        return width, height, left, right, top, bottom

    def _draw_series(self, traces) -> None:
        width, height, left, right, top, bottom = self._base()
        usable = []
        categorical: dict[str, float] = {}
        for trace in traces[:10]:
            y_value = getattr(trace, "y", None)
            y_raw = list(y_value) if y_value is not None else []
            x_value = getattr(trace, "x", None)
            x_raw = list(x_value) if x_value is not None else list(range(len(y_raw)))
            points = []
            for index, (x_value, y_value) in enumerate(zip(x_raw, y_raw)):
                try:
                    y_number = float(y_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(y_number):
                    continue
                try:
                    x_number = float(x_value)
                except (TypeError, ValueError):
                    key = str(x_value)
                    x_number = categorical.setdefault(key, float(len(categorical)))
                if np.isfinite(x_number):
                    points.append((x_number, y_number, index))
            if points:
                usable.append((trace, points))
        if not usable:
            self.show_message("The selected values are not numeric enough to draw this chart.")
            return
        all_x = [point[0] for _trace, points in usable for point in points]
        all_y = [point[1] for _trace, points in usable for point in points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        if x_max <= x_min:
            x_max = x_min + 1.0
        if y_max <= y_min:
            padding = max(abs(y_min) * 0.1, 1.0)
            y_min -= padding
            y_max += padding
        else:
            padding = (y_max - y_min) * 0.08
            y_min -= padding
            y_max += padding
        plot_w, plot_h = width - left - right, height - top - bottom

        def xy(x_value: float, y_value: float) -> tuple[float, float]:
            return (
                left + (x_value - x_min) / (x_max - x_min) * plot_w,
                top + (y_max - y_value) / (y_max - y_min) * plot_h,
            )

        self._axes(width, height, left, right, top, bottom, x_min, x_max, y_min, y_max)
        legend_x = left + 8
        for series_index, (trace, points) in enumerate(usable):
            color = self.PALETTE[series_index % len(self.PALETTE)]
            mode = str(getattr(trace, "mode", "lines") or "lines")
            coords = [coordinate for x_value, y_value, _index in points for coordinate in xy(x_value, y_value)]
            if "lines" in mode and len(coords) >= 4:
                self.canvas.create_line(*coords, fill=color, width=2)
            if "markers" in mode or getattr(trace, "type", "") == "violin" or "lines" not in mode:
                for x_value, y_value, _index in points[:1500]:
                    x_pos, y_pos = xy(x_value, y_value)
                    self.canvas.create_oval(x_pos - 2.5, y_pos - 2.5, x_pos + 2.5, y_pos + 2.5, fill=color, outline=color)
            name = str(getattr(trace, "name", "") or f"Series {series_index + 1}")
            if name and not name.endswith((" upper", " uncertainty")):
                self.canvas.create_line(legend_x, top + 8, legend_x + 18, top + 8, fill=color, width=3)
                self.canvas.create_text(legend_x + 23, top + 8, anchor="w", text=name[:28], fill="#dce8f2", font=("Segoe UI", 8))
                legend_x += min(190, 48 + len(name[:28]) * 6)

    def _axes(self, width, height, left, right, top, bottom, x_min, x_max, y_min, y_max) -> None:
        plot_w, plot_h = width - left - right, height - top - bottom
        for tick in range(6):
            fraction = tick / 5.0
            y_pos = top + plot_h - fraction * plot_h
            y_value = y_min + fraction * (y_max - y_min)
            self.canvas.create_line(left, y_pos, width - right, y_pos, fill="#294158")
            self.canvas.create_text(left - 9, y_pos, anchor="e", text=f"{y_value:.4g}", fill="#dce8f2", font=("Segoe UI", 8))
            x_pos = left + fraction * plot_w
            x_value = x_min + fraction * (x_max - x_min)
            self.canvas.create_text(x_pos, height - bottom + 12, anchor="n", text=f"{x_value:.6g}", fill="#dce8f2", font=("Segoe UI", 8))
        self.canvas.create_text(14, top + plot_h / 2, text=self.y_title, angle=90, fill="#dce8f2", font=("Segoe UI", 9))
        self.canvas.create_text(left + plot_w / 2, height - 9, text=self.x_title, fill="#dce8f2", font=("Segoe UI", 9))

    def _draw_heatmap(self, trace, traces=()) -> None:
        width, height, left, right, top, bottom = self._base()
        try:
            values = np.asarray(trace.z, dtype=float)
        except Exception:
            self.show_message("The heatmap matrix could not be read.")
            return
        if values.ndim != 2 or values.size == 0:
            self.show_message("The heatmap needs a two-dimensional numeric matrix.")
            return
        rows, columns = values.shape
        plot_w, plot_h = width - left - right, height - top - bottom
        finite = values[np.isfinite(values)]
        limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1e-9)
        for row in range(rows):
            for column in range(columns):
                value = values[row, column]
                if not np.isfinite(value):
                    color = "#294158"
                else:
                    strength = min(abs(float(value)) / limit, 1.0)
                    if value >= 0:
                        color = f"#{int(55 + 170 * strength):02x}{int(93 + 50 * (1-strength)):02x}{int(115 + 40 * (1-strength)):02x}"
                    else:
                        color = f"#{int(55 + 30 * (1-strength)):02x}{int(105 + 80 * strength):02x}{int(130 + 105 * strength):02x}"
                x0 = left + column / columns * plot_w
                x1 = left + (column + 1) / columns * plot_w
                y0 = top + row / rows * plot_h
                y1 = top + (row + 1) / rows * plot_h
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#081522")
        x_raw = getattr(trace, "x", None)
        y_raw = getattr(trace, "y", None)
        x_labels = list(x_raw) if x_raw is not None else []
        y_labels = list(y_raw) if y_raw is not None else []
        if columns <= 12:
            for column, label in enumerate(x_labels[:columns]):
                x_pos = left + (column + 0.5) / columns * plot_w
                self.canvas.create_text(x_pos, height - bottom + 10, anchor="n", text=f"{label:.4g}" if isinstance(label, (int, float)) else str(label)[:10], fill="#dce8f2", font=("Segoe UI", 7))
        if rows <= 12:
            for row, label in enumerate(y_labels[:rows]):
                y_pos = top + (row + 0.5) / rows * plot_h
                self.canvas.create_text(left - 7, y_pos, anchor="e", text=f"{label:.4g}" if isinstance(label, (int, float)) else str(label)[:10], fill="#dce8f2", font=("Segoe UI", 7))
        best_trace = next((item for item in traces if str(getattr(item, "name", "")).lower().startswith("best")), None)
        if best_trace is not None and x_labels and y_labels:
            try:
                best_x = float(list(best_trace.x)[0])
                best_y = float(list(best_trace.y)[0])
                x_index = min(range(len(x_labels)), key=lambda index: abs(float(x_labels[index]) - best_x))
                y_index = min(range(len(y_labels)), key=lambda index: abs(float(y_labels[index]) - best_y))
                x_pos = left + (x_index + 0.5) / columns * plot_w
                y_pos = top + (y_index + 0.5) / rows * plot_h
                self.canvas.create_text(x_pos, y_pos, text="★", fill="#ffffff", font=("Segoe UI Symbol", 18, "bold"))
            except (AttributeError, TypeError, ValueError):
                pass
        self.canvas.create_text(left + plot_w / 2, height - 9, text=self.x_title, fill="#dce8f2", font=("Segoe UI", 9))
        self.canvas.create_text(14, top + plot_h / 2, text=self.y_title, angle=90, fill="#dce8f2", font=("Segoe UI", 9))

    def _draw_bar(self, trace) -> None:
        width, height, left, right, top, bottom = self._base()
        try:
            values = np.asarray(list(trace.x if getattr(trace, "orientation", "") == "h" else trace.y), dtype=float)
        except Exception:
            self.show_message("The bar values could not be read.")
            return
        labels = list(trace.y if getattr(trace, "orientation", "") == "h" else trace.x)
        if values.size == 0:
            self.show_message("The selected values produced no bars.")
            return
        plot_w, plot_h = width - left - right, height - top - bottom
        maximum = max(float(np.nanmax(values)), 1e-9)
        bar_h = plot_h / max(len(values), 1)
        for index, (label, value) in enumerate(zip(labels, values)):
            y0 = top + index * bar_h + 2
            y1 = top + (index + 1) * bar_h - 2
            x1 = left + max(0.0, float(value)) / maximum * plot_w
            self.canvas.create_rectangle(left, y0, x1, y1, fill=self.PALETTE[index % len(self.PALETTE)], outline="")
            if len(values) <= 18:
                self.canvas.create_text(left + 5, (y0 + y1) / 2, anchor="w", text=str(label)[:24], fill="#ffffff", font=("Segoe UI", 8))


class ChartStudioApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1600x980")
        self.root.minsize(1180, 720)
        self.frame: pd.DataFrame | None = None
        self.source_path: Path | None = None
        self.last_output: Path | None = None
        self._preview_after: str | None = None
        self.status = StringVar(value="Load an Omega result CSV/JSON or use a built-in demonstration.")
        self.chart_requirement = StringVar(value="The live chart updates when you change columns or chart type.")

        self.chart_type = StringVar(value=CHART_TYPES[0])
        self.x_column = StringVar(value="")
        self.group_column = StringVar(value="")
        self.lower_column = StringVar(value="")
        self.upper_column = StringVar(value="")
        self.title = StringVar(value="Omega FISH interactive chart")
        self.x_title = StringVar(value="Year")
        self.y_title = StringVar(value="Value")
        self.normalize = BooleanVar(value=False)
        self.log_y = BooleanVar(value=False)
        self.optimization_goal = StringVar(value="minimize")

        self.profile_name = StringVar(value="Omega default")
        self.template = StringVar(value="plotly_white")
        self.font_family = StringVar(value="Segoe UI, Arial, sans-serif")
        self.font_size = IntVar(value=13)
        self.title_size = IntVar(value=20)
        self.line_width = DoubleVar(value=2.5)
        self.marker_size = DoubleVar(value=7.0)
        self.show_grid = BooleanVar(value=True)
        self.show_legend = BooleanVar(value=True)
        self.range_slider = BooleanVar(value=True)
        self.editable = BooleanVar(value=True)
        self.scroll_zoom = BooleanVar(value=True)
        self.show_spikes = BooleanVar(value=True)
        self.hovermode = StringVar(value="x unified")
        self.legend_orientation = StringVar(value="horizontal")
        self.background = StringVar(value="#ffffff")
        self.plot_background = StringVar(value="#ffffff")
        self.palette = StringVar(value="#2563eb,#dc2626,#059669,#7c3aed,#d97706,#0891b2,#be185d,#4b5563")
        self.downsample_limit = IntVar(value=10000)
        self.chart_height = IntVar(value=720)

        self._configure_style()
        self._build()
        self._bind_live_preview_controls()
        self._refresh_profiles()
        self.load_demo()

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
        style.configure("HeaderSub.TLabel", background="#0b1f33", foreground="#c8d6e5", font=("Segoe UI", 10))
        style.configure("SideTitle.TLabel", background="#102a43", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("SideText.TLabel", background="#102a43", foreground="#d9e6f2", font=("Segoe UI", 9))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 9))
        style.configure("Status.TLabel", background="#e8eef5", foreground="#334155", padding=(10, 7))
        style.configure("Treeview", rowheight=26, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame")
        shell.pack(fill=BOTH, expand=True)

        header = ttk.Frame(shell, style="Header.TFrame", padding=(24, 18))
        header.pack(side=TOP, fill=X)
        ttk.Label(header, text="Interactive Chart Studio", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Zoom, pan, brush-select, annotate, edit labels, overlay model runs, inspect uncertainty, "
                "save personal display profiles, and export publication-quality figures."
            ),
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(shell, style="App.TFrame")
        body.pack(fill=BOTH, expand=True)

        sidebar = ScrollFrame(body, width=360)
        sidebar.pack(side=LEFT, fill=Y)
        controls = sidebar.inner

        self._side_label(controls, "Data source")
        ttk.Button(controls, text="Load CSV or JSON", command=self.load_file).pack(fill=X, pady=2)
        ttk.Button(controls, text="Load Omega results folder", command=self.load_results_folder).pack(fill=X, pady=2)
        ttk.Button(controls, text="Use interactive demonstration", command=self.load_demo).pack(fill=X, pady=2)
        ttk.Separator(controls).pack(fill=X, pady=12)

        self._side_label(controls, "Chart type")
        self.chart_type_combo = ttk.Combobox(controls, textvariable=self.chart_type, values=CHART_TYPES, state="readonly")
        self.chart_type_combo.pack(fill=X, pady=(4, 7))
        self.chart_type_combo.bind("<<ComboboxSelected>>", self._chart_type_changed)

        self._side_label(controls, "X column")
        self.x_combo = ttk.Combobox(controls, textvariable=self.x_column, state="readonly")
        self.x_combo.pack(fill=X, pady=(4, 7))
        self.x_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_live_preview())

        self._side_label(controls, "Y columns — multi-select")
        self.y_list = Listbox(controls, selectmode="extended", exportselection=False, height=7)
        self.y_list.pack(fill=X, pady=(4, 7))
        self.y_list.bind("<<ListboxSelect>>", lambda _event: self._schedule_live_preview())

        self._side_label(controls, "Group / label column")
        self.group_combo = ttk.Combobox(controls, textvariable=self.group_column, state="readonly")
        self.group_combo.pack(fill=X, pady=(4, 7))
        self.group_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_live_preview())

        self._side_label(controls, "Lower uncertainty / colour column")
        self.lower_combo = ttk.Combobox(controls, textvariable=self.lower_column, state="readonly")
        self.lower_combo.pack(fill=X, pady=(4, 7))
        self.lower_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_live_preview())

        self._side_label(controls, "Upper uncertainty / size column")
        self.upper_combo = ttk.Combobox(controls, textvariable=self.upper_column, state="readonly")
        self.upper_combo.pack(fill=X, pady=(4, 7))
        self.upper_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_live_preview())

        self._side_label(controls, "Optimization goal")
        self.optimization_goal_combo = ttk.Combobox(
            controls,
            textvariable=self.optimization_goal,
            values=("minimize", "maximize"),
            state="readonly",
        )
        self.optimization_goal_combo.pack(fill=X, pady=(4, 7))
        self.optimization_goal_combo.bind("<<ComboboxSelected>>", lambda _event: self._schedule_live_preview())

        ttk.Separator(controls).pack(fill=X, pady=12)
        self._side_label(controls, "Titles and transforms")
        self._entry(controls, "Chart title", self.title)
        self._entry(controls, "X-axis title", self.x_title)
        self._entry(controls, "Y-axis title", self.y_title)
        ttk.Checkbutton(controls, text="Normalize series", variable=self.normalize).pack(anchor="w", pady=2)
        ttk.Checkbutton(controls, text="Logarithmic Y axis", variable=self.log_y).pack(anchor="w", pady=2)

        ttk.Separator(controls).pack(fill=X, pady=12)
        self._side_label(controls, "Personal chart profile")
        self.profile_combo = ttk.Combobox(controls, textvariable=self.profile_name, state="readonly")
        self.profile_combo.pack(fill=X, pady=(4, 5))
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _event: self.load_profile())
        self._entry(controls, "Save as profile name", self.profile_name)
        ttk.Button(controls, text="Save profile", command=self.save_profile).pack(fill=X, pady=2)
        ttk.Button(controls, text="Reset profile", command=self.reset_profile).pack(fill=X, pady=2)

        ttk.Separator(controls).pack(fill=X, pady=12)
        self._side_label(controls, "Display controls")
        ttk.Label(controls, text="Theme", style="SideText.TLabel").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.template,
            values=("plotly_white", "plotly_dark", "simple_white", "ggplot2", "seaborn", "presentation", "none"),
            state="readonly",
        ).pack(fill=X, pady=(3, 6))
        self._spin(controls, "Font size", self.font_size, 8, 32, 1)
        self._spin(controls, "Title size", self.title_size, 10, 48, 1)
        self._spin(controls, "Line width", self.line_width, 0.5, 12.0, 0.5)
        self._spin(controls, "Marker size", self.marker_size, 1.0, 30.0, 1.0)
        self._spin(controls, "Chart height", self.chart_height, 360, 1400, 20)
        self._spin(controls, "Downsample above", self.downsample_limit, 500, 250000, 500)
        self._entry(controls, "Font family", self.font_family)
        self._entry(controls, "Paper background", self.background)
        self._entry(controls, "Plot background", self.plot_background)
        self._entry(controls, "Palette — comma separated", self.palette)
        ttk.Label(controls, text="Hover mode", style="SideText.TLabel").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.hovermode,
            values=("x unified", "y unified", "closest", "x", "y"),
            state="readonly",
        ).pack(fill=X, pady=(3, 6))
        ttk.Label(controls, text="Legend", style="SideText.TLabel").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.legend_orientation,
            values=("horizontal", "vertical"),
            state="readonly",
        ).pack(fill=X, pady=(3, 6))
        for text, variable in (
            ("Show grid", self.show_grid),
            ("Show legend", self.show_legend),
            ("Show range slider", self.range_slider),
            ("Editable titles and annotations", self.editable),
            ("Mouse-wheel zoom", self.scroll_zoom),
            ("Crosshair / spikes", self.show_spikes),
        ):
            ttk.Checkbutton(controls, text=text, variable=variable).pack(anchor="w", pady=2)

        ttk.Separator(controls).pack(fill=X, pady=12)
        ttk.Button(controls, text="UPDATE LIVE CHART", command=self.update_live_preview).pack(fill=X, pady=2)
        ttk.Button(controls, text="OPEN INTERACTIVE PREVIEW", style="Primary.TButton", command=self.preview).pack(fill=X, pady=3)
        ttk.Button(controls, text="Save chart HTML", command=self.save_chart).pack(fill=X, pady=2)
        ttk.Button(controls, text="Build automatic results dashboard", command=self.build_auto_dashboard).pack(fill=X, pady=2)
        ttk.Button(controls, text="Open latest chart", command=self.open_latest).pack(fill=X, pady=2)
        ttk.Button(controls, text="Open chart output folder", command=self.open_output_folder).pack(fill=X, pady=2)

        main = ttk.Frame(body, style="App.TFrame", padding=(16, 14))
        main.pack(side=RIGHT, fill=BOTH, expand=True)
        notebook = ttk.Notebook(main)
        self.notebook = notebook
        notebook.pack(fill=BOTH, expand=True)

        chart_tab = ttk.Frame(notebook, padding=8)
        data_tab = ttk.Frame(notebook, padding=10)
        guide_tab = ttk.Frame(notebook, padding=14)
        profile_tab = ttk.Frame(notebook, padding=14)
        notebook.add(chart_tab, text="Live Chart")
        notebook.add(data_tab, text="Data Preview")
        notebook.add(guide_tab, text="Chart Controls")
        notebook.add(profile_tab, text="Personalisation")

        ttk.Label(chart_tab, textvariable=self.chart_requirement, wraplength=1000, justify="left").pack(anchor="w", fill=X, pady=(0, 7))
        self.live_preview = NativeChartPreview(chart_tab)
        self.live_preview.pack(fill=BOTH, expand=True)
        self.live_preview.show_message("Loading chart data...")

        self.tree = ttk.Treeview(data_tab, show="headings")
        yscroll = ttk.Scrollbar(data_tab, orient="vertical", command=self.tree.yview)
        xscroll = ttk.Scrollbar(data_tab, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        data_tab.rowconfigure(0, weight=1)
        data_tab.columnconfigure(0, weight=1)

        guide = (
            "Interactive controls available in every chart:\n\n"
            "• Mouse wheel: zoom around the cursor.\n"
            "• Drag: zoom rectangle; toolbar also supports pan, lasso and box selection.\n"
            "• Double-click: reset axes.\n"
            "• Hover: linked crosshair and precise values.\n"
            "• Legend click: hide/show a series; double-click isolates it.\n"
            "• Range slider: focus on selected years without deleting data.\n"
            "• Modebar: export PNG, annotate lines/shapes, erase annotations, autoscale and reset.\n"
            "• Editable mode: click chart titles, axis titles and annotations to change them.\n\n"
            "Long time series are downsampled with Largest-Triangle-Three-Buckets so peaks and turning points remain visible while interaction stays fast. The underlying full data are not changed."
        )
        ttk.Label(guide_tab, text=guide, justify="left", wraplength=980, font=("Segoe UI", 11)).pack(anchor="nw")

        profile_text = (
            "Profiles save the full visual setup: theme, fonts, line and marker sizes, grid, legend, range slider, "
            "zoom behaviour, backgrounds, colour palette, hover mode, chart height and downsampling limit.\n\n"
            "Profiles are stored under your Windows user account in .omega_fish/chart_profiles.json. They do not alter model data, model weights or scientific results."
        )
        ttk.Label(profile_tab, text=profile_text, justify="left", wraplength=980, font=("Segoe UI", 11)).pack(anchor="nw")

        ttk.Label(shell, textvariable=self.status, style="Status.TLabel").pack(fill=X, side=TOP)

    @staticmethod
    def _side_label(parent, text: str) -> None:
        ttk.Label(parent, text=text, style="SideTitle.TLabel").pack(anchor="w")

    @staticmethod
    def _entry(parent, label: str, variable) -> None:
        ttk.Label(parent, text=label, style="SideText.TLabel").pack(anchor="w")
        ttk.Entry(parent, textvariable=variable).pack(fill=X, pady=(3, 6))

    @staticmethod
    def _spin(parent, label: str, variable, minimum: float, maximum: float, increment: float) -> None:
        ttk.Label(parent, text=label, style="SideText.TLabel").pack(anchor="w")
        ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum, increment=increment).pack(fill=X, pady=(3, 6))

    def _bind_live_preview_controls(self) -> None:
        for variable in (
            self.title,
            self.x_title,
            self.y_title,
            self.normalize,
            self.log_y,
            self.template,
            self.line_width,
            self.marker_size,
            self.show_grid,
            self.show_legend,
            self.background,
            self.plot_background,
            self.palette,
            self.optimization_goal,
        ):
            variable.trace_add("write", lambda *_args: self._schedule_live_preview())

    def _schedule_live_preview(self) -> None:
        try:
            if self._preview_after is not None:
                self.root.after_cancel(self._preview_after)
            # A short debounce prevents repeated Plotly rebuilds while a user is
            # moving through controls or typing a title.
            self._preview_after = self.root.after(300, self.update_live_preview)
        except Exception:
            self._preview_after = None

    def update_live_preview(self) -> None:
        self._preview_after = None
        if self.frame is None:
            self.live_preview.show_message("Load data to display a chart.")
            return
        try:
            figure = self._build_figure()
            self.live_preview.show_figure(figure, self.title.get().strip() or self.chart_type.get(), self.x_title.get(), self.y_title.get())
            self.status.set(f"Live {self.chart_type.get().lower()} chart updated. Use OPEN INTERACTIVE PREVIEW for zoom, hover, selection and export.")
        except Exception as exc:
            self.live_preview.show_message(f"This chart needs another input:\n{exc}")
            self.status.set(f"Live chart needs attention: {exc}")

    def _chart_type_changed(self, _event=None) -> None:
        self._configure_chart_type()
        self._schedule_live_preview()

    def _select_y_columns(self, names: list[str]) -> None:
        if self.frame is None:
            return
        columns = list(self.frame.columns)
        self.y_list.selection_clear(0, END)
        for name in names:
            if name in columns:
                self.y_list.selection_set(columns.index(name))

    def _configure_chart_type(self) -> None:
        if self.frame is None:
            return
        columns = list(self.frame.columns)
        numeric = [column for column in columns if pd.api.types.is_numeric_dtype(self.frame[column])]

        def pick(names: tuple[str, ...], fallback: str = "") -> str:
            lowered = {column.lower(): column for column in columns}
            for name in names:
                if name.lower() in lowered:
                    return lowered[name.lower()]
            return fallback if fallback in columns else ""

        year = pick(("year", "date", "time", "iteration", "run"), columns[0])
        lower = pick(("lower", "lwr", "q05", "p05", "lo"))
        upper = pick(("upper", "upr", "q95", "p95", "hi"))
        chart = self.chart_type.get()
        fallback_y = [column for column in numeric if column != year]
        settings: dict[str, tuple[str, list[str], str, str, str, str]] = {
            "Time series / overlays": (
                year,
                [pick(("Omega median", "biomass", "depletion", "index"), fallback_y[0] if fallback_y else ""), pick(("Alternative structure", "predicted"), fallback_y[1] if len(fallback_y) > 1 else "")],
                "",
                lower,
                upper,
                "Select one or more numeric Y columns. Optional lower and upper columns add an uncertainty band to the first series.",
            ),
            "Residual heatmap": (year, [pick(("residual", "index_log_residual"), fallback_y[0] if fallback_y else "")], pick(("fleet", "source", "age", "model")), "", "", "Use a group column for rows, or select several residual columns to draw one row per series."),
            "Jitter distribution": (pick(("run", "iteration"), year), [pick(("objective", "nll", "value"), fallback_y[0] if fallback_y else "")], pick(("optimizer", "algorithm", "model")), "", "", "Select an objective/value column and an optimizer or configuration group."),
            "Optimizer agreement": (pick(("objective", "nll"), year), [pick(("terminal_depletion", "depletion", "value"), fallback_y[0] if fallback_y else "")], pick(("optimizer", "algorithm", "model")), "", "", "Select objective on X, an estimated quantity on Y, and optimizer/configuration as the group."),
            "Optimization surface / grid": (pick(("parameter_x", "r", "growth_rate"), year), [pick(("parameter_y", "k", "carrying_capacity"), fallback_y[0] if fallback_y else "")], "", pick(("fitness", "objective", "score", "objective_delta")), "", "Select the two parameters on X and Y, then select objective or fitness in the colour column. The best combination is marked with a star."),
            "Likelihood-component conflict": (pick(("component", "source", "model"), year), [pick(("delta_nll", "objective", "value"), fallback_y[0] if fallback_y else "")], pick(("component", "source", "model")), "", "", "Select a likelihood-component label and its objective change."),
            "Likelihood profile": (pick(("parameter_value", "fixed_value", "value"), year), [pick(("objective", "delta_nll"), fallback_y[0] if fallback_y else ""), pick(("component_a",), fallback_y[1] if len(fallback_y) > 1 else "")], "", "", "", "Select the fixed parameter value on X and total objective first on Y; extra Y columns show component profiles."),
            "Retrospective analysis": (year, [pick(("biomass", "depletion", "estimate", "observed"), fallback_y[0] if fallback_y else "")], pick(("peel", "run", "model")), "", "", "Retrospective charts require a Full/Base group plus Peel 1, Peel 2, and later truncated refits. Generate these in Automatic Expert Workflow."),
            "Hindcast prediction": (year, [pick(("observed", "actual"), fallback_y[0] if fallback_y else ""), pick(("predicted", "prediction"), fallback_y[1] if len(fallback_y) > 1 else "")], "", lower, upper, "Select observed then predicted Y columns. Optional lower/upper columns show predictive uncertainty."),
            "Structural ensemble fan": (year, [pick(("Omega median", "median", "biomass"), fallback_y[0] if fallback_y else ""), pick(("Alternative structure", "member"), fallback_y[1] if len(fallback_y) > 1 else "")], "", lower, upper, "Select the ensemble median first, optional member series after it, plus lower and upper uncertainty columns."),
            "Closed-loop MSE trade-off": (pick(("average_catch", "mean_catch", "catch"), year), [pick(("probability_above_limit", "prob_above_limit", "p_above_limit"), fallback_y[0] if fallback_y else "")], pick(("procedure", "strategy", "model")), pick(("catch_cv", "variability"), lower), pick(("closure_probability", "size"), upper), "Select yield on X, conservation performance on Y, procedure as label, and optional colour/size measures."),
            "Interval coverage": (pick(("nominal", "confidence_level"), year), [pick(("empirical", "coverage"), fallback_y[0] if fallback_y else "")], pick(("parameter", "method", "model")), "", "", "Select nominal coverage on X, empirical coverage on Y, and parameter or method as the group."),
        }
        x_col, y_cols, group, low_col, high_col, requirement = settings[chart]
        clean_y = []
        for name in y_cols:
            if name and name not in clean_y and name != x_col:
                clean_y.append(name)
        self.x_column.set(x_col)
        self.group_column.set(group)
        self.lower_column.set(low_col)
        self.upper_column.set(high_col)
        self._select_y_columns(clean_y)
        self.chart_requirement.set(requirement)

    def _refresh_profiles(self) -> None:
        profiles = PROFILE_STORE.load_all()
        self.profile_combo["values"] = tuple(profiles)
        if self.profile_name.get() not in profiles:
            self.profile_name.set("Omega default")
        self.load_profile()

    def _profile(self) -> ChartProfile:
        palette = tuple(value.strip() for value in self.palette.get().split(",") if value.strip())
        return ChartProfile(
            name=self.profile_name.get().strip() or "Personal profile",
            template=self.template.get(),
            font_family=self.font_family.get(),
            font_size=self.font_size.get(),
            title_size=self.title_size.get(),
            line_width=self.line_width.get(),
            marker_size=self.marker_size.get(),
            show_grid=self.show_grid.get(),
            show_legend=self.show_legend.get(),
            range_slider=self.range_slider.get(),
            editable=self.editable.get(),
            scroll_zoom=self.scroll_zoom.get(),
            show_spikes=self.show_spikes.get(),
            hovermode=self.hovermode.get(),
            legend_orientation=self.legend_orientation.get(),
            background=self.background.get(),
            plot_background=self.plot_background.get(),
            palette=palette,
            downsample_limit=self.downsample_limit.get(),
            default_height=self.chart_height.get(),
        ).validated()

    def load_profile(self) -> None:
        profiles = PROFILE_STORE.load_all()
        profile = profiles.get(self.profile_name.get(), ChartProfile()).validated()
        self.profile_name.set(profile.name)
        self.template.set(profile.template)
        self.font_family.set(profile.font_family)
        self.font_size.set(profile.font_size)
        self.title_size.set(profile.title_size)
        self.line_width.set(profile.line_width)
        self.marker_size.set(profile.marker_size)
        self.show_grid.set(profile.show_grid)
        self.show_legend.set(profile.show_legend)
        self.range_slider.set(profile.range_slider)
        self.editable.set(profile.editable)
        self.scroll_zoom.set(profile.scroll_zoom)
        self.show_spikes.set(profile.show_spikes)
        self.hovermode.set(str(profile.hovermode))
        self.legend_orientation.set("vertical" if profile.legend_orientation == "v" else "horizontal")
        self.background.set(profile.background)
        self.plot_background.set(profile.plot_background)
        self.palette.set(",".join(profile.palette))
        self.downsample_limit.set(profile.downsample_limit)
        self.chart_height.set(profile.default_height)

    def save_profile(self) -> None:
        try:
            profile = self._profile()
            PROFILE_STORE.save(profile)
            self._refresh_profiles()
            self.status.set(f"Saved chart profile: {profile.name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def reset_profile(self) -> None:
        self.profile_name.set("Omega default")
        self.load_profile()
        self.status.set("Restored the Omega default chart profile.")

    def load_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Load Omega chart data",
            filetypes=(("CSV and JSON", "*.csv *.json"), ("CSV", "*.csv"), ("JSON", "*.json"), ("All files", "*.*")),
        )
        if not filename:
            return
        try:
            path = Path(filename)
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            elif path.suffix.lower() == ".json":
                frame = self._frame_from_json(json.loads(path.read_text(encoding="utf-8")))
            else:
                raise ValueError("Select a CSV or JSON file.")
            self._set_frame(frame, path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def load_results_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose an Omega results folder")
        if not folder:
            return
        folder_path = Path(folder)
        preferred = (
            "spatial_history.csv",
            "cpue_standardized.csv",
            "fleet_history.csv",
            "mse_summary.csv",
            "reliability.csv",
        )
        for name in preferred:
            path = folder_path / name
            if path.exists():
                self._set_frame(pd.read_csv(path), path)
                self.status.set(f"Loaded {name}. Use 'Build automatic results dashboard' for every supported file in the folder.")
                return
        candidates = sorted(folder_path.glob("*.csv"))
        if candidates:
            self._set_frame(pd.read_csv(candidates[0]), candidates[0])
            return
        messagebox.showinfo(APP_TITLE, "No CSV result files were found in that folder.")

    def load_demo(self) -> None:
        years = np.arange(1980, 2026)
        row = np.arange(len(years))
        depletion_a = np.clip(0.95 - 0.014 * (years - years[0]) + 0.06 * np.sin((years - 1980) / 4), 0.12, 1.0)
        depletion_b = np.clip(0.93 - 0.012 * (years - years[0]) + 0.04 * np.sin((years - 1981) / 5), 0.14, 1.0)
        uncertainty = 0.04 + 0.0015 * (years - years[0])
        frame = pd.DataFrame(
            {
                "year": years,
                "Omega median": depletion_a,
                "Alternative structure": depletion_b,
                "lower": np.clip(depletion_a - uncertainty, 0, None),
                "upper": np.clip(depletion_a + uncertainty, None, 1.2),
                "model": ["Omega"] * len(years),
                "residual": np.sin(years / 3.1) * 0.8,
                "run": row + 1,
                "optimizer": np.asarray(("DE", "L-BFGS-B", "CMA-ES"))[row % 3],
                "objective": 120.0 + 0.04 * np.square(row - 22) + 0.25 * np.sin(row),
                "terminal_depletion": np.clip(0.30 + 0.025 * np.sin(row / 2.5), 0.0, 1.0),
                "parameter_x": 0.08 + (row % 8) * 0.035,
                "parameter_y": 2200.0 + (row // 8) * 700.0,
                "fitness": 80.0 + np.square((row % 8) - 3.0) * 2.5 + np.square((row // 8) - 2.0) * 4.0,
                "component": np.asarray(("index", "biomass", "composition", "prior"))[row % 4],
                "delta_nll": np.abs(np.sin(row / 6.0)) * 3.0,
                "parameter_value": np.linspace(0.10, 0.90, len(years)),
                "component_a": 0.6 * np.abs(np.linspace(-1.0, 1.0, len(years))) ** 2,
                "component_b": 0.4 * np.abs(np.linspace(-1.0, 1.0, len(years))) ** 2,
                "peel": np.asarray(("Full", "Peel 1", "Peel 2", "Peel 3"))[row % 4],
                "observed": depletion_a + 0.025 * np.sin(row * 1.7),
                "predicted": depletion_a,
                "procedure": np.asarray(("Constant catch", "HCR 40/10", "P*", "Fixed F"))[row % 4],
                "average_catch": 80.0 + row * 4.0,
                "probability_above_limit": np.clip(0.98 - row * 0.012, 0.20, 0.99),
                "catch_cv": 0.08 + (row % 7) * 0.03,
                "closure_probability": np.clip(0.05 + row * 0.015, 0.0, 1.0),
                "nominal": np.asarray((0.50, 0.80, 0.90, 0.95))[row % 4],
                "empirical": np.clip(np.asarray((0.48, 0.78, 0.87, 0.92))[row % 4] + 0.01 * np.sin(row), 0.0, 1.0),
                "parameter": np.asarray(("K", "r", "depletion", "sigma"))[row % 4],
            }
        )
        self._set_frame(frame, None)
        self.title.set("Omega model trajectories and uncertainty")
        self.x_title.set("Year")
        self.y_title.set("Relative biomass / depletion")
        self.lower_column.set("lower")
        self.upper_column.set("upper")
        self._configure_chart_type()
        self._schedule_live_preview()

    @staticmethod
    def _frame_from_json(payload: Any) -> pd.DataFrame:
        if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
            return pd.DataFrame(payload)
        if isinstance(payload, Mapping):
            for value in payload.values():
                try:
                    return ChartStudioApp._frame_from_json(value)
                except ValueError:
                    continue
        raise ValueError("The JSON did not contain a list of tabular records.")

    def _set_frame(self, frame: pd.DataFrame, path: Path | None) -> None:
        if frame.empty:
            raise ValueError("The selected data file is empty.")
        frame = frame.copy()
        frame.columns = [str(column) for column in frame.columns]
        self.frame = frame
        self.source_path = path
        columns = list(frame.columns)
        values = ("", *columns)
        self.x_combo["values"] = values
        self.group_combo["values"] = values
        self.lower_combo["values"] = values
        self.upper_combo["values"] = values
        self.y_list.delete(0, END)
        for column in columns:
            self.y_list.insert(END, column)
        numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
        x_guess = next((column for column in columns if column.lower() in {"year", "date", "time", "iteration", "run"}), columns[0])
        self.x_column.set(x_guess)
        selected = [column for column in numeric if column != x_guess][:2]
        for column in selected:
            self.y_list.selection_set(columns.index(column))
        self.group_column.set("")
        self.lower_column.set(next((column for column in columns if column.lower() in {"lower", "lwr", "q05", "p05", "lo"}), ""))
        self.upper_column.set(next((column for column in columns if column.lower() in {"upper", "upr", "q95", "p95", "hi"}), ""))
        self._show_preview(frame)
        source = str(path) if path else "built-in demonstration"
        self.status.set(f"Loaded {len(frame):,} rows × {len(columns)} columns from {source}.")

    def _show_preview(self, frame: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        columns = list(frame.columns)
        self.tree["columns"] = columns
        for column in columns:
            self.tree.heading(column, text=column)
            width = max(90, min(220, len(column) * 10 + 30))
            self.tree.column(column, width=width, stretch=True)
        for row in frame.head(500).itertuples(index=False, name=None):
            self.tree.insert("", END, values=[self._display(value) for value in row])

    @staticmethod
    def _display(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _selected_y(self) -> list[str]:
        if self.frame is None:
            return []
        columns = list(self.frame.columns)
        return [columns[index] for index in self.y_list.curselection()]

    def _require_frame(self) -> pd.DataFrame:
        if self.frame is None:
            raise ValueError("Load data first.")
        return self.frame

    def _build_figure(self):
        frame = self._require_frame()
        factory = InteractiveChartFactory(self._profile())
        chart_type = self.chart_type.get()
        x_col = self.x_column.get()
        y_cols = self._selected_y()
        group_col = self.group_column.get()
        lower_col = self.lower_column.get()
        upper_col = self.upper_column.get()
        if not y_cols:
            raise ValueError("Select at least one Y column.")
        if not x_col and chart_type not in {"Jitter distribution", "Likelihood-component conflict"}:
            raise ValueError("Select an X column.")

        title = self.title.get().strip() or chart_type
        if chart_type == "Time series / overlays":
            series: list[SeriesSpec] = []
            if group_col:
                for group, subset in frame.groupby(group_col, dropna=False):
                    for y_col in y_cols:
                        series.append(
                            SeriesSpec(
                                f"{group} — {y_col}",
                                subset[x_col].tolist(),
                                pd.to_numeric(subset[y_col], errors="coerce").tolist(),
                                lower=pd.to_numeric(subset[lower_col], errors="coerce").tolist() if lower_col else None,
                                upper=pd.to_numeric(subset[upper_col], errors="coerce").tolist() if upper_col else None,
                                mode="lines+markers",
                            )
                        )
            else:
                for index, y_col in enumerate(y_cols):
                    series.append(
                        SeriesSpec(
                            y_col,
                            frame[x_col].tolist(),
                            pd.to_numeric(frame[y_col], errors="coerce").tolist(),
                            lower=pd.to_numeric(frame[lower_col], errors="coerce").tolist() if lower_col and index == 0 else None,
                            upper=pd.to_numeric(frame[upper_col], errors="coerce").tolist() if upper_col and index == 0 else None,
                            mode="lines+markers",
                        )
                    )
            return factory.time_series(
                series,
                title=title,
                x_title=self.x_title.get(),
                y_title=self.y_title.get(),
                normalize=self.normalize.get(),
                log_y=self.log_y.get(),
            )

        if chart_type == "Residual heatmap":
            value_col = y_cols[0]
            if group_col:
                pivot = frame.pivot_table(index=group_col, columns=x_col, values=value_col, aggfunc="mean")
                return factory.residual_heatmap(
                    pivot.to_numpy(),
                    x_labels=pivot.columns.tolist(),
                    y_labels=pivot.index.tolist(),
                    title=title,
                )
            matrix = frame[y_cols].apply(pd.to_numeric, errors="coerce").to_numpy().T
            return factory.residual_heatmap(matrix, x_labels=frame[x_col].tolist(), y_labels=y_cols, title=title)

        records = frame.to_dict(orient="records")
        if chart_type == "Jitter distribution":
            return factory.jitter_distribution(records, value_key=y_cols[0], group_key=group_col or x_col or y_cols[0], title=title)
        if chart_type == "Optimizer agreement":
            return factory.optimizer_agreement(records, x_key=x_col, y_key=y_cols[0], label_key=group_col or y_cols[0], title=title)
        if chart_type == "Optimization surface / grid":
            if not lower_col:
                raise ValueError("Choose an objective or fitness column under Lower uncertainty / colour column.")
            return factory.optimization_surface(
                records,
                x_key=x_col,
                y_key=y_cols[0],
                value_key=lower_col,
                maximize=self.optimization_goal.get() == "maximize",
                title=title,
            )
        if chart_type == "Likelihood-component conflict":
            return factory.likelihood_conflict(records, component_key=group_col or x_col, value_key=y_cols[0], title=title)
        if chart_type == "Likelihood profile":
            return factory.likelihood_profile(records, parameter_key=x_col, objective_key=y_cols[0], component_keys=y_cols[1:], title=title, parameter_label=self.x_title.get())
        if chart_type == "Retrospective analysis":
            if not group_col:
                raise ValueError("Choose a group column identifying Full, Peel 1, Peel 2, and so on.")
            full: dict[Any, float] = {}
            peels: list[dict[Any, float]] = []
            groups = list(frame.groupby(group_col, dropna=False))
            for group, subset in groups:
                values = dict(zip(subset[x_col].tolist(), pd.to_numeric(subset[y_cols[0]], errors="coerce").tolist()))
                if str(group).lower() in {"full", "base", "reference", "0"} or not full:
                    if not full:
                        full = values
                    else:
                        peels.append(values)
                else:
                    peels.append(values)
            return factory.retrospective(full, peels, title=title, y_title=self.y_title.get())
        if chart_type == "Hindcast prediction":
            if len(y_cols) < 2:
                raise ValueError("Select observed and predicted Y columns, in that order.")
            renamed = []
            for row in records:
                renamed.append(
                    {
                        "year": row.get(x_col),
                        "observed": row.get(y_cols[0]),
                        "predicted": row.get(y_cols[1]),
                        "lower": row.get(lower_col) if lower_col else np.nan,
                        "upper": row.get(upper_col) if upper_col else np.nan,
                    }
                )
            return factory.hindcast(renamed, title=title)
        if chart_type == "Structural ensemble fan":
            if not lower_col or not upper_col:
                raise ValueError("Choose lower and upper uncertainty columns.")
            members = {column: pd.to_numeric(frame[column], errors="coerce").tolist() for column in y_cols[1:]}
            return factory.ensemble_fan(
                frame[x_col].tolist(),
                pd.to_numeric(frame[y_cols[0]], errors="coerce").tolist(),
                pd.to_numeric(frame[lower_col], errors="coerce").tolist(),
                pd.to_numeric(frame[upper_col], errors="coerce").tolist(),
                members,
                title=title,
                y_title=self.y_title.get(),
            )
        if chart_type == "Closed-loop MSE trade-off":
            return factory.mse_tradeoff(
                records,
                x_key=x_col,
                y_key=y_cols[0],
                color_key=lower_col or (y_cols[1] if len(y_cols) > 1 else y_cols[0]),
                size_key=upper_col or (y_cols[2] if len(y_cols) > 2 else y_cols[0]),
                label_key=group_col or x_col,
                title=title,
            )
        if chart_type == "Interval coverage":
            return factory.interval_coverage(records, nominal_key=x_col, empirical_key=y_cols[0], parameter_key=group_col or y_cols[0], title=title)
        raise ValueError(f"Unsupported chart type: {chart_type}")

    def preview(self) -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        output = REPORT_ROOT / "omega_chart_preview.html"
        self._render(output, open_after=True)

    def save_chart(self) -> None:
        filename = filedialog.asksaveasfilename(
            title="Save interactive Omega chart",
            defaultextension=".html",
            filetypes=(("Interactive HTML", "*.html"),),
            initialdir=str(REPORT_ROOT),
            initialfile="omega_interactive_chart.html",
        )
        if filename:
            self._render(Path(filename), open_after=False)

    def _render(self, output: Path, *, open_after: bool) -> None:
        self.status.set("Building interactive chart...")

        def worker() -> None:
            try:
                figure = self._build_figure()
                path = InteractiveChartFactory(self._profile()).write_html(figure, output, title=self.title.get())
                self.last_output = path
                self.root.after(0, lambda: self.status.set(f"Interactive chart written to {path}"))
                if open_after:
                    self.root.after(0, lambda: webbrowser.open(path.as_uri()))
            except Exception as exc:
                detail = traceback.format_exc()
                self.root.after(0, lambda: self.status.set("Chart generation failed."))
                self.root.after(0, lambda: messagebox.showerror(APP_TITLE, f"{exc}\n\n{detail[-1800:]}"))

        threading.Thread(target=worker, daemon=True).start()

    def build_auto_dashboard(self) -> None:
        base = self.source_path.parent if self.source_path else ROOT / "reports" / "complete_demo_release_1_1"
        if not base.exists():
            base = ROOT / "models" / "Omega_Complete_Release_11_Demo"
        try:
            figures: dict[str, Any] = {}
            factory = InteractiveChartFactory(self._profile())
            spatial = base / "spatial_history.csv"
            if spatial.exists():
                frame = pd.read_csv(spatial)
                x = "year" if "year" in frame else frame.columns[0]
                numeric = [column for column in frame.columns if column != x and pd.api.types.is_numeric_dtype(frame[column])]
                series = [SeriesSpec(column, frame[x].tolist(), frame[column].tolist()) for column in numeric[:8]]
                figures["Spatial trajectories"] = factory.time_series(series, title="Spatial and population trajectories", y_title="Value")
            cpue = base / "cpue_standardized.csv"
            if cpue.exists():
                frame = pd.read_csv(cpue)
                x = "year" if "year" in frame else frame.columns[0]
                numeric = [column for column in frame.columns if column != x and pd.api.types.is_numeric_dtype(frame[column])]
                figures["CPUE"] = factory.time_series(
                    [SeriesSpec(column, frame[x].tolist(), frame[column].tolist()) for column in numeric[:8]],
                    title="CPUE standardisation and index comparison",
                    y_title="Index",
                )
            mse = base / "mse_summary.csv"
            if mse.exists():
                frame = pd.read_csv(mse)
                records = frame.to_dict(orient="records")
                x = next((column for column in ("average_catch", "mean_catch", "catch") if column in frame), None)
                y = next((column for column in ("probability_above_limit", "prob_above_limit", "p_above_limit") if column in frame), None)
                if x and y:
                    figures["MSE trade-offs"] = factory.mse_tradeoff(records, x_key=x, y_key=y, label_key="procedure" if "procedure" in frame else frame.columns[0])
            fleet = base / "fleet_history.csv"
            if fleet.exists():
                frame = pd.read_csv(fleet)
                x = "year" if "year" in frame else frame.columns[0]
                numeric = [column for column in frame.columns if column != x and pd.api.types.is_numeric_dtype(frame[column])]
                figures["Fleet history"] = factory.time_series(
                    [SeriesSpec(column, frame[x].tolist(), frame[column].tolist()) for column in numeric[:8]],
                    title="Fleet, retention and mortality history",
                    y_title="Value",
                )
            if not figures:
                figures["Current data"] = self._build_figure()
            output = REPORT_ROOT / "omega_results_dashboard.html"
            factory.write_dashboard(
                figures,
                output,
                title="Omega FISH interactive results dashboard",
                metadata={
                    "Source folder": base,
                    "Chart profile": self._profile().name,
                    "Charts": len(figures),
                    "Data are transformed": "No — display-only downsampling may be used for long series",
                },
            )
            self.last_output = output
            self.status.set(f"Built {len(figures)}-panel dashboard: {output}")
            webbrowser.open(output.as_uri())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def open_latest(self) -> None:
        if self.last_output and self.last_output.exists():
            webbrowser.open(self.last_output.as_uri())
        else:
            messagebox.showinfo(APP_TITLE, "No chart has been generated yet.")

    @staticmethod
    def open_output_folder() -> None:
        REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(REPORT_ROOT)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(REPORT_ROOT.as_uri())


def main() -> None:
    root = Tk()
    ChartStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
