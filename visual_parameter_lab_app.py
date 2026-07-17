from __future__ import annotations

import csv
from math import exp, log
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, X, Y, Canvas, DoubleVar, Menu, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import numpy as np

from stock_model.core import _production, _reference_points
from stock_model.data_io import StockDataset, read_stock_file


ROOT = Path(__file__).resolve().parent
DEMO_FILE = ROOT / "Data_Sets" / "Data_set_Age_Structured_Demo" / "model_ready_timeseries.csv"


class ScrollFrame(ttk.Frame):
    def __init__(self, parent, *, width: int = 360) -> None:
        super().__init__(parent, width=width, style="Card.TFrame")
        self.pack_propagate(False)
        self.canvas = Canvas(self, width=width - 18, highlightthickness=0, background="#102a43")
        self.canvas.omega_role = "workspace_controls"  # type: ignore[attr-defined]
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=12, style="Card.TFrame")
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


class ParameterSlider(ttk.Frame):
    def __init__(self, parent, title: str, variable: DoubleVar, minimum: float, maximum: float, formatter, changed) -> None:
        super().__init__(parent)
        self.variable = variable
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.formatter = formatter
        self.default_value = float(variable.get())
        self.changed = changed
        self.value_text = StringVar()
        header = ttk.Frame(self)
        header.pack(fill=X)
        ttk.Label(header, text=title, font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        ttk.Label(header, textvariable=self.value_text).pack(side=RIGHT)
        self.scale = ttk.Scale(
            self,
            from_=self.minimum,
            to=self.maximum,
            variable=variable,
            command=lambda _value: self._changed(changed),
        )
        self.scale.pack(fill=X, pady=(3, 0))
        self.scale.bind("<Button-3>", self._show_context_menu)
        limits = ttk.Frame(self)
        limits.pack(fill=X)
        ttk.Label(limits, text=formatter(self.minimum), style="Muted.TLabel").pack(side=LEFT)
        ttk.Label(limits, text=formatter(self.maximum), style="Muted.TLabel").pack(side=RIGHT)
        self.refresh_label()

    def set_range(self, minimum: float, maximum: float) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.scale.configure(from_=self.minimum, to=self.maximum)
        children = self.winfo_children()[-1].winfo_children()
        children[0].configure(text=self.formatter(self.minimum))
        children[-1].configure(text=self.formatter(self.maximum))
        self.refresh_label()

    def refresh_label(self) -> None:
        self.value_text.set(self.formatter(float(self.variable.get())))

    def set_default(self, value: float) -> None:
        self.default_value = float(value)

    def _show_context_menu(self, event) -> str:
        menu = Menu(self, tearoff=False)
        menu.add_command(label="Reset this parameter", command=self._reset_default)
        menu.add_command(label="Copy current value", command=self._copy_value)
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _reset_default(self) -> None:
        self.variable.set(self.default_value)
        self.refresh_label()
        self.changed()

    def _copy_value(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.value_text.get())

    def _changed(self, changed) -> None:
        self.refresh_label()
        changed()


class VisualParameterLabApp:
    """Dataset-aware surplus-production scenario explorer with live sliders."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Omega FISH Model — Visual Parameter Lab")
        self.root.geometry("1420x880")
        self.root.minsize(980, 650)
        self.dataset_path = StringVar(value=str(DEMO_FILE))
        self.dataset_name = StringVar(value="No dataset loaded")
        self.source_note = StringVar(value="")
        self.model = StringVar(value="schaefer")
        self.k = DoubleVar(value=6000.0)
        self.growth = DoubleVar(value=0.25)
        self.initial_depletion = DoubleVar(value=0.85)
        self.catch_multiplier = DoubleVar(value=1.0)
        self.natural_mortality = DoubleVar(value=0.20)
        self.recruitment_strength = DoubleVar(value=1.0)
        self.recruitment_variability = DoubleVar(value=0.25)
        self.catchability = DoubleVar(value=1.0)
        self.observation_error = DoubleVar(value=0.25)
        self.terminal_biomass = StringVar(value="—")
        self.terminal_depletion = StringVar(value="—")
        self.reference_yield = StringVar(value="—")
        self.terminal_fishing_mortality = StringVar(value="—")
        self.interpretation = StringVar(value="Load a dataset to explore a scenario.")
        self.status = StringVar(value="Ready. Slider changes update the scenario immediately.")
        self.dataset: StockDataset | None = None
        self.dataset_source_path: Path | None = None
        self.default_values: dict[str, float] = {}
        self._build()
        if DEMO_FILE.exists():
            self.load_dataset_path(DEMO_FILE, "built-in demonstration")

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(18, 13), style="Header.TFrame")
        header.pack(fill=X)
        ttk.Label(header, text="Visual Parameter Lab", style="Header.TLabel", font=("Segoe UI", 18, "bold")).pack(side=LEFT)
        ttk.Label(
            header,
            text="Move a slider to see the biomass path change",
            style="Header.TLabel",
        ).pack(side=LEFT, padx=18)
        ttk.Button(header, text="RESET REALISTIC DEFAULTS", command=self.reset_realistic_defaults).pack(side=RIGHT)

        body = ttk.Frame(self.root, padding=12)
        body.pack(fill=BOTH, expand=True)
        controls_shell = ScrollFrame(body, width=370)
        controls_shell.pack(side=LEFT, fill=Y, padx=(0, 12))
        self.controls_scroll = controls_shell
        controls = controls_shell.inner

        ttk.Label(controls, text="Current dataset", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(controls, textvariable=self.dataset_name, wraplength=295, justify="left").pack(anchor="w", fill=X, pady=(3, 2))
        ttk.Label(controls, textvariable=self.source_note, style="Muted.TLabel", wraplength=295, justify="left").pack(anchor="w", fill=X)
        ttk.Button(controls, text="Choose CSV or Excel", command=self.choose_dataset).pack(fill=X, pady=(8, 3))
        ttk.Separator(controls).pack(fill=X, pady=12)

        model_row = ttk.Frame(controls)
        model_row.pack(fill=X, pady=(0, 10))
        ttk.Label(model_row, text="Production model", font=("Segoe UI", 9, "bold")).pack(side=LEFT)
        model_box = ttk.Combobox(model_row, textvariable=self.model, values=("schaefer", "fox", "pella"), state="readonly", width=12)
        model_box.pack(side=RIGHT)
        model_box.bind("<<ComboboxSelected>>", lambda _event: self.update_scenario())

        self.k_slider = ParameterSlider(
            controls,
            "Unfished biomass / carrying capacity (K)",
            self.k,
            1000.0,
            20000.0,
            lambda value: f"{value:,.0f} t",
            self.update_scenario,
        )
        self.k_slider.pack(fill=X, pady=8)
        self.growth_slider = ParameterSlider(
            controls,
            "Population growth rate (r)",
            self.growth,
            0.05,
            0.80,
            lambda value: f"{value:.3f} / yr",
            self.update_scenario,
        )
        self.growth_slider.pack(fill=X, pady=8)
        self.depletion_slider = ParameterSlider(
            controls,
            "Starting biomass fraction",
            self.initial_depletion,
            0.20,
            1.00,
            lambda value: f"{value:.2f} K",
            self.update_scenario,
        )
        self.depletion_slider.pack(fill=X, pady=8)
        self.catch_slider = ParameterSlider(
            controls,
            "Fishing pressure (catch / F multiplier)",
            self.catch_multiplier,
            0.0,
            2.0,
            lambda value: f"{value:.2f} × observed",
            self.update_scenario,
        )
        self.catch_slider.pack(fill=X, pady=8)
        self.mortality_slider = ParameterSlider(
            controls,
            "Natural mortality (M)",
            self.natural_mortality,
            0.03,
            0.50,
            lambda value: f"{value:.3f} / yr",
            self.update_scenario,
        )
        self.mortality_slider.pack(fill=X, pady=8)
        self.recruitment_slider = ParameterSlider(
            controls,
            "Recruitment strength",
            self.recruitment_strength,
            0.40,
            1.60,
            lambda value: f"{value:.2f} × baseline",
            self.update_scenario,
        )
        self.recruitment_slider.pack(fill=X, pady=8)
        self.recruitment_variability_slider = ParameterSlider(
            controls,
            "Recruitment variability",
            self.recruitment_variability,
            0.0,
            0.80,
            lambda value: f"σ {value:.2f}",
            self.update_scenario,
        )
        self.recruitment_variability_slider.pack(fill=X, pady=8)
        self.catchability_slider = ParameterSlider(
            controls,
            "Survey catchability (q multiplier)",
            self.catchability,
            0.30,
            2.50,
            lambda value: f"{value:.2f} × q",
            self.update_scenario,
        )
        self.catchability_slider.pack(fill=X, pady=8)
        self.observation_error_slider = ParameterSlider(
            controls,
            "Observation error",
            self.observation_error,
            0.05,
            0.80,
            lambda value: f"log σ {value:.2f}",
            self.update_scenario,
        )
        self.observation_error_slider.pack(fill=X, pady=8)
        ttk.Separator(controls).pack(fill=X, pady=12)
        ttk.Label(
            controls,
            text=(
                "Exploration only: these sliders do not fit parameters or prove scientific accuracy. "
                "Use Integrated Assessment and Diagnostics for fitted estimates and validation evidence."
            ),
            wraplength=295,
            justify="left",
            style="Muted.TLabel",
        ).pack(anchor="w", fill=X)

        main = ttk.Frame(body)
        main.pack(side=RIGHT, fill=BOTH, expand=True)
        metric_row = ttk.Frame(main)
        metric_row.pack(fill=X, pady=(0, 10))
        for column in range(4):
            metric_row.columnconfigure(column, weight=1)
        for column, (title, variable) in enumerate(
            (
                ("Terminal biomass", self.terminal_biomass),
                ("Terminal depletion", self.terminal_depletion),
                ("Reference MSY", self.reference_yield),
                ("Terminal fishing mortality", self.terminal_fishing_mortality),
            )
        ):
            card = ttk.Frame(metric_row, padding=10, style="Card.TFrame")
            card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 5, 0 if column == 3 else 5))
            ttk.Label(card, text=title, style="Muted.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=variable, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(2, 0))

        self.chart = Canvas(main, highlightthickness=1, highlightbackground="#41566f", background="#081522")
        self.chart.pack(fill=BOTH, expand=True)
        self.chart.bind("<Configure>", lambda _event: self.update_scenario())
        self.chart.bind("<Button-3>", self._show_chart_context_menu)
        ttk.Label(main, textvariable=self.interpretation, wraplength=1000, justify="left").pack(anchor="w", fill=X, pady=(9, 3))
        ttk.Label(main, textvariable=self.status, style="Muted.TLabel", wraplength=1000, justify="left").pack(anchor="w", fill=X)

    def choose_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a model-ready Omega dataset",
            filetypes=(("Data files", "*.csv *.xlsx *.xlsm"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm"), ("All files", "*.*")),
        )
        if path:
            self.load_dataset_path(Path(path), "selected directly in Visual Parameter Lab")

    def load_dataset_path(self, path: str | Path, source_note: str = "selected dataset") -> None:
        try:
            resolved = Path(path).expanduser().resolve()
            dataset = read_stock_file(resolved)
        except Exception as exc:
            self.status.set(f"Dataset could not be loaded: {exc}")
            return
        self.dataset = dataset
        self.dataset_source_path = resolved
        self.dataset_path.set(str(resolved))
        self.dataset_name.set(dataset.name.replace("_", " "))
        self.source_note.set(source_note)
        self._derive_realistic_defaults()
        self.reset_realistic_defaults()
        observations = int(dataset.frame["biomass"].notna().sum())
        indices = int(dataset.frame["index"].notna().sum())
        self.status.set(f"Loaded {len(dataset.frame)} years, {observations} biomass observations and {indices} index observations.")

    def _derive_realistic_defaults(self) -> None:
        if self.dataset is None:
            return
        frame = self.dataset.frame
        catch_peak = max(float(frame["catch"].max()), 1.0)
        biomass_values = frame["biomass"].dropna()
        if not biomass_values.empty:
            k_default = max(float(biomass_values.max()) * 1.20, catch_peak * 6.0)
        else:
            k_default = catch_peak * 16.0
        k_default = max(k_default, catch_peak * 4.5, 100.0)
        mortality_default = 0.20
        if "natural_mortality" in frame.columns and frame["natural_mortality"].notna().any():
            mortality_default = float(frame["natural_mortality"].dropna().median())
        register = self.dataset_source_path.parent / "parameter_register.csv" if self.dataset_source_path is not None else None
        if register is not None and register.exists():
            try:
                with register.open("r", encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("parameter") == "natural_mortality_M" and row.get("value"):
                            mortality_default = float(row["value"])
                            break
            except (OSError, TypeError, ValueError):
                pass
        recruitment_default = 1.0
        if "recruitment_multiplier" in frame.columns and frame["recruitment_multiplier"].notna().any():
            recruitment_default = float(np.clip(frame["recruitment_multiplier"].dropna().median(), 0.4, 1.6))
        self.default_values = {
            "k": float(k_default),
            "growth": 0.25,
            "initial_depletion": 0.85,
            "catch_multiplier": 1.0,
            "natural_mortality": float(np.clip(mortality_default, 0.03, 0.50)),
            "recruitment_strength": recruitment_default,
            "recruitment_variability": 0.25,
            "catchability": 1.0,
            "observation_error": 0.25,
        }
        self.k_slider.set_range(max(25.0, k_default * 0.25), max(k_default * 3.0, catch_peak * 24.0))

    def reset_realistic_defaults(self) -> None:
        if not self.default_values:
            return
        self.k.set(self.default_values["k"])
        self.growth.set(self.default_values["growth"])
        self.initial_depletion.set(self.default_values["initial_depletion"])
        self.catch_multiplier.set(self.default_values["catch_multiplier"])
        self.natural_mortality.set(self.default_values["natural_mortality"])
        self.recruitment_strength.set(self.default_values["recruitment_strength"])
        self.recruitment_variability.set(self.default_values["recruitment_variability"])
        self.catchability.set(self.default_values["catchability"])
        self.observation_error.set(self.default_values["observation_error"])
        self.model.set("schaefer")
        for key, slider in zip(
            ("k", "growth", "initial_depletion", "catch_multiplier", "natural_mortality", "recruitment_strength", "recruitment_variability", "catchability", "observation_error"),
            self._sliders(),
        ):
            slider.set_default(self.default_values[key])
            slider.refresh_label()
        self.update_scenario()

    def _sliders(self) -> tuple[ParameterSlider, ...]:
        return (
            self.k_slider,
            self.growth_slider,
            self.depletion_slider,
            self.catch_slider,
            self.mortality_slider,
            self.recruitment_slider,
            self.recruitment_variability_slider,
            self.catchability_slider,
            self.observation_error_slider,
        )

    def _show_chart_context_menu(self, event) -> str:
        menu = Menu(self.root, tearoff=False)
        menu.add_command(label="Reset realistic defaults", command=self.reset_realistic_defaults)
        menu.add_command(label="Choose another dataset", command=self.choose_dataset)
        menu.add_separator()
        menu.add_command(label="Copy terminal biomass", command=lambda: self._copy_text(self.terminal_biomass.get()))
        menu.add_command(label="Copy terminal depletion", command=lambda: self._copy_text(self.terminal_depletion.get()))
        menu.tk_popup(event.x_root, event.y_root)
        return "break"

    def _copy_text(self, value: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(value)

    def update_scenario(self) -> None:
        if self.dataset is None:
            return
        try:
            for slider in self._sliders():
                slider.refresh_label()
            frame = self.dataset.frame
            years = frame["year"].to_numpy(dtype=float)
            catches = frame["catch"].to_numpy(dtype=float) * float(self.catch_multiplier.get())
            k = float(self.k.get())
            growth = float(self.growth.get())
            start = float(self.initial_depletion.get())
            model = self.model.get()
            biomass = self._simulate_scenario(years, catches, k, growth, start, model)
            reference = _reference_points(k, growth, model)
            self.terminal_biomass.set(f"{biomass[-1]:,.0f} t")
            self.terminal_depletion.set(f"{biomass[-1] / k:.1%} of K")
            self.reference_yield.set(f"{reference['msy']:,.0f} t / yr")
            terminal_f = float(catches[-1] / max(biomass[-1], 1e-12))
            self.terminal_fishing_mortality.set(f"{terminal_f:.3f} / yr")
            depletion = float(biomass[-1] / k)
            if depletion < 0.10:
                message = "Scenario warning: terminal biomass is below 10% of K. This parameter-and-catch combination produces severe depletion."
            elif depletion < 0.40:
                message = "Scenario result: terminal biomass is below 40% of K. Inspect diagnostics and uncertainty before drawing a management conclusion."
            else:
                message = "Scenario result: terminal biomass remains at or above 40% of K. This is a deterministic illustration, not a fitted accuracy claim."
            self.interpretation.set(message)
            self._draw_chart(years, biomass, reference["bmsy"])
        except Exception as exc:
            self.status.set(f"Scenario could not be drawn: {exc}")

    def _simulate_scenario(self, years: np.ndarray, catches: np.ndarray, k: float, growth: float, start: float, model: str) -> np.ndarray:
        biomass = np.empty(len(years), dtype=float)
        biomass[0] = max(k * start, 1e-6 * k)
        baseline_m = float(self.default_values.get("natural_mortality", 0.20))
        mortality_change = float(self.natural_mortality.get()) - baseline_m
        recruitment_strength = float(self.recruitment_strength.get())
        recruitment_sigma = float(self.recruitment_variability.get())
        frame = self.dataset.frame if self.dataset is not None else None
        if frame is not None and "recruitment_multiplier" in frame.columns:
            recruitment_data = frame["recruitment_multiplier"].to_numpy(dtype=float)
            recruitment_data = np.where(np.isfinite(recruitment_data) & (recruitment_data > 0), recruitment_data, 1.0)
        else:
            recruitment_data = np.ones(len(years), dtype=float)
        for index in range(1, len(years)):
            previous = biomass[index - 1]
            cycle = np.sin(2.0 * np.pi * (index - 1) / 7.0)
            recruitment_factor = recruitment_strength * recruitment_data[index - 1] * exp(recruitment_sigma * cycle - 0.5 * recruitment_sigma**2)
            production = _production(previous, k, growth, model) * recruitment_factor
            mortality_adjustment = mortality_change * previous
            biomass[index] = max(1e-6 * k, previous + production - mortality_adjustment - catches[index - 1])
        return biomass

    def _draw_chart(self, years: np.ndarray, biomass: np.ndarray, bmsy: float) -> None:
        canvas = self.chart
        width = max(560, canvas.winfo_width())
        height = max(330, canvas.winfo_height())
        canvas.delete("all")
        margin_left, margin_right, margin_top, margin_bottom = 78, 30, 36, 58
        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom
        frame = self.dataset.frame if self.dataset is not None else None
        observed = frame["biomass"].to_numpy(dtype=float) if frame is not None else np.full(len(years), np.nan)
        index = frame["index"].to_numpy(dtype=float) if frame is not None else np.full(len(years), np.nan)
        scaled_index = np.full(len(years), np.nan)
        mask = np.isfinite(index) & (index > 0) & np.isfinite(biomass) & (biomass > 0)
        if mask.any():
            q = exp(float(np.mean(np.log(index[mask]) - np.log(biomass[mask]))))
            scaled_index = index / max(q * float(self.catchability.get()), 1e-12)
        observation_sigma = float(self.observation_error.get())
        uncertainty_lower = biomass * exp(-1.96 * observation_sigma)
        uncertainty_upper = biomass * exp(1.96 * observation_sigma)
        y_values = [float(np.nanmax(uncertainty_upper)), float(bmsy)]
        if np.isfinite(observed).any():
            y_values.append(float(np.nanmax(observed)))
        if np.isfinite(scaled_index).any():
            y_values.append(float(np.nanmax(scaled_index)))
        y_max = max(y_values) * 1.10
        y_max = max(y_max, 1.0)

        def point(year: float, value: float) -> tuple[float, float]:
            x = margin_left + (year - years[0]) / max(years[-1] - years[0], 1.0) * plot_width
            y = margin_top + (1.0 - value / y_max) * plot_height
            return x, y

        grid_color = "#294158"
        text_color = "#dce8f2"
        canvas.create_text(margin_left, 12, anchor="nw", text="Biomass scenario through time", fill=text_color, font=("Segoe UI", 12, "bold"))
        for tick in range(6):
            value = y_max * tick / 5.0
            y = margin_top + plot_height - plot_height * tick / 5.0
            canvas.create_line(margin_left, y, width - margin_right, y, fill=grid_color)
            canvas.create_text(margin_left - 10, y, anchor="e", text=f"{value:,.0f}", fill=text_color, font=("Segoe UI", 8))
        tick_count = min(6, len(years))
        for index_tick in np.linspace(0, len(years) - 1, tick_count).astype(int):
            x, _ = point(float(years[index_tick]), 0.0)
            canvas.create_text(x, height - margin_bottom + 14, anchor="n", text=str(int(years[index_tick])), fill=text_color, font=("Segoe UI", 8))
        canvas.create_text(14, margin_top + plot_height / 2, text="Biomass (tonnes)", angle=90, fill=text_color, font=("Segoe UI", 9))
        canvas.create_text(margin_left + plot_width / 2, height - 10, text="Year", fill=text_color, font=("Segoe UI", 9))

        bmsy_y = point(float(years[0]), bmsy)[1]
        canvas.create_line(margin_left, bmsy_y, width - margin_right, bmsy_y, fill="#a3b8c9", dash=(5, 4), width=1)
        canvas.create_text(width - margin_right - 4, bmsy_y - 4, anchor="se", text="BMSY", fill="#a3b8c9", font=("Segoe UI", 8))

        upper_points = [coordinate for year, value in zip(years, uncertainty_upper) for coordinate in point(float(year), float(value))]
        lower_points = [coordinate for year, value in reversed(list(zip(years, uncertainty_lower))) for coordinate in point(float(year), float(value))]
        canvas.create_polygon(*(upper_points + lower_points), fill="#173752", outline="", stipple="gray50")
        model_points = [coordinate for year, value in zip(years, biomass) for coordinate in point(float(year), float(value))]
        canvas.create_line(*model_points, fill="#42bff5", width=3, smooth=False)
        canvas.create_text(margin_left + 8, margin_top + 25, anchor="nw", text="shaded range = selected observation error", fill="#9bb7cc", font=("Segoe UI", 8))
        canvas.create_text(margin_left + 8, margin_top + 8, anchor="nw", text="— model scenario", fill="#42bff5", font=("Segoe UI", 9, "bold"))

        if np.isfinite(observed).any():
            for year, value in zip(years, observed):
                if np.isfinite(value):
                    x, y = point(float(year), float(value))
                    canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ffb85c", outline="#ffb85c")
            canvas.create_text(margin_left + 150, margin_top + 8, anchor="nw", text="● observed biomass", fill="#ffb85c", font=("Segoe UI", 9, "bold"))
        elif np.isfinite(scaled_index).any():
            series_points = [coordinate for year, value in zip(years, scaled_index) if np.isfinite(value) for coordinate in point(float(year), float(value))]
            if len(series_points) >= 4:
                canvas.create_line(*series_points, fill="#ffb85c", width=2, dash=(4, 3))
                canvas.create_text(
                    margin_left + 150,
                    margin_top + 8,
                    anchor="nw",
                    text="-- index rescaled for visual comparison",
                    fill="#ffb85c",
                    font=("Segoe UI", 9, "bold"),
                )


def main() -> None:
    root = Tk()
    VisualParameterLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
