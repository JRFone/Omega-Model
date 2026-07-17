from __future__ import annotations

import json
import threading
import webbrowser
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, X, BooleanVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Callable

from stock_model.advanced_mse import (
    AdvancedMSESettings,
    MSEAssessmentSettings,
    MSEManagementProcedure,
    MSEObservationSettings,
    default_operating_scenarios,
    generate_management_grid,
    run_advanced_mse,
)
from stock_model.age_structured import AgeFitSettings, AgeStructuredSettings, fit_age_structured, read_composition_file
from stock_model.biomass_truth_engine import BiomassTruthSettings, estimate_best_supported_biomass
from stock_model.data_io import read_stock_file
from stock_model.experimental_diagnostics import ExperimentalDiagnosticSettings, run_experimental_diagnostics
from stock_model.truth_mse_charts import (
    write_advanced_mse_dashboard,
    write_biomass_truth_dashboard,
    write_experimental_diagnostics_dashboard,
)

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports" / "biomass_mse_lab"


class MSETruthLabApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Omega FISH — Biomass Evidence, Advanced MSE & Experimental Diagnostics")
        self.root.geometry("1460x940")
        self.root.minsize(1120, 740)
        self.dataset_path = StringVar()
        self.age_path = StringVar()
        self.analysis_level = StringVar(value="quick")
        self.assessment_mode = StringVar(value="fast_filter")
        self.mse_years = IntVar(value=15)
        self.mse_simulations = IntVar(value=10)
        self.workers = IntVar(value=1)
        self.full_grid = BooleanVar(value=False)
        self.status = StringVar(value="Ready. Real data support a best-supported biomass estimate, not an assumption-free known truth.")
        self.last_json: Path | None = None
        self.last_html: Path | None = None
        self._build()

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=14)
        header.pack(fill=X)
        ttk.Label(header, text="Omega Biomass Evidence & Advanced MSE Lab", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Automatically synthesise biomass evidence, test management in a separate age-structured operating model, and run experimental diagnostics without hiding uncertainty or overrides.",
            wraplength=1250,
        ).pack(anchor="w", pady=(4, 0))

        inputs = ttk.LabelFrame(self.root, text="Inputs and workload", padding=10)
        inputs.pack(fill=X, padx=14, pady=(0, 8))
        self._file_row(inputs, 0, "Model-ready catch/index/biomass data", self.dataset_path, (("Data", "*.csv *.xlsx *.xlsm"), ("All", "*.*")))
        self._file_row(inputs, 1, "Optional age composition", self.age_path, (("Data", "*.csv *.xlsx *.xlsm"), ("All", "*.*")))
        controls = ttk.Frame(inputs)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Label(controls, text="Analysis level").pack(side=LEFT)
        ttk.Combobox(controls, textvariable=self.analysis_level, values=("quick", "standard", "formal"), state="readonly", width=12).pack(side=LEFT, padx=6)
        ttk.Button(controls, text="Apply preset", command=self.apply_preset).pack(side=LEFT)
        ttk.Label(controls, text="MSE assessment").pack(side=LEFT, padx=(18, 0))
        ttk.Combobox(controls, textvariable=self.assessment_mode, values=("fast_filter", "biomass_ensemble", "full_age_structured"), state="readonly", width=22).pack(side=LEFT, padx=6)
        ttk.Label(controls, text="Years").pack(side=LEFT, padx=(18, 0))
        ttk.Spinbox(controls, from_=5, to=100, textvariable=self.mse_years, width=7).pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Simulations/scenario").pack(side=LEFT, padx=(12, 0))
        ttk.Spinbox(controls, from_=1, to=10000, textvariable=self.mse_simulations, width=8).pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Workers").pack(side=LEFT, padx=(12, 0))
        ttk.Spinbox(controls, from_=1, to=64, textvariable=self.workers, width=6).pack(side=LEFT, padx=4)
        ttk.Checkbutton(controls, text="Full management grid", variable=self.full_grid).pack(side=LEFT, padx=12)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=BOTH, expand=True, padx=14, pady=4)
        biomass_tab = ttk.Frame(notebook, padding=12)
        mse_tab = ttk.Frame(notebook, padding=12)
        diagnostics_tab = ttk.Frame(notebook, padding=12)
        results_tab = ttk.Frame(notebook, padding=12)
        notebook.add(biomass_tab, text="Best-supported biomass")
        notebook.add(mse_tab, text="Advanced age-structured MSE")
        notebook.add(diagnostics_tab, text="Experimental diagnostics")
        notebook.add(results_tab, text="Evidence and reports")

        ttk.Label(
            biomass_tab,
            text="Fits multiple production structures and index variants, weights them by holdout prediction, propagates parameter/process uncertainty and grades how well absolute biomass is identified.",
            wraplength=1150,
        ).pack(anchor="w")
        ttk.Button(biomass_tab, text="Estimate best-supported biomass", command=self.run_biomass).pack(anchor="w", pady=12)

        ttk.Label(
            mse_tab,
            text="Runs a separate age-structured operating truth, simulated catch/index/composition observations, imperfect reassessment, management decisions, sector allocations, closures, compliance and implementation error across multiple biological truths.",
            wraplength=1150,
        ).pack(anchor="w")
        ttk.Button(mse_tab, text="Run advanced closed-loop MSE", command=self.run_mse).pack(anchor="w", pady=12)
        ttk.Label(
            mse_tab,
            text="Formal 10/10 configuration requires full age-structured reassessment, at least five operating scenarios and at least 500 simulations per scenario. That can take substantial time.",
            wraplength=1150,
        ).pack(anchor="w")

        ttk.Label(
            diagnostics_tab,
            text="Runs change-point, nonlinear-memory, residual-spectrum, hyperstability, Hessian sloppiness, posterior-predictive, data-cloning and adversarial perturbation diagnostics. Experimental flags identify questions; they do not prove causes.",
            wraplength=1150,
        ).pack(anchor="w")
        ttk.Button(diagnostics_tab, text="Run experimental diagnostic suite", command=self.run_diagnostics).pack(anchor="w", pady=12)

        buttons = ttk.Frame(results_tab)
        buttons.pack(fill=X)
        ttk.Button(buttons, text="Open latest JSON", command=self.open_json).pack(side=LEFT)
        ttk.Button(buttons, text="Open latest interactive dashboard", command=self.open_html).pack(side=LEFT, padx=6)
        ttk.Button(buttons, text="Open reports folder", command=self.open_reports).pack(side=LEFT)
        from tkinter import Text

        self.results = Text(results_tab, wrap="word", font=("Consolas", 9))
        self.results.pack(fill=BOTH, expand=True, pady=(10, 0))
        ttk.Label(self.root, textvariable=self.status, padding=(14, 7)).pack(fill=X)

    def _file_row(self, parent, row: int, label: str, variable: StringVar, filetypes) -> None:
        ttk.Label(parent, text=label, width=34).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse", command=lambda: self._browse(variable, filetypes)).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _browse(variable: StringVar, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def apply_preset(self) -> None:
        level = self.analysis_level.get()
        if level == "formal":
            self.assessment_mode.set("full_age_structured")
            self.mse_years.set(30)
            self.mse_simulations.set(500)
            self.full_grid.set(True)
        elif level == "standard":
            self.assessment_mode.set("biomass_ensemble")
            self.mse_years.set(25)
            self.mse_simulations.set(75)
            self.full_grid.set(False)
        else:
            self.assessment_mode.set("fast_filter")
            self.mse_years.set(15)
            self.mse_simulations.set(10)
            self.full_grid.set(False)
        self.status.set(f"Applied {level} preset. Every setting remains editable.")

    def _dataset(self):
        path = Path(self.dataset_path.get()).expanduser()
        if not path.exists():
            raise FileNotFoundError("Select a model-ready data file first.")
        return read_stock_file(path)

    def _age_composition(self):
        text = self.age_path.get().strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError("The age-composition file does not exist.")
        return read_composition_file(path)

    def _paths(self, stem: str) -> tuple[Path, Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        REPORTS.mkdir(parents=True, exist_ok=True)
        return REPORTS / f"{stem}_{stamp}.json", REPORTS / f"{stem}_{stamp}.html"

    def _background(self, label: str, task: Callable[[], tuple[Any, Path, Path]]) -> None:
        self.status.set(label)

        def worker() -> None:
            try:
                payload, json_path, html_path = task()
                self.last_json = json_path
                self.last_html = html_path
                summary = payload.summary if hasattr(payload, "summary") else payload.get("summary", {})
                display = json.dumps({"summary": summary, "json": str(json_path), "dashboard": str(html_path)}, indent=2, default=str)
                self.root.after(0, lambda: self._show(display))
                self.root.after(0, lambda: self.status.set(f"Complete. Evidence saved to {json_path}."))
            except Exception as exc:
                self.root.after(0, lambda: self.status.set("Run failed. No result was hidden."))
                self.root.after(0, lambda: messagebox.showerror("Omega Biomass & MSE Lab", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _show(self, text: str) -> None:
        self.results.delete("1.0", END)
        self.results.insert(END, text)

    def run_biomass(self) -> None:
        def task():
            dataset = self._dataset()
            level = self.analysis_level.get()
            settings = BiomassTruthSettings(
                search_draws=180 if level == "quick" else 500 if level == "standard" else 1000,
                samples=350 if level == "quick" else 1500 if level == "standard" else 5000,
                holdout_years=3 if level == "quick" else 5,
            )
            result = estimate_best_supported_biomass(dataset, settings)
            json_path, html_path = self._paths("best_supported_biomass")
            json_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
            write_biomass_truth_dashboard(result, html_path)
            return result, json_path, html_path

        self._background("Fitting and cross-validating the biomass evidence ensemble...", task)

    def _fit_base_age_model(self):
        dataset = self._dataset()
        composition = self._age_composition()
        level = self.analysis_level.get()
        fit_settings = AgeFitSettings(
            population=14 if level == "quick" else 28 if level == "standard" else 48,
            generations=5 if level == "quick" else 14 if level == "standard" else 30,
            local_rounds=2 if level == "quick" else 4,
            seed=8841,
        )
        return fit_age_structured(dataset, AgeStructuredSettings(), fit_settings, age_composition=composition)

    def run_mse(self) -> None:
        if self.analysis_level.get() == "formal" and self.assessment_mode.get() == "full_age_structured" and self.mse_simulations.get() >= 500:
            if not messagebox.askyesno("Formal MSE workload", "This configuration may take many hours or longer. Continue?"):
                return

        def task():
            base = self._fit_base_age_model()
            if self.full_grid.get():
                procedures = generate_management_grid()
            else:
                procedures = [
                    MSEManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.20, fishing_fraction_of_fmsy=0.55, maximum_catch_change=0.10),
                    MSEManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.75),
                    MSEManagementProcedure("Yield focused", target_depletion=0.35, limit_depletion=0.10, fishing_fraction_of_fmsy=0.95, maximum_catch_change=0.30),
                    MSEManagementProcedure("Seasonal closure", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, seasonal_closure_fraction=0.25),
                    MSEManagementProcedure("Lower recreational effort", target_depletion=0.40, limit_depletion=0.15, fishing_fraction_of_fmsy=0.85, bag_limit_effort_multiplier=0.70),
                ]
            level = self.analysis_level.get()
            scenarios = default_operating_scenarios() if level != "quick" else default_operating_scenarios()[:3]
            payload = run_advanced_mse(
                base,
                procedures,
                scenarios=scenarios,
                observation=MSEObservationSettings(),
                assessment=MSEAssessmentSettings(mode=self.assessment_mode.get(), assessment_interval=3, data_lag_years=1),
                settings=AdvancedMSESettings(
                    years=self.mse_years.get(),
                    simulations_per_scenario=self.mse_simulations.get(),
                    workers=self.workers.get(),
                    sample_trajectories_per_cell=2,
                ),
            )
            json_path, html_path = self._paths("advanced_mse")
            json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            write_advanced_mse_dashboard(payload, html_path)
            return payload, json_path, html_path

        self._background("Running separate-truth age-structured closed-loop MSE...", task)

    def run_diagnostics(self) -> None:
        def task():
            dataset = self._dataset()
            level = self.analysis_level.get()
            payload = run_experimental_diagnostics(
                dataset,
                settings=ExperimentalDiagnosticSettings(
                    search_draws=140 if level == "quick" else 260 if level == "standard" else 600,
                    posterior_predictive_replicates=100 if level == "quick" else 500 if level == "standard" else 2000,
                    mutual_information_permutations=60 if level == "quick" else 300 if level == "standard" else 1000,
                    data_clone_factors=(1, 2) if level == "quick" else (1, 2, 4, 8),
                ),
            )
            json_path, html_path = self._paths("experimental_diagnostics")
            json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            write_experimental_diagnostics_dashboard(payload, html_path)
            return payload, json_path, html_path

        self._background("Running simple and complex experimental diagnostics...", task)

    def open_json(self) -> None:
        if self.last_json and self.last_json.exists():
            webbrowser.open(self.last_json.as_uri())
        else:
            messagebox.showinfo("Omega", "Run an analysis first.")

    def open_html(self) -> None:
        if self.last_html and self.last_html.exists():
            webbrowser.open(self.last_html.as_uri())
        else:
            messagebox.showinfo("Omega", "Run an analysis first.")

    @staticmethod
    def open_reports() -> None:
        REPORTS.mkdir(parents=True, exist_ok=True)
        webbrowser.open(REPORTS.as_uri())


def main() -> None:
    root = Tk()
    MSETruthLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
