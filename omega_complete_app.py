from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, TOP, X, Y, Canvas, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from typing import Any, Mapping, Sequence

from stock_model.benchmark_suite import run_benchmarks
from stock_model.closed_loop_mse import MSESettings, ManagementProcedure, OperatingModelSettings, run_closed_loop_mse
from stock_model.complete_assessment import run_complete_demo

APP_TITLE = "Omega FISH Model — Validation and MSE"
ROOT = Path(__file__).resolve().parent


class CompleteAssessmentApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x920")
        self.root.minsize(1050, 700)
        self.status = StringVar(value="Ready. Run the complete demonstration or an individual module.")
        self.output: dict[str, Any] | None = None
        self._build()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(side=TOP, fill=X)
        ttk.Button(toolbar, text="Run complete Releases 4–11 demonstration", command=self.run_complete).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Run benchmarks", command=self.run_benchmark).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Run closed-loop MSE", command=self.run_mse).pack(side=LEFT, padx=3)
        ttk.Button(toolbar, text="Export package", command=self.export).pack(side=LEFT, padx=12)
        ttk.Label(toolbar, textvariable=self.status).pack(side=LEFT, padx=12)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=BOTH, expand=True, padx=8, pady=6)
        self.summary_tab = ttk.Frame(self.tabs)
        self.spatial_tab = ttk.Frame(self.tabs)
        self.cpue_tab = ttk.Frame(self.tabs)
        self.inference_tab = ttk.Frame(self.tabs)
        self.tagging_tab = ttk.Frame(self.tabs)
        self.reliability_tab = ttk.Frame(self.tabs)
        self.mse_tab = ttk.Frame(self.tabs)
        self.benchmark_tab = ttk.Frame(self.tabs)
        self.json_tab = ttk.Frame(self.tabs)
        for frame, title in [
            (self.summary_tab, "Summary"),
            (self.spatial_tab, "Sex / Space / Seasons"),
            (self.cpue_tab, "CPUE Standardisation"),
            (self.inference_tab, "Inference and Uncertainty"),
            (self.tagging_tab, "Tagging"),
            (self.reliability_tab, "Reliability"),
            (self.mse_tab, "Closed-loop MSE"),
            (self.benchmark_tab, "Benchmarks"),
            (self.json_tab, "Raw JSON"),
        ]:
            self.tabs.add(frame, text=title)
        self.summary_tree = self._tree(self.summary_tab)
        self.spatial_tree = self._tree(self.spatial_tab)
        self.cpue_tree = self._tree(self.cpue_tab)
        self.inference_tree = self._tree(self.inference_tab)
        self.tagging_tree = self._tree(self.tagging_tab)
        self.reliability_tree = self._tree(self.reliability_tab)
        self.mse_tree = self._tree(self.mse_tab)
        self.benchmark_tree = self._tree(self.benchmark_tab)
        from tkinter import Text
        self.json_text = Text(self.json_tab, wrap="none", font=("Consolas", 9))
        self.json_text.pack(fill=BOTH, expand=True)

        note = ttk.Label(
            self.root,
            text=(
                "Implemented scope: spatial/sex/seasonal population dynamics for spatial/sex/seasonal population dynamics, tagging, composition likelihoods, "
                "CPUE standardisation, numerical inference, reliability grading, closed-loop MSE, SS3 templates and benchmark reporting. "
                "This is not independent peer-review certification."
            ),
            wraplength=1400,
            justify="left",
            padding=8,
        )
        note.pack(fill=X)

    @staticmethod
    def _tree(parent) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill=BOTH, expand=True)
        tree = ttk.Treeview(frame, show="headings")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.pack(side=LEFT, fill=BOTH, expand=True)
        vertical.pack(side="right", fill=Y)
        horizontal.pack(side="bottom", fill=X)
        return tree

    @staticmethod
    def _fill(tree: ttk.Treeview, rows: Sequence[Mapping[str, Any]]) -> None:
        tree.delete(*tree.get_children())
        if not rows:
            tree["columns"] = ()
            return
        columns = list(dict.fromkeys(key for row in rows for key in row.keys()))
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=140, minwidth=80, stretch=True)
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = f"{value:.6g}"
                elif isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, default=str)[:500]
                values.append(value)
            tree.insert("", END, values=values)

    def _background(self, label: str, function) -> None:
        self.status.set(label)
        def worker() -> None:
            try:
                value = function()
                self.root.after(0, lambda: self._apply(value))
            except Exception:
                error = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror("Omega FISH error", error))
                self.root.after(0, lambda: self.status.set("Operation failed. See error dialog."))
        threading.Thread(target=worker, daemon=True).start()

    def run_complete(self) -> None:
        self._background("Running cumulative Releases 4–11 demonstration...", lambda: run_complete_demo(years=12))

    def run_benchmark(self) -> None:
        self._background("Running deterministic benchmark suite...", lambda: {"benchmarks": run_benchmarks()})

    def run_mse(self) -> None:
        procedures = [
            ManagementProcedure("Conservative", target_depletion=0.45, limit_depletion=0.15, target_f_fraction=0.75),
            ManagementProcedure("Balanced", target_depletion=0.40, limit_depletion=0.10, target_f_fraction=1.0),
            ManagementProcedure("Yield focused", target_depletion=0.35, limit_depletion=0.10, target_f_fraction=1.15),
        ]
        self._background(
            "Running closed-loop management strategy evaluation...",
            lambda: {"mse": run_closed_loop_mse(OperatingModelSettings(), procedures, MSESettings(years=25, simulations=250))},
        )

    def _apply(self, payload: dict[str, Any]) -> None:
        self.output = payload
        summary = []
        if "scope" in payload:
            summary.append({"item": "Scope", "value": payload["scope"]})
        if "reliability" in payload:
            summary.extend([
                {"item": "Reliability grade", "value": payload["reliability"]["grade"]},
                {"item": "Reliability label", "value": payload["reliability"]["label"]},
            ])
        if "benchmarks" in payload:
            b = payload["benchmarks"]["summary"]
            summary.append({"item": "Benchmarks", "value": f"{b['passed']} / {b['total']} passed"})
        if "mse" in payload:
            summary.append({"item": "MSE strategies", "value": len(payload["mse"].get("summary", []))})
        self._fill(self.summary_tree, summary)
        if "spatial_sex_seasonal" in payload:
            self._fill(self.spatial_tree, payload["spatial_sex_seasonal"].get("history", []))
        if "cpue_standardization" in payload:
            self._fill(self.cpue_tree, payload["cpue_standardization"].get("annual_index", []))
        if "inference" in payload:
            rows = [{"parameter": key, "estimate": value, "se": payload["inference"].get("standard_errors", {}).get(key)} for key, value in payload["inference"].get("parameters", {}).items()]
            self._fill(self.inference_tree, rows)
        if "tagging" in payload:
            self._fill(self.tagging_tree, payload["tagging"].get("predictions", []))
        if "reliability" in payload:
            self._fill(self.reliability_tree, payload["reliability"].get("items", []))
        if "mse" in payload:
            self._fill(self.mse_tree, payload["mse"].get("summary", []))
        if "benchmarks" in payload:
            self._fill(self.benchmark_tree, payload["benchmarks"].get("results", []))
        self.json_text.delete("1.0", END)
        self.json_text.insert("1.0", json.dumps(payload, indent=2, default=str))
        self.status.set("Complete.")

    def export(self) -> None:
        if self.output is None:
            messagebox.showinfo("Omega FISH", "Run a module before exporting.")
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        try:
            if "scope" in self.output:
                run_complete_demo(years=12, output_dir=folder)
            else:
                Path(folder, "omega_complete_output.json").write_text(json.dumps(self.output, indent=2, default=str), encoding="utf-8")
            self.status.set(f"Exported to {folder}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


def main() -> None:
    root = Tk()
    CompleteAssessmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
