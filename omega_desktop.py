from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import BOTH, LEFT, TOP, X, StringVar, Tk, messagebox
from tkinter import ttk

APP_NAME = "Omega FISH Model"
APP_VERSION = "1.4.1"
SOURCE_ROOT = Path(__file__).resolve().parent
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_ROOT


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


def run_mode(mode: str) -> None:
    root = Tk()
    if mode == "integrated":
        from integrated_assessment_app import IntegratedAssessmentApp

        apply_window_identity(root, f"{APP_NAME} — Integrated Assessment")
        IntegratedAssessmentApp(root)
    elif mode == "quant":
        from quant_lab_app import QuantLabApp

        apply_window_identity(root, f"{APP_NAME} — Quant Lab")
        QuantLabApp(root)
    elif mode == "validation":
        from omega_complete_app import CompleteAssessmentApp

        apply_window_identity(root, f"{APP_NAME} — Validation and MSE")
        CompleteAssessmentApp(root)
    elif mode == "noaa":
        from noaa_validation_app import NOAAValidationApp

        apply_window_identity(root, f"{APP_NAME} — NOAA / SS3 Validation")
        NOAAValidationApp(root)
    elif mode == "expert":
        from expert_workflow_app import ExpertWorkflowApp

        apply_window_identity(root, f"{APP_NAME} — Automatic Expert Workflow")
        ExpertWorkflowApp(root)
    elif mode == "charts":
        from chart_studio_app import ChartStudioApp

        apply_window_identity(root, f"{APP_NAME} — Interactive Chart Studio")
        ChartStudioApp(root)
    elif mode == "priority":
        from priority_diagnostics_app import PriorityDiagnosticsApp

        apply_window_identity(root, f"{APP_NAME} — Native Engine & Priority Diagnostics")
        PriorityDiagnosticsApp(root)
    elif mode == "truthmse":
        from mse_truth_lab_app import MSETruthLabApp

        apply_window_identity(root, f"{APP_NAME} — Biomass Evidence & Advanced MSE")
        MSETruthLabApp(root)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    root.mainloop()


class OmegaLauncher:
    def __init__(self, root: Tk) -> None:
        self.root = root
        apply_window_identity(root, f"{APP_NAME} {APP_VERSION}")
        root.geometry("1440x980")
        root.minsize(1080, 720)
        self.status = StringVar(value="Omega FISH 1.4.1. Evidence-weighted biomass synthesis, separate-truth age-structured MSE and experimental diagnostics are available.")
        self._configure_style()
        self._build()


    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Launcher.TFrame", background="#f4f7fb")
        style.configure("LauncherHeader.TFrame", background="#0b1f33")
        style.configure("LauncherTitle.TLabel", background="#0b1f33", foreground="#ffffff", font=("Segoe UI", 25, "bold"))
        style.configure("LauncherSub.TLabel", background="#0b1f33", foreground="#c4d2df", font=("Segoe UI", 10))
        style.configure("Workspace.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Workspace.TLabelframe.Label", background="#ffffff", foreground="#102a43", font=("Segoe UI", 11, "bold"))
        style.configure("WorkspaceText.TLabel", background="#ffffff", foreground="#52606d", font=("Segoe UI", 9))
        style.configure("LauncherStatus.TLabel", background="#e8eef5", foreground="#334155", padding=(10, 7))

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(24, 20), style="LauncherHeader.TFrame")
        header.pack(side=TOP, fill=X)
        ttk.Label(header, text=f"{APP_NAME} {APP_VERSION}", style="LauncherTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Integrated stock assessment, quantitative diagnostics, uncertainty, benchmarks and management strategy evaluation",
            style="LauncherSub.TLabel",
            wraplength=940,
        ).pack(anchor="w", pady=(5, 0))

        body = ttk.Frame(self.root, padding=18, style="Launcher.TFrame")
        body.pack(fill=BOTH, expand=True)
        for column in range(3):
            body.columnconfigure(column, weight=1)
        for row in range(4):
            body.rowconfigure(row, weight=1)

        self._card(
            body, 0, 0, "Biomass Evidence & Advanced MSE",
            "Estimate the best-supported biomass from competing models and indices, run a separate-truth age-structured MSE, optimise management procedures and test experimental diagnostics.",
            lambda: self.launch("truthmse"),
        )
        self._card(
            body, 0, 1, "Integrated Assessment",
            "Load real CSV or Excel data; fit age-structured models; inspect recruitment, selectivity, retention, discards, compositions, projections and strategy tests.",
            lambda: self.launch("integrated"),
        )
        self._card(
            body, 0, 2, "Native Engine & Priority Diagnostics",
            "Build and inspect the compiled C++ engine; run fully refitted likelihood profiles, genuine age-structured ASPM/ASPM-R and formal known-truth interval coverage.",
            lambda: self.launch("priority"),
        )
        self._card(
            body, 1, 0, "Automatic Expert Workflow",
            "Run convergence, jitter, independent optimizers, profiles, retrospectives, hindcasts, ASPM, influence, recovery, coverage, ensembles, MSE and reliability grading.",
            lambda: self.launch("expert"),
        )
        self._card(
            body, 1, 1, "Interactive Chart Studio",
            "Zoom, pan, brush, annotate, edit labels, overlay runs, inspect uncertainty, save personal chart profiles and export interactive dashboards.",
            lambda: self.launch("charts"),
        )
        self._card(
            body, 1, 2, "Quant Lab",
            "Run global optimisers, high-dimensional diagnostics, stress tests, model ensembles, walk-forward validation and risk frontiers.",
            lambda: self.launch("quant"),
        )
        self._card(
            body, 2, 0, "NOAA / SS3 Validation",
            "Download pinned NOAA test models, audit SS3 structure, run deterministic parity checks, execute native SS3 and track capability gaps.",
            lambda: self.launch("noaa"),
        )
        self._card(
            body, 2, 1, "Legacy Validation and MSE",
            "Run deterministic benchmarks, reliability diagnostics, tagging demonstrations, CPUE standardisation and closed-loop management strategy evaluation.",
            lambda: self.launch("validation"),
        )
        self._card(
            body, 2, 2, "System Self-Check",
            "Verify Python and compiled backends, interfaces, deterministic benchmarks, NOAA validation, priority diagnostics and the full test suite.",
            self.self_check,
        )
        self._card(
            body, 3, 0, "Roadmap and Evidence",
            "Open the native-engine architecture, mathematical specification, validation plan, capability gaps and evidence-controlled release documentation.",
            self.open_docs,
        )

        footer = ttk.Frame(self.root, padding=(18, 8, 18, 16))
        footer.pack(fill=X)
        ttk.Button(footer, text="Open documentation", command=self.open_docs).pack(side=LEFT)
        ttk.Button(footer, text="Open reports folder", command=self.open_reports).pack(side=LEFT, padx=6)
        ttk.Label(footer, textvariable=self.status, style="LauncherStatus.TLabel").pack(side=LEFT, padx=14, fill=X, expand=True)

    @staticmethod
    def _card(parent, row: int, column: int, title: str, description: str, command) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=18, style="Workspace.TLabelframe")
        frame.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
        ttk.Label(frame, text=description, wraplength=280, justify="left", style="WorkspaceText.TLabel").pack(anchor="w", fill=X, expand=True)
        ttk.Button(frame, text=f"Open {title}", command=command).pack(anchor="e", pady=(18, 0))

    def launch(self, mode: str) -> None:
        try:
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--mode", mode]
            else:
                command = [sys.executable, str(SOURCE_ROOT / "omega_desktop.py"), "--mode", mode]
            subprocess.Popen(command, cwd=str(APP_DIR))
            self.status.set(f"Opened {mode} workspace.")
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def self_check(self) -> None:
        self.status.set("Running complete software self-check...")

        def worker() -> None:
            try:
                from omega_self_check import run_self_check

                result = run_self_check(full_tests=not getattr(sys, "frozen", False))
                reports = APP_DIR / "reports"
                reports.mkdir(parents=True, exist_ok=True)
                output = reports / "self_check_latest.json"
                output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                status = result["software_status"]
                detail = f"{result['checks_passed']} passed; {result['checks_failed']} failed. Saved to {output}."
                self.root.after(0, lambda: self.status.set(f"Self-check {status}: {detail}"))
                self.root.after(0, lambda: messagebox.showinfo("Omega FISH self-check", f"Software status: {status}\n\n{detail}"))
            except Exception as exc:
                self.root.after(0, lambda: self.status.set("Self-check failed."))
                self.root.after(0, lambda: messagebox.showerror("Omega FISH self-check", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def open_docs(self) -> None:
        candidates = [resource_path("README.md"), resource_path("README_READY_TO_RUN.md"), resource_path("BIOMASS_EVIDENCE_ENGINE.md"), resource_path("ADVANCED_MSE.md"), resource_path("EXPERIMENTAL_DIAGNOSTICS.md"), resource_path("RELEASE_1_4_BIOMASS_MSE_EXPERIMENTAL.md"), resource_path("NATIVE_ENGINE_ARCHITECTURE.md"), resource_path("PRIORITY_DIAGNOSTICS_1_3.md"), resource_path("EXPERT_WORKFLOW.md"), resource_path("INTERACTIVE_CHARTS.md"), resource_path("INTEGRATED_ASSESSMENT.md"), resource_path("QUANT_LAB.md")]
        for path in candidates:
            if path.exists():
                try:
                    os.startfile(path)  # type: ignore[attr-defined]
                except Exception:
                    webbrowser.open(path.as_uri())
                return
        messagebox.showinfo(APP_NAME, "Documentation was not found beside the program.")

    def open_reports(self) -> None:
        reports = APP_DIR / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(reports)  # type: ignore[attr-defined]
        except Exception:
            webbrowser.open(reports.as_uri())


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["integrated", "quant", "validation", "noaa", "expert", "charts", "priority", "truthmse"])
    args, _unknown = parser.parse_known_args()
    if args.mode:
        run_mode(args.mode)
        return
    root = Tk()
    OmegaLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
