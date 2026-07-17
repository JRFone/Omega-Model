# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(SPECPATH)
doc_names = [
    "README_READY_TO_RUN.md",
    "INTEGRATED_ASSESSMENT.md",
    "QUANT_LAB.md",
    "MATHEMATICAL_SPECIFICATION.md",
    "DATA_DICTIONARY_COMPLETE.md",
    "MODEL_GOVERNANCE.md",
    "VALIDATION_PLAN_COMPLETE.md",
    "KNOWN_LIMITATIONS_COMPLETE.md",
    "RELEASES_4_TO_11.md",
    "VERSION.txt",
    "EXPERT_WORKFLOW.md",
    "INTERACTIVE_CHARTS.md",
    "RELEASE_1_2_EXPERT_WORKFLOW_CHARTS.md",
    "NATIVE_ENGINE_ARCHITECTURE.md",
    "PRIORITY_DIAGNOSTICS_1_3.md",
    "RELEASE_1_3_NATIVE_PRIORITY_DIAGNOSTICS.md",
    "BIOMASS_EVIDENCE_ENGINE.md",
    "ADVANCED_MSE.md",
    "EXPERIMENTAL_DIAGNOSTICS.md",
    "RELEASE_1_4_BIOMASS_MSE_EXPERIMENTAL.md",
]
datas = [(str(root / "assets"), "assets")] + collect_data_files("plotly")
if (root / "validation_data").exists():
    datas.append((str(root / "validation_data"), "validation_data"))
if (root / "Data_Sets").exists():
    datas.append((str(root / "Data_Sets"), "Data_Sets"))
for name in doc_names:
    path = root / name
    if path.exists():
        datas.append((str(path), "."))

native_binaries = []
native_dir = root / "stock_model" / "native_libs"
if native_dir.exists():
    allowed_native_suffixes = {".dll"} if sys.platform.startswith("win") else {".dylib"} if sys.platform == "darwin" else {".so"}
    for candidate in native_dir.iterdir():
        if candidate.suffix.lower() in allowed_native_suffixes:
            native_binaries.append((str(candidate), "stock_model/native_libs"))
    for candidate in native_dir.iterdir():
        if candidate.suffix.lower() in {".json", ".md"}:
            datas.append((str(candidate), "stock_model/native_libs"))

hiddenimports = collect_submodules("stock_model") + collect_submodules("plotly") + [
    "integrated_assessment_app",
    "quant_lab_app",
    "omega_complete_app",
    "noaa_validation_app",
    "expert_workflow_app",
    "chart_studio_app",
    "priority_diagnostics_app",
    "mse_truth_lab_app",
    "omega_self_check",
    "tkinter",
    "tkinter.ttk",
]

a = Analysis(
    [str(root / "omega_desktop.py")],
    pathex=[str(root)],
    binaries=native_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Omega FISH Model",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(root / "assets" / "omega_fish.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Omega FISH Model",
)
