from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native"
BUILD = ROOT / "build" / "native"
DESTINATION = ROOT / "stock_model" / "native_libs"


def _run(command: list[str], *, cwd: Path | None = None, tool_path: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    if tool_path is not None:
        environment["PATH"] = str(tool_path) + os.pathsep + environment.get("PATH", "")
    subprocess.run(command, cwd=cwd, check=True, env=environment)


def _tool(name: str) -> str:
    executable = f"{name}.exe" if platform.system() == "Windows" else name
    venv_candidate = Path(sys.executable).resolve().parent / executable
    if venv_candidate.exists():
        return str(venv_candidate)
    located = shutil.which(executable) or shutil.which(name)
    if located:
        return located
    raise FileNotFoundError(f"Required build tool was not found: {name}")


def _compiler() -> str | None:
    """Find a usable compiler even in a normal Windows shell.

    WinGet installs WinLibs into a versioned package directory that is not
    always added to PATH until a new login.  Discovering it here makes the
    documented one-command native build work immediately after installation.
    """
    located = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl.exe")
    if located or platform.system() != "Windows":
        return located
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        package_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        candidates = sorted(
            package_root.glob("BrechtSanders.WinLibs*/*/bin/g++.exe"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return None


def _version_line(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "unknown")


def _copy_with_retry(source: Path, destination: Path, attempts: int = 6) -> None:
    """Copy a just-built DLL after transient test/antivirus handles close."""
    for attempt in range(attempts):
        try:
            shutil.copy2(source, destination)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(0.5 * (attempt + 1))


def _library_candidates(build_dir: Path) -> list[Path]:
    names = {
        "omega_native.dll",
        "libomega_native.dll",
        "libomega_native.so",
        "libomega_native.dylib",
        "omega_native.so",
        "omega_native.dylib",
    }
    return [path for path in build_dir.rglob("*") if path.is_file() and path.name in names]


def build(*, clean: bool, configuration: str, openmp: bool, tests: bool) -> dict[str, object]:
    if clean and BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)

    cmake = _tool("cmake")
    ctest = _tool("ctest")
    generator_arguments: list[str] = []
    generator = "default"
    compiler = _compiler()
    tool_path = Path(compiler).resolve().parent if compiler else None
    if platform.system() == "Windows":
        ninja = _tool("ninja")
        if compiler and Path(compiler).name.lower() in {"g++.exe", "clang++.exe"}:
            generator = "Ninja"
            generator_arguments = ["-G", generator, f"-DCMAKE_CXX_COMPILER={compiler}", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
        else:
            # A normal PowerShell session does not put MSVC's cl.exe on PATH.
            # Select the installed Visual Studio generator explicitly and let
            # CMake locate the compiler toolchain.
            generator = "Visual Studio 17 2022"
            generator_arguments = ["-G", generator, "-A", "x64"]

    configure = [
        cmake,
        "-S",
        str(NATIVE),
        "-B",
        str(BUILD),
        f"-DCMAKE_BUILD_TYPE={configuration}",
        f"-DOMEGA_ENABLE_OPENMP={'ON' if openmp else 'OFF'}",
        f"-DOMEGA_BUILD_TESTS={'ON' if tests else 'OFF'}",
        *generator_arguments,
    ]
    _run(configure, tool_path=tool_path)
    _run([cmake, "--build", str(BUILD), "--config", configuration, "--parallel"], tool_path=tool_path)

    runtime_sources: list[Path] = []
    if platform.system() == "Windows" and compiler and Path(compiler).name.lower() == "g++.exe":
        runtime_dir = Path(compiler).resolve().parent
        test_runtime_dir = BUILD / "bin"
        test_runtime_dir.mkdir(parents=True, exist_ok=True)
        for name in ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libgomp-1.dll", "libwinpthread-1.dll", "libdl.dll"):
            source = runtime_dir / name
            if source.exists():
                # CTest starts the executable from a clean process. Keep MinGW's
                # runtime dependencies beside the test executable so the test is
                # independent of the developer shell's PATH.
                _copy_with_retry(source, test_runtime_dir / name)
                runtime_sources.append(source)

    if tests:
        _run([ctest, "--test-dir", str(BUILD), "-C", configuration, "--output-on-failure"], tool_path=tool_path)

    candidates = _library_candidates(BUILD)
    if not candidates:
        raise FileNotFoundError("CMake completed but no Omega native shared library was found.")
    library = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    destination = DESTINATION / library.name
    _copy_with_retry(library, destination)

    runtime_libraries: list[str] = []
    for source in runtime_sources:
        installed_runtime = DESTINATION / source.name
        _copy_with_retry(source, installed_runtime)
        runtime_libraries.append(str(installed_runtime))

    dll_directory = os.add_dll_directory(str(destination.parent)) if platform.system() == "Windows" else None
    try:
        native = ctypes.CDLL(str(destination))
    finally:
        if dll_directory is not None:
            dll_directory.close()
    native.omega_engine_abi_version.restype = ctypes.c_int
    native.omega_engine_build_info.restype = ctypes.c_char_p
    native.omega_engine_has_openmp.restype = ctypes.c_int
    build_info_value = native.omega_engine_build_info()

    status = {
        "status": "built",
        "platform": platform.platform(),
        "python": sys.version,
        "configuration": configuration,
        "generator": generator,
        "compiler": compiler,
        "compiler_version": _version_line([compiler, "--version"]) if compiler else "CMake-selected MSVC",
        "cmake_version": _version_line([cmake, "--version"]),
        "openmp_requested": openmp,
        "openmp_enabled": bool(native.omega_engine_has_openmp()),
        "tests_run": tests,
        "abi_version": int(native.omega_engine_abi_version()),
        "build_info": build_info_value.decode("utf-8", errors="replace") if build_info_value else None,
        "library": str(destination),
        "size_bytes": destination.stat().st_size,
        "runtime_libraries": runtime_libraries,
    }
    status_path = DESTINATION / "native_build.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Omega's C++ native numerical backend.")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--configuration", default="Release", choices=["Release", "RelWithDebInfo", "Debug"])
    parser.add_argument("--no-openmp", action="store_true")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()
    build(clean=args.clean, configuration=args.configuration, openmp=not args.no_openmp, tests=not args.no_tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
