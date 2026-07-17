from __future__ import annotations

import ctypes
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .core import ModelSettings, _objective_breakdown

_COMPONENT_NAMES = (
    "index_likelihood",
    "biomass_likelihood",
    "terminal_depletion_constraint",
    "observation_error_prior",
    "productivity_prior",
    "initial_depletion_prior",
    "catch_to_capacity_penalty",
)
_MODEL_CODES = {"schaefer": 0, "fox": 1, "pella": 2, "pella_tomlinson": 2, "pella-tomlinson": 2}
_EXPECTED_ABI = 3


class _CProductionSettings(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("pella_shape", ctypes.c_double),
        ("target_depletion", ctypes.c_double),
        ("use_target_depletion", ctypes.c_int),
        ("target_depletion_cv", ctypes.c_double),
        ("r_prior_median", ctypes.c_double),
        ("r_prior_cv", ctypes.c_double),
        ("obs_cv", ctypes.c_double),
        ("index_weight", ctypes.c_double),
        ("biomass_weight", ctypes.c_double),
        ("initial_depletion_prior_mean", ctypes.c_double),
        ("initial_depletion_prior_sd", ctypes.c_double),
        ("initial_depletion_prior_weight", ctypes.c_double),
        ("catch_to_capacity_penalty_weight", ctypes.c_double),
        ("observation_prior_log_sd", ctypes.c_double),
    ]


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    backend: str
    abi_version: int | None
    build_info: str | None
    library_path: str | None
    openmp: bool
    max_threads: int
    reason: str | None = None


@dataclass
class NativeObjectiveResult:
    objective: float
    gradient: np.ndarray
    biomass: np.ndarray
    sigma: float
    components: dict[str, float]
    backend: str


class NativeEngine:
    def __init__(self) -> None:
        self._library: ctypes.CDLL | None = None
        self._path: Path | None = None
        self._reason: str | None = None
        self._dll_directories: list[Any] = []
        self._lock = Lock()
        self._load()

    @staticmethod
    def _candidate_paths() -> list[Path]:
        root = Path(__file__).resolve().parent
        names = ["omega_native.dll", "libomega_native.dll", "libomega_native.so", "libomega_native.dylib", "omega_native.so", "omega_native.dylib"]
        candidates = [root / "native_libs" / name for name in names]
        override = os.environ.get("OMEGA_NATIVE_LIBRARY")
        if override:
            candidates.insert(0, Path(override).expanduser())
        return candidates

    def _load(self) -> None:
        for path in self._candidate_paths():
            if not path.exists():
                continue
            try:
                if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
                    self._dll_directories.append(os.add_dll_directory(str(path.resolve().parent)))
                library = ctypes.CDLL(str(path))
                self._configure(library)
                abi = int(library.omega_engine_abi_version())
                if abi != _EXPECTED_ABI:
                    self._reason = f"Native ABI {abi} does not match expected ABI {_EXPECTED_ABI}."
                    continue
                self._library = library
                self._path = path
                self._reason = None
                return
            except Exception as exc:  # pragma: no cover - platform loader details vary
                self._reason = f"Unable to load {path.name}: {exc}"
        if self._reason is None:
            self._reason = "No compiled Omega native library was found. Run build_native_backend.py."

    @staticmethod
    def _configure(library: ctypes.CDLL) -> None:
        double_pointer = ctypes.POINTER(ctypes.c_double)
        int_pointer = ctypes.POINTER(ctypes.c_int)
        library.omega_engine_abi_version.restype = ctypes.c_int
        library.omega_engine_build_info.restype = ctypes.c_char_p
        library.omega_engine_has_openmp.restype = ctypes.c_int
        library.omega_engine_max_threads.restype = ctypes.c_int
        library.omega_engine_set_threads.argtypes = [ctypes.c_int]
        library.omega_production.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_double]
        library.omega_production.restype = ctypes.c_double
        library.omega_simulate_production.argtypes = [
            double_pointer, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, double_pointer,
        ]
        library.omega_simulate_production.restype = ctypes.c_int
        library.omega_production_objective_gradient.argtypes = [
            double_pointer, int_pointer, double_pointer, double_pointer, double_pointer,
            ctypes.c_int, ctypes.POINTER(_CProductionSettings), double_pointer, double_pointer,
            double_pointer, double_pointer,
        ]
        library.omega_production_objective_gradient.restype = ctypes.c_int
        library.omega_production_batch_objective.argtypes = [
            double_pointer, ctypes.c_int, int_pointer, double_pointer, double_pointer,
            double_pointer, ctypes.c_int, ctypes.POINTER(_CProductionSettings),
            double_pointer, double_pointer,
        ]
        library.omega_production_batch_objective.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._library is not None

    def status(self) -> NativeStatus:
        if self._library is None:
            return NativeStatus(False, "python", None, None, None, False, 1, self._reason)
        info = self._library.omega_engine_build_info()
        return NativeStatus(
            True,
            "cpp",
            int(self._library.omega_engine_abi_version()),
            info.decode("utf-8", errors="replace") if info else None,
            str(self._path),
            bool(self._library.omega_engine_has_openmp()),
            int(self._library.omega_engine_max_threads()),
            None,
        )

    def set_threads(self, threads: int) -> None:
        if self._library is not None:
            self._library.omega_engine_set_threads(max(1, int(threads)))

    @staticmethod
    def _settings(settings: ModelSettings) -> _CProductionSettings:
        model = str(settings.model).strip().lower().removeprefix("state_space_")
        return _CProductionSettings(
            _MODEL_CODES.get(model, 0),
            float(settings.pella_shape),
            float(settings.target_depletion or 0.0),
            int(settings.target_depletion is not None),
            float(settings.target_depletion_cv),
            float(settings.r_prior_median),
            float(settings.r_prior_cv),
            float(settings.obs_cv),
            float(settings.index_weight),
            float(settings.biomass_weight),
            float(settings.initial_depletion_prior_mean),
            float(settings.initial_depletion_prior_sd),
            float(settings.initial_depletion_prior_weight),
            float(settings.catch_to_capacity_penalty_weight),
            float(settings.observation_prior_log_sd),
        )

    @staticmethod
    def _arrays(
        years: Sequence[int], catches: Sequence[float], index: Sequence[float], biomass_observed: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = np.ascontiguousarray(years, dtype=np.int32)
        c = np.ascontiguousarray(catches, dtype=np.float64)
        i = np.ascontiguousarray(index, dtype=np.float64)
        b = np.ascontiguousarray(biomass_observed, dtype=np.float64)
        if not (len(y) == len(c) == len(i) == len(b)) or len(y) == 0:
            raise ValueError("Years, catch, index and biomass arrays must have equal length.")
        if len(y) == 0:
            raise ValueError("At least one model year is required.")
        return y, c, i, b

    def objective_gradient(
        self,
        theta: Sequence[float],
        years: Sequence[int],
        catches: Sequence[float],
        index: Sequence[float],
        biomass_observed: Sequence[float],
        settings: ModelSettings,
        *,
        allow_fallback: bool = True,
    ) -> NativeObjectiveResult:
        theta_array = np.ascontiguousarray(theta, dtype=np.float64)
        if theta_array.shape != (4,):
            raise ValueError("theta must contain four transformed production-model parameters.")
        y, c, i, b = self._arrays(years, catches, index, biomass_observed)
        if self._library is None:
            if not allow_fallback:
                raise RuntimeError(self._reason or "Native engine is unavailable.")
            objective, prediction, sigma, components = _objective_breakdown(theta_array, y, c, i, b, settings)
            gradient = np.empty(4, dtype=float)
            for position in range(4):
                step = 1e-5 * max(1.0, abs(float(theta_array[position])))
                plus = theta_array.copy(); plus[position] += step
                minus = theta_array.copy(); minus[position] -= step
                f_plus = _objective_breakdown(plus, y, c, i, b, settings)[0]
                f_minus = _objective_breakdown(minus, y, c, i, b, settings)[0]
                gradient[position] = (f_plus - f_minus) / (2.0 * step)
            return NativeObjectiveResult(float(objective), gradient, np.asarray(prediction), float(sigma), components, "python-fallback")

        objective = ctypes.c_double()
        gradient = np.empty(4, dtype=np.float64)
        prediction = np.empty(len(y), dtype=np.float64)
        components = np.empty(len(_COMPONENT_NAMES), dtype=np.float64)
        c_settings = self._settings(settings)
        status = self._library.omega_production_objective_gradient(
            theta_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(y),
            ctypes.byref(c_settings),
            ctypes.byref(objective),
            gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            prediction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            components.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"Omega native objective failed with status {status}.")
        sigma = float(np.clip(np.exp(theta_array[3]), 0.03, 1.5))
        return NativeObjectiveResult(
            float(objective.value),
            gradient,
            prediction,
            sigma,
            {name: float(value) for name, value in zip(_COMPONENT_NAMES, components)},
            "cpp",
        )

    def batch_objective(
        self,
        theta_matrix: Sequence[Sequence[float]] | np.ndarray,
        years: Sequence[int],
        catches: Sequence[float],
        index: Sequence[float],
        biomass_observed: Sequence[float],
        settings: ModelSettings,
        *,
        gradients: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, str]:
        matrix = np.ascontiguousarray(theta_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 4:
            raise ValueError("theta_matrix must have shape (candidates, 4).")
        y, c, i, b = self._arrays(years, catches, index, biomass_observed)
        if self._library is None:
            values = np.asarray([_objective_breakdown(row, y, c, i, b, settings)[0] for row in matrix], dtype=float)
            if not gradients:
                return values, None, "python-fallback"
            gradient_rows = [self.objective_gradient(row, y, c, i, b, settings).gradient for row in matrix]
            return values, np.asarray(gradient_rows), "python-fallback"

        objectives = np.empty(matrix.shape[0], dtype=np.float64)
        gradient_matrix = np.empty_like(matrix) if gradients else None
        c_settings = self._settings(settings)
        status = self._library.omega_production_batch_objective(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            matrix.shape[0],
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            i.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(y),
            ctypes.byref(c_settings),
            objectives.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None if gradient_matrix is None else gradient_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if status != 0:
            raise RuntimeError(f"Omega native batch objective failed with status {status}.")
        return objectives, gradient_matrix, "cpp-openmp" if self.status().openmp else "cpp"


_ENGINE = NativeEngine()


def get_native_engine() -> NativeEngine:
    return _ENGINE


def native_status() -> dict[str, Any]:
    return asdict(_ENGINE.status())
