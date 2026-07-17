from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from .core import FitResult, ModelSettings, _reference_points
from .data_io import StockDataset
from .native_backend import get_native_engine

_PARAMETER_NAMES = ("k", "r", "initial_depletion", "sigma")
_TRANSFORMED_NAMES = ("log_k", "log_r", "logit_initial_depletion", "log_sigma")
_EPS = 1e-12


@dataclass(frozen=True)
class ProfileSettings:
    points: int = 21
    confidence_levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
    workers: int = 1
    multistarts: int = 3
    seed: int = 17311
    max_iterations: int = 500
    gtol: float = 1e-7
    ftol: float = 1e-12
    range_multiplier: float = 2.5
    use_cache: bool = True
    cache_dir: str | None = None


@dataclass(frozen=True)
class TwoDimensionalProfileSettings:
    x_points: int = 15
    y_points: int = 15
    workers: int = 1
    multistarts: int = 2
    seed: int = 19391
    max_iterations: int = 400
    range_multiplier: float = 2.0
    use_cache: bool = True
    cache_dir: str | None = None


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return math.log(value / (1.0 - value))


def _decode(theta: Sequence[float]) -> dict[str, float]:
    values = np.asarray(theta, dtype=float)
    return {
        "k": float(math.exp(values[0])),
        "r": float(math.exp(values[1])),
        "initial_depletion": float(1.0 / (1.0 + math.exp(-values[2]))),
        "sigma": float(np.clip(math.exp(values[3]), 0.03, 1.5)),
    }


def _encode_from_fit(fitted: FitResult) -> np.ndarray:
    return np.asarray(
        [
            math.log(max(float(fitted.best["k_b0"]), _EPS)),
            math.log(max(float(fitted.best["r"]), _EPS)),
            _logit(float(fitted.best["initial_depletion"])),
            math.log(max(float(fitted.best["sigma"]), 0.03)),
        ],
        dtype=float,
    )


def _parameter_index(parameter: str) -> int:
    key = str(parameter).strip().lower().replace("b0", "k")
    aliases = {
        "k_b0": "k",
        "carrying_capacity": "k",
        "productivity": "r",
        "b0_frac": "initial_depletion",
        "initial": "initial_depletion",
        "observation_sigma": "sigma",
    }
    key = aliases.get(key, key)
    if key not in _PARAMETER_NAMES:
        raise ValueError(f"Unsupported production profile parameter: {parameter!r}. Supported: {', '.join(_PARAMETER_NAMES)}")
    return _PARAMETER_NAMES.index(key)


def _bounds(dataset: StockDataset, settings: ModelSettings) -> list[tuple[float, float]]:
    catches = dataset.frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)
    max_catch = max(float(np.nanmax(catches)), 1.0)
    total_catch = max(float(np.nansum(catches)), max_catch)
    k_low = max(max_catch * 1.051, total_catch * 0.05)
    k_high = max(k_low * 20.0, total_catch * 30.0)
    return [
        (math.log(k_low), math.log(k_high)),
        (math.log(0.0051), math.log(1.199)),
        (-8.0, 8.0),
        (math.log(0.03), math.log(1.5)),
    ]


def _profile_grid(center: np.ndarray, parameter_index: int, bounds: Sequence[tuple[float, float]], points: int, multiplier: float) -> np.ndarray:
    low, high = bounds[parameter_index]
    if parameter_index in (0, 1, 3):
        half_width = max(0.30, math.log(max(float(multiplier), 1.05)))
    else:
        half_width = max(1.0, float(multiplier))
    grid_low = max(low, float(center[parameter_index]) - half_width)
    grid_high = min(high, float(center[parameter_index]) + half_width)
    if grid_high <= grid_low:
        grid_low, grid_high = low, high
    return np.linspace(grid_low, grid_high, max(5, int(points)))


def _dataset_payload(dataset: StockDataset, settings: ModelSettings) -> dict[str, Any]:
    frame = dataset.frame.sort_values("year").reset_index(drop=True)
    return {
        "years": frame["year"].to_numpy(dtype=int).tolist(),
        "catches": (frame["catch"].to_numpy(dtype=float) * max(float(settings.catch_multiplier), 0.0)).tolist(),
        "index": frame["index"].to_numpy(dtype=float).tolist(),
        "biomass": frame["biomass"].to_numpy(dtype=float).tolist(),
        "settings": asdict(settings),
    }


def _settings_from_dict(values: Mapping[str, Any]) -> ModelSettings:
    allowed = {field for field in ModelSettings.__dataclass_fields__}
    return ModelSettings(**{key: value for key, value in values.items() if key in allowed})


def _cache_key(payload: Mapping[str, Any]) -> str:
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=True)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str | None, key: str) -> Path | None:
    if not cache_dir:
        return None
    path = Path(cache_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path / f"profile_{key}.json"


def _safe_objective(theta: np.ndarray, arrays: Mapping[str, Any], settings: ModelSettings) -> tuple[float, np.ndarray, dict[str, float], np.ndarray, str]:
    engine = get_native_engine()
    result = engine.objective_gradient(
        theta,
        arrays["years"],
        arrays["catches"],
        arrays["index"],
        arrays["biomass"],
        settings,
    )
    return result.objective, result.gradient, result.components, result.biomass, result.backend


def _optimise_fixed(
    arrays: Mapping[str, Any],
    settings: ModelSettings,
    center: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    fixed: Mapping[int, float],
    multistarts: int,
    seed: int,
    max_iterations: int,
    gtol: float,
    ftol: float,
) -> dict[str, Any]:
    center_array = np.asarray(center, dtype=float)
    free_indices = [index for index in range(4) if index not in fixed]
    free_bounds = [bounds[index] for index in free_indices]
    rng = np.random.default_rng(seed)

    def assemble(free: Sequence[float]) -> np.ndarray:
        theta = center_array.copy()
        for index, value in fixed.items():
            theta[int(index)] = float(value)
        for index, value in zip(free_indices, free):
            theta[index] = float(value)
        return theta

    starts: list[np.ndarray] = [np.asarray([center_array[index] for index in free_indices], dtype=float)]
    for _ in range(max(0, int(multistarts) - 1)):
        jittered = starts[0].copy()
        for position, (low, high) in enumerate(free_bounds):
            scale = 0.12 * (high - low)
            jittered[position] = np.clip(jittered[position] + rng.normal(0.0, scale), low, high)
        starts.append(jittered)

    best_result: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for start_number, start in enumerate(starts, 1):
        evaluations = 0

        def fun(free: np.ndarray) -> tuple[float, np.ndarray]:
            nonlocal evaluations
            evaluations += 1
            theta = assemble(free)
            objective, gradient, _components, _biomass, _backend = _safe_objective(theta, arrays, settings)
            return float(objective), np.asarray([gradient[index] for index in free_indices], dtype=float)

        try:
            result = minimize(
                lambda values: fun(values)[0],
                start,
                method="L-BFGS-B",
                jac=lambda values: fun(values)[1],
                bounds=free_bounds,
                options={"maxiter": max(20, int(max_iterations)), "gtol": float(gtol), "ftol": float(ftol), "maxls": 60},
            )
            theta = assemble(result.x)
            objective, gradient, components, biomass, backend = _safe_objective(theta, arrays, settings)
            attempt = {
                "start": start_number,
                "success": bool(result.success and np.isfinite(objective)),
                "optimizer_success": bool(result.success),
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", 0)),
                "evaluations": evaluations,
                "objective": float(objective),
                "theta": theta.tolist(),
                "gradient": np.asarray(gradient, dtype=float).tolist(),
                "maximum_free_gradient": float(max((abs(float(gradient[index])) for index in free_indices), default=0.0)),
                "components": components,
                "biomass": np.asarray(biomass, dtype=float).tolist(),
                "backend": backend,
            }
        except Exception as exc:
            attempt = {
                "start": start_number,
                "success": False,
                "optimizer_success": False,
                "message": f"{type(exc).__name__}: {exc}",
                "iterations": 0,
                "evaluations": evaluations,
                "objective": float("inf"),
                "theta": assemble(start).tolist(),
                "gradient": [float("nan")] * 4,
                "maximum_free_gradient": float("nan"),
                "components": {},
                "biomass": [],
                "backend": "failed",
            }
        attempts.append(attempt)
        if np.isfinite(float(attempt["objective"])) and (best_result is None or float(attempt["objective"]) < float(best_result["objective"])):
            best_result = attempt

    if best_result is None:
        best_result = attempts[0]
    best_result = dict(best_result)
    best_result["attempts"] = attempts
    best_result["multistart_objective_spread"] = float(
        np.ptp([float(row["objective"]) for row in attempts if np.isfinite(float(row["objective"]))])
    ) if any(np.isfinite(float(row["objective"])) for row in attempts) else float("nan")
    return best_result


def _profile_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    arrays = task["arrays"]
    settings = _settings_from_dict(task["settings"])
    fixed_index = int(task["fixed_index"])
    fixed_value = float(task["fixed_value"])
    result = _optimise_fixed(
        arrays,
        settings,
        task["center"],
        [tuple(pair) for pair in task["bounds"]],
        {fixed_index: fixed_value},
        int(task["multistarts"]),
        int(task["seed"]),
        int(task["max_iterations"]),
        float(task["gtol"]),
        float(task["ftol"]),
    )
    decoded = _decode(result["theta"])
    k = decoded["k"]
    biomass = np.asarray(result.get("biomass") or [], dtype=float)
    reference = _reference_points(k, decoded["r"], settings.model, settings.pella_shape)
    result.update(
        {
            "profile_index": int(task["profile_index"]),
            "fixed_transformed_value": fixed_value,
            "fixed_value": decoded[_PARAMETER_NAMES[fixed_index]],
            "parameters": decoded,
            "terminal_biomass": float(biomass[-1]) if len(biomass) else float("nan"),
            "terminal_depletion": float(biomass[-1] / k) if len(biomass) and k > 0 else float("nan"),
            "msy": float(reference["msy"]),
            "bmsy": float(reference["bmsy"]),
            "fmsy": float(reference["fmsy"]),
        }
    )
    return result


def _crossing(rows: Sequence[Mapping[str, Any]], threshold: float, side: str) -> float | None:
    valid = [row for row in rows if np.isfinite(float(row.get("delta_nll", np.nan)))]
    if not valid:
        return None
    minimum_index = min(range(len(valid)), key=lambda index: float(valid[index]["delta_nll"]))
    if side == "low":
        pairs = [(valid[index - 1], valid[index]) for index in range(minimum_index, 0, -1)]
    else:
        pairs = [(valid[index], valid[index + 1]) for index in range(minimum_index, len(valid) - 1)]
    for left, right in pairs:
        y1, y2 = float(left["delta_nll"]), float(right["delta_nll"])
        if (y1 - threshold) * (y2 - threshold) <= 0 and y1 != y2:
            x1, x2 = float(left["fixed_value"]), float(right["fixed_value"])
            fraction = (threshold - y1) / (y2 - y1)
            return float(x1 + fraction * (x2 - x1))
    return None


def profile_likelihood(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    parameter: str,
    profile_settings: ProfileSettings | None = None,
) -> dict[str, Any]:
    config = profile_settings or ProfileSettings()
    parameter_index = _parameter_index(parameter)
    center = _encode_from_fit(fitted)
    bounds = _bounds(dataset, settings)
    grid = _profile_grid(center, parameter_index, bounds, config.points, config.range_multiplier)
    arrays = _dataset_payload(dataset, settings)

    cache_payload = {
        "kind": "one_dimensional_profile",
        "dataset": arrays,
        "center": center.tolist(),
        "bounds": bounds,
        "parameter": _PARAMETER_NAMES[parameter_index],
        "grid": grid.tolist(),
        "configuration": asdict(config),
        "engine_abi": get_native_engine().status().abi_version,
    }
    cache_file = _cache_path(config.cache_dir, _cache_key(cache_payload)) if config.use_cache else None
    if cache_file and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached.setdefault("summary", {})["cache"] = "hit"
        return cached

    tasks = []
    for index, fixed_value in enumerate(grid):
        tasks.append(
            {
                "arrays": {key: value for key, value in arrays.items() if key != "settings"},
                "settings": arrays["settings"],
                "center": center.tolist(),
                "bounds": bounds,
                "fixed_index": parameter_index,
                "fixed_value": float(fixed_value),
                "profile_index": index,
                "multistarts": max(1, int(config.multistarts)),
                "seed": int(config.seed) + index * 104729,
                "max_iterations": int(config.max_iterations),
                "gtol": float(config.gtol),
                "ftol": float(config.ftol),
            }
        )

    rows: list[dict[str, Any]] = []
    if int(config.workers) > 1:
        with ProcessPoolExecutor(max_workers=min(int(config.workers), len(tasks))) as executor:
            futures = {executor.submit(_profile_worker, task): task["profile_index"] for task in tasks}
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    index = futures[future]
                    rows.append(
                        {
                            "profile_index": index,
                            "fixed_transformed_value": float(grid[index]),
                            "fixed_value": float("nan"),
                            "objective": float("inf"),
                            "success": False,
                            "message": f"{type(exc).__name__}: {exc}",
                            "components": {},
                        }
                    )
    else:
        rows = [_profile_worker(task) for task in tasks]
    rows.sort(key=lambda row: int(row["profile_index"]))

    finite_objectives = [float(row["objective"]) for row in rows if np.isfinite(float(row.get("objective", np.nan)))]
    minimum = min(finite_objectives) if finite_objectives else float("inf")
    for row in rows:
        objective = float(row.get("objective", float("inf")))
        row["delta_nll"] = objective - minimum if np.isfinite(objective) and np.isfinite(minimum) else float("nan")
        row["converged"] = bool(row.get("success")) and float(row.get("maximum_free_gradient", float("inf"))) <= 1e-3

    intervals = []
    for level in sorted(set(float(value) for value in config.confidence_levels)):
        threshold = 0.5 * float(chi2.ppf(level, df=1))
        low = _crossing(rows, threshold, "low")
        high = _crossing(rows, threshold, "high")
        intervals.append(
            {
                "confidence_level": level,
                "delta_nll_threshold": threshold,
                "low": low,
                "high": high,
                "bounded_low": low is not None,
                "bounded_high": high is not None,
                "complete": low is not None and high is not None,
            }
        )

    best_row = min(rows, key=lambda row: float(row.get("objective", float("inf")))) if rows else {}
    failed = sum(not bool(row.get("success")) for row in rows)
    nonconverged = sum(not bool(row.get("converged")) for row in rows)
    output = {
        "summary": {
            "status": "PASS" if failed == 0 and nonconverged <= max(1, len(rows) // 10) else "WARN" if failed <= max(2, len(rows) // 5) else "FAIL",
            "parameter": _PARAMETER_NAMES[parameter_index],
            "points": len(rows),
            "failed_points": failed,
            "nonconverged_points": nonconverged,
            "minimum_objective": minimum,
            "profile_mle": best_row.get("fixed_value"),
            "base_mle": _decode(center)[_PARAMETER_NAMES[parameter_index]],
            "all_other_active_parameters_refitted": True,
            "multistarts_per_point": max(1, int(config.multistarts)),
            "workers": max(1, int(config.workers)),
            "backend": get_native_engine().status().backend,
            "cache": "miss" if cache_file else "disabled",
        },
        "parameter": _PARAMETER_NAMES[parameter_index],
        "transformed_parameter": _TRANSFORMED_NAMES[parameter_index],
        "intervals": intervals,
        "profile": rows,
        "configuration": asdict(config),
        "interpretation": (
            "Each profile point fixes the selected parameter and fully re-optimises every other active production-model parameter. "
            "Failed and non-stationary points remain visible rather than being silently interpolated away."
        ),
    }
    if cache_file:
        cache_file.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    return output


def profile_likelihood_2d(
    dataset: StockDataset,
    settings: ModelSettings,
    fitted: FitResult,
    x_parameter: str,
    y_parameter: str,
    profile_settings: TwoDimensionalProfileSettings | None = None,
) -> dict[str, Any]:
    config = profile_settings or TwoDimensionalProfileSettings()
    x_index = _parameter_index(x_parameter)
    y_index = _parameter_index(y_parameter)
    if x_index == y_index:
        raise ValueError("Two-dimensional profiles require two different parameters.")
    center = _encode_from_fit(fitted)
    bounds = _bounds(dataset, settings)
    x_grid = _profile_grid(center, x_index, bounds, config.x_points, config.range_multiplier)
    y_grid = _profile_grid(center, y_index, bounds, config.y_points, config.range_multiplier)
    arrays = _dataset_payload(dataset, settings)

    tasks: list[dict[str, Any]] = []
    for x_position, x_value in enumerate(x_grid):
        for y_position, y_value in enumerate(y_grid):
            tasks.append(
                {
                    "arrays": {key: value for key, value in arrays.items() if key != "settings"},
                    "settings": arrays["settings"],
                    "center": center.tolist(),
                    "bounds": bounds,
                    "fixed": {x_index: float(x_value), y_index: float(y_value)},
                    "x_position": x_position,
                    "y_position": y_position,
                    "multistarts": max(1, int(config.multistarts)),
                    "seed": int(config.seed) + x_position * 104729 + y_position * 15485863,
                    "max_iterations": int(config.max_iterations),
                }
            )

    def execute(task: Mapping[str, Any]) -> dict[str, Any]:
        result = _optimise_fixed(
            task["arrays"],
            _settings_from_dict(task["settings"]),
            task["center"],
            [tuple(pair) for pair in task["bounds"]],
            {int(key): float(value) for key, value in task["fixed"].items()},
            int(task["multistarts"]),
            int(task["seed"]),
            int(task["max_iterations"]),
            1e-7,
            1e-12,
        )
        decoded = _decode(result["theta"])
        return {
            "x_position": int(task["x_position"]),
            "y_position": int(task["y_position"]),
            "x": decoded[_PARAMETER_NAMES[x_index]],
            "y": decoded[_PARAMETER_NAMES[y_index]],
            "objective": float(result["objective"]),
            "success": bool(result["success"]),
            "maximum_free_gradient": result.get("maximum_free_gradient"),
            "parameters": decoded,
            "components": result.get("components") or {},
            "backend": result.get("backend"),
        }

    # Use threads only for the 2D helper because the local execute closure cannot
    # be pickled reliably on Windows. Native objective work remains compiled; a
    # future task scheduler can replace this orchestration without changing the API.
    if int(config.workers) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(int(config.workers), len(tasks))) as executor:
            rows = list(executor.map(execute, tasks))
    else:
        rows = [execute(task) for task in tasks]
    finite = [float(row["objective"]) for row in rows if np.isfinite(float(row["objective"]))]
    minimum = min(finite) if finite else float("inf")
    for row in rows:
        row["delta_nll"] = float(row["objective"] - minimum) if np.isfinite(float(row["objective"])) else float("nan")
    return {
        "summary": {
            "status": "PASS" if all(bool(row["success"]) for row in rows) else "WARN",
            "x_parameter": _PARAMETER_NAMES[x_index],
            "y_parameter": _PARAMETER_NAMES[y_index],
            "grid_points": len(rows),
            "minimum_objective": minimum,
            "all_other_active_parameters_refitted": True,
        },
        "x_values": sorted({float(row["x"]) for row in rows}),
        "y_values": sorted({float(row["y"]) for row in rows}),
        "surface": rows,
        "configuration": asdict(config),
    }
