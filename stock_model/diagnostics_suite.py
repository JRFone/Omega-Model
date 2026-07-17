from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class ReliabilityItem:
    diagnostic: str
    status: str
    value: float | str | bool | None
    threshold: str
    impact: str
    explanation: str


def lag1(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or np.std(x[:-1]) <= 0 or np.std(x[1:]) <= 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def runs_test(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return {"runs": 0, "z": 0.0, "p_approx": 1.0}
    signs = x >= np.median(x)
    runs = int(1 + np.sum(signs[1:] != signs[:-1]))
    n1 = int(signs.sum()); n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return {"runs": runs, "z": 0.0, "p_approx": 1.0}
    mean = 1 + 2 * n1 * n2 / (n1 + n2)
    variance = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - mean) / max(np.sqrt(variance), _EPS)
    p = float(np.exp(-0.717 * abs(z) - 0.416 * z * z))
    return {"runs": runs, "z": float(z), "p_approx": p}


def data_conflict_matrix(series: Mapping[str, Sequence[float]], min_overlap: int = 4) -> dict[str, Any]:
    names = list(series)
    rows = []
    correlations = np.eye(len(names), dtype=float)
    for i, left in enumerate(names):
        x = np.asarray(series[left], dtype=float)
        for j, right in enumerate(names):
            if j <= i:
                continue
            y = np.asarray(series[right], dtype=float)
            n = min(len(x), len(y))
            mask = np.isfinite(x[:n]) & np.isfinite(y[:n]) & (x[:n] > 0) & (y[:n] > 0)
            if mask.sum() < min_overlap:
                correlation = float("nan")
                slope = float("nan")
                conflict = "insufficient"
            else:
                lx = np.log(x[:n][mask]); ly = np.log(y[:n][mask])
                correlation = float(np.corrcoef(lx, ly)[0, 1]) if np.std(lx) > 0 and np.std(ly) > 0 else 0.0
                slope = float(np.polyfit(lx, ly, 1)[0]) if np.std(lx) > 0 else 0.0
                conflict = "high" if correlation < -0.25 else "moderate" if correlation < 0.25 else "low"
            correlations[i, j] = correlation
            correlations[j, i] = correlation
            rows.append({
                "series_a": left,
                "series_b": right,
                "overlap": int(mask.sum()),
                "log_correlation": correlation,
                "log_slope": slope,
                "conflict": conflict,
            })
    valid = [abs(float(row["log_correlation"])) for row in rows if np.isfinite(row["log_correlation"])]
    negative = [max(-float(row["log_correlation"]), 0.0) for row in rows if np.isfinite(row["log_correlation"])]
    score = 100.0 * float(np.mean(negative)) if negative else 0.0
    return {"series": names, "pairs": rows, "correlation_matrix": correlations.tolist(), "conflict_score_0_100": score, "mean_absolute_correlation": float(np.mean(valid)) if valid else 0.0}


def retrospective_metrics(full: Mapping[int, float], peels: Sequence[Mapping[int, float]], terminal_year: int | None = None) -> dict[str, Any]:
    if not full:
        raise ValueError("Full time series is empty.")
    terminal = terminal_year if terminal_year is not None else max(full)
    rows = []
    ratios = []
    for index, peel in enumerate(peels, 1):
        year = terminal - index
        if year not in full or year not in peel or abs(full[year]) <= _EPS:
            continue
        relative = (float(peel[year]) - float(full[year])) / float(full[year])
        ratios.append(relative)
        rows.append({"peel": index, "terminal_year": year, "full": float(full[year]), "peeled": float(peel[year]), "relative_difference": relative})
    rho = float(np.mean(ratios)) if ratios else float("nan")
    return {"mohn_rho": rho, "absolute_mohn_rho": abs(rho) if np.isfinite(rho) else float("nan"), "peels": rows}


def prior_likelihood_influence(prior_objective: Sequence[float], data_objective: Sequence[float], parameter_values: Sequence[float]) -> dict[str, Any]:
    prior = np.asarray(prior_objective, dtype=float)
    data = np.asarray(data_objective, dtype=float)
    values = np.asarray(parameter_values, dtype=float)
    mask = np.isfinite(prior) & np.isfinite(data) & np.isfinite(values)
    if mask.sum() < 3:
        raise ValueError("At least three profile points are required.")
    prior_range = float(np.ptp(prior[mask])); data_range = float(np.ptp(data[mask]))
    fraction = prior_range / max(prior_range + data_range, _EPS)
    return {
        "prior_objective_range": prior_range,
        "data_objective_range": data_range,
        "prior_influence_fraction": fraction,
        "classification": "prior-dominated" if fraction > 0.65 else "mixed" if fraction > 0.35 else "data-dominated",
        "parameter_min": float(values[mask].min()),
        "parameter_max": float(values[mask].max()),
    }


def leave_one_out_influence(base_value: float, omitted_values: Mapping[str, float]) -> list[dict[str, float | str]]:
    rows = []
    for label, value in omitted_values.items():
        delta = float(value) - float(base_value)
        rows.append({"omitted": label, "value": float(value), "absolute_change": abs(delta), "relative_change": delta / max(abs(float(base_value)), _EPS)})
    rows.sort(key=lambda row: float(row["absolute_change"]), reverse=True)
    return rows


def reliability_grade(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    items: list[ReliabilityItem] = []
    gradient_raw = diagnostics.get("maximum_gradient")
    try:
        max_gradient = float(gradient_raw) if gradient_raw is not None else None
    except (TypeError, ValueError):
        max_gradient = None
    if max_gradient is None or not np.isfinite(max_gradient):
        gradient_status = "NOT TESTED"
        max_gradient = None
    else:
        gradient_status = "PASS" if max_gradient <= 1e-3 else "WARN" if max_gradient <= 1e-2 else "FAIL"
    items.append(ReliabilityItem("Maximum gradient", gradient_status, max_gradient, "≤0.001 preferred", "High", "Large gradients indicate the optimiser may not have reached a stationary solution."))
    hessian = diagnostics.get("hessian_positive_definite")
    items.append(ReliabilityItem("Hessian", "PASS" if hessian is True else "FAIL" if hessian is False else "NOT TESTED", hessian, "Positive definite", "High", "A non-positive Hessian can indicate a saddle point, flat ridge or unquantified uncertainty."))
    rho = diagnostics.get("mohn_rho")
    rho_value = float(rho) if rho is not None and np.isfinite(rho) else None
    rho_status = "NOT TESTED" if rho_value is None else "PASS" if abs(rho_value) <= 0.15 else "WARN" if abs(rho_value) <= 0.30 else "FAIL"
    items.append(ReliabilityItem("Retrospective bias", rho_status, rho_value, "|Mohn rho| ≤0.15 preferred", "High", "Persistent retrospective revisions weaken confidence in terminal stock status."))
    holdout = diagnostics.get("holdout_relative_error")
    holdout_value = float(holdout) if holdout is not None and np.isfinite(holdout) else None
    holdout_status = "NOT TESTED" if holdout_value is None else "PASS" if holdout_value <= 0.20 else "WARN" if holdout_value <= 0.40 else "FAIL"
    items.append(ReliabilityItem("Predictive holdout", holdout_status, holdout_value, "≤20% relative error preferred", "High", "A model that cannot predict omitted observations may be fitting history without capturing process."))
    conflict = diagnostics.get("conflict_score_0_100")
    conflict_value = float(conflict) if conflict is not None and np.isfinite(conflict) else None
    conflict_status = "NOT TESTED" if conflict_value is None else "PASS" if conflict_value < 15 else "WARN" if conflict_value < 35 else "FAIL"
    items.append(ReliabilityItem("Data conflict", conflict_status, conflict_value, "<15 low, 15–35 moderate, >35 high", "High", "Strongly contradictory data sources force the model to compromise or rely on weights and assumptions."))
    condition = diagnostics.get("hessian_condition_number")
    condition_value = float(condition) if condition is not None and np.isfinite(condition) else None
    condition_status = "NOT TESTED" if condition_value is None else "PASS" if condition_value < 1e5 else "WARN" if condition_value < 1e8 else "FAIL"
    items.append(ReliabilityItem("Parameter conditioning", condition_status, condition_value, "<100,000 preferred", "Medium", "Large condition numbers indicate weakly identified parameter combinations."))
    boundaries = diagnostics.get("near_boundary") or []
    boundary_status = "PASS" if len(boundaries) == 0 else "WARN" if len(boundaries) <= 2 else "FAIL"
    items.append(ReliabilityItem("Boundary parameters", boundary_status, len(boundaries), "0 preferred", "Medium", "Boundary estimates can indicate insufficient information or restrictive parameter ranges."))
    optimiser_spread = diagnostics.get("optimizer_terminal_depletion_spread")
    optimiser_value = float(optimiser_spread) if optimiser_spread is not None and np.isfinite(optimiser_spread) else None
    optimiser_status = "NOT TESTED" if optimiser_value is None else "PASS" if optimiser_value <= 0.03 else "WARN" if optimiser_value <= 0.08 else "FAIL"
    items.append(ReliabilityItem("Optimizer agreement", optimiser_status, optimiser_value, "Terminal depletion spread ≤0.03", "High", "Different optimisers should reach materially similar solutions."))
    failures = sum(item.status == "FAIL" for item in items)
    warnings = sum(item.status == "WARN" for item in items)
    tested = sum(item.status not in {"NOT TESTED"} for item in items)
    if failures >= 3:
        grade = "F"
    elif failures == 2:
        grade = "D"
    elif failures == 1:
        grade = "C"
    elif warnings >= 3:
        grade = "C"
    elif warnings >= 1:
        grade = "B"
    else:
        grade = "A" if tested >= 6 else "B"
    label = {"A": "strong within tested scope", "B": "usable with stated cautions", "C": "conditional", "D": "low reliability", "F": "not suitable as a sole management basis"}[grade]
    return {
        "grade": grade,
        "label": label,
        "failures": failures,
        "warnings": warnings,
        "tested": tested,
        "items": [item.__dict__ for item in items],
        "disclaimer": "This grade summarises implemented diagnostics; it does not constitute independent peer review or legal/scientific certification.",
    }
