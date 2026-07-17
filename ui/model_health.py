from __future__ import annotations

import math
from typing import Any, Mapping


def assess_model_health(test_name: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """Create a conservative visual verdict without equating execution with accuracy."""

    label = test_name.replace("Running ", "").replace("...", "").strip()
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    status = str(summary.get("status", "UNKNOWN")).upper()
    lowered = label.lower()

    if "profile" in lowered or "profile" in payload:
        failed = int(summary.get("failed_points", 0) or 0)
        nonconverged = int(summary.get("nonconverged_points", 0) or 0)
        deltas = [
            float(row.get("delta_nll"))
            for row in payload.get("profile", [])
            if isinstance(row, Mapping) and _finite(row.get("delta_nll"))
        ]
        flat = bool(deltas) and max(deltas) < 1.92
        if failed or nonconverged > max(1, int(summary.get("points", 0) or 0) // 10):
            verdict = "NUMERICAL WARNING"
            confounding = "Cannot judge until convergence improves"
            reason = f"{failed} failed and {nonconverged} non-converged profile points."
            action = "Increase multistarts/workload and inspect failed points."
        elif flat:
            verdict = "POSSIBLE CONFOUNDING"
            confounding = "High — profile is too flat over the tested range"
            reason = "The profile never rises by the usual 95% one-parameter ΔNLL threshold of 1.92."
            action = "Add informative data, widen the profile range, and inspect parameter correlations."
        else:
            verdict = "PARAMETER IDENTIFIED" if status == "PASS" else "WEAK IDENTIFICATION"
            confounding = "Low in this one-dimensional profile" if status == "PASS" else "Possible"
            reason = "The refitted profile has curvature and usable optimisation points."
            action = "Check other parameters and two-dimensional profiles before claiming identifiability."
        accuracy = "Not tested — profile shape measures identifiability, not real-world accuracy"
    elif "aspm" in lowered or "maximum_informative_terminal_difference" in summary:
        difference = summary.get("maximum_informative_terminal_difference")
        if status == "PASS":
            verdict = "STRUCTURALLY ROBUST"
            confounding = "Low sensitivity to removing composition likelihoods"
        elif status == "WARN":
            verdict = "POSSIBLE CONFOUNDING"
            confounding = "Moderate structural sensitivity"
        else:
            verdict = "STRUCTURALLY SENSITIVE"
            confounding = "High — trajectories depend materially on information source"
        reason = f"Maximum informative terminal-depletion difference: {_display(difference)}."
        accuracy = "Not absolute accuracy — this compares model structures"
        action = "Inspect composition influence, index influence, selectivity, and biology assumptions."
    elif "coverage" in lowered or "maximum_absolute_coverage_error" in summary:
        error = summary.get("maximum_absolute_coverage_error")
        bias = summary.get("maximum_absolute_mean_relative_bias")
        failures = summary.get("failure_fraction")
        if status == "PASS":
            verdict = "INTERVAL ACCURACY SUPPORTED"
            accuracy = "Supported in this known-truth simulation"
        elif status == "WARN":
            verdict = "QUESTIONABLE INTERVAL ACCURACY"
            accuracy = "Mixed known-truth recovery"
        else:
            verdict = "INACCURATE / UNRELIABLE INTERVALS"
            accuracy = "Not supported by this known-truth test"
        confounding = "Not isolated by coverage alone"
        reason = f"Max coverage error {_display(error)}; max relative bias {_display(bias)}; failed-fit fraction {_display(failures)}."
        action = "Increase formal replicates, fix failed fits, and inspect bias by parameter."
    elif status == "PASS":
        verdict = "SOFTWARE CHECK PASSED"
        accuracy = "Scientific accuracy not established"
        confounding = "Not assessed by this test"
        reason = "The selected software or parity check met its programmed criteria."
        action = "Use profiles, ASPM, coverage, residuals, and sensitivity tests for model evidence."
    elif status == "WARN":
        verdict = "REVIEW REQUIRED"
        accuracy = "Not established"
        confounding = "Unknown"
        reason = "The test completed with warnings."
        action = "Open the detailed evidence and resolve warnings before interpretation."
    else:
        verdict = "TEST FAILED"
        accuracy = "Not supported"
        confounding = "Cannot judge reliably"
        reason = "The test did not meet its programmed criteria."
        action = "Inspect detailed failures; do not hide or average them away."

    return {
        "test": label,
        "quick_verdict": verdict,
        "accuracy_evidence": accuracy,
        "confounding_risk": confounding,
        "reason": reason,
        "next_action": action,
        "test_status": status,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _display(value: Any) -> str:
    if not _finite(value):
        return "not available"
    return f"{float(value):.4g}"
