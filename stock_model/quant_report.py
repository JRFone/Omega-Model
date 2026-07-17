from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


TABLE_KEYS = {
    "optimizer_candidates": ("optimizer", "candidates"),
    "optimizer_history": ("optimizer", "history"),
    "risk_pareto": ("risk_frontier", "pareto"),
    "stress_tests": ("stress_tests", "stress_tests"),
    "sobol_sensitivity": ("sobol", "sensitivity"),
    "walk_forward_folds": ("walk_forward", "folds"),
    "walk_forward_predictions": ("walk_forward", "predictions"),
    "optimizer_agreement_runs": ("optimizer_agreement", "runs"),
    "optimizer_agreement_quantities": ("optimizer_agreement", "agreement"),
    "ensemble_models": ("model_ensemble", "models"),
    "ensemble_projection": ("model_ensemble", "combined_projection"),
}


def _at(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            result[key] = json.dumps(value, separators=(",", ":"), default=str)
        else:
            result[key] = value
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = [_flatten(row) for row in rows]
    fields = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _table(rows: list[dict[str, Any]], limit: int = 100) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    fields = []
    for row in rows[:limit]:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list, tuple)):
                fields.append(key)
    head = "".join(f"<th>{html.escape(key.replace('_', ' ').title())}</th>" for key in fields)
    body = []
    for row in rows[:limit]:
        cells = []
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = "" if value is None else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _cards(summary: dict[str, Any]) -> str:
    cards = []
    for key, value in summary.items():
        if isinstance(value, (dict, list, tuple)):
            continue
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        cards.append(
            "<div class='card'><span>" + html.escape(key.replace("_", " ").title()) + "</span><strong>" + html.escape(text) + "</strong></div>"
        )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def generate_quant_report(
    payload: dict[str, Any],
    output_dir: str | Path,
    title: str = "Omega FISH Quant Lab Report",
) -> dict[str, Any]:
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "quant_lab_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    csv_files = {}
    for name, path in TABLE_KEYS.items():
        rows = _at(payload, path)
        if isinstance(rows, list):
            csv_path = folder / f"{name}.csv"
            _write_csv(csv_path, rows)
            csv_files[name] = str(csv_path)

    sections = []
    summary = payload.get("summary") or {}
    sections.append("<section><h2>Run summary</h2>" + _cards(summary) + "</section>")
    modules = [
        ("Baseline fit", payload.get("baseline_fit", {}).get("best") or {}),
        ("Global optimiser", payload.get("optimizer", {}).get("summary") or {}),
        ("Local identifiability", payload.get("optimizer", {}).get("diagnostics", {}).get("local_identifiability") or {}),
        ("Risk frontier", payload.get("risk_frontier", {}).get("summary") or {}),
        ("Stress tests", payload.get("stress_tests", {}).get("summary") or {}),
        ("Sensitivity", payload.get("sobol", {}).get("summary") or {}),
        ("Walk-forward validation", payload.get("walk_forward", {}).get("summary") or {}),
        ("Optimizer agreement", payload.get("optimizer_agreement", {}).get("summary") or {}),
        ("Model ensemble", payload.get("model_ensemble", {}).get("summary") or {}),
    ]
    for heading, values in modules:
        if values:
            sections.append(f"<section><h2>{html.escape(heading)}</h2>{_cards(values)}</section>")

    tables = [
        ("Top optimiser candidates", _at(payload, ("optimizer", "candidates"))),
        ("HCR Pareto frontier", _at(payload, ("risk_frontier", "pareto"))),
        ("Stress-test results", _at(payload, ("stress_tests", "stress_tests"))),
        ("Sensitivity indices", _at(payload, ("sobol", "sensitivity"))),
        ("Walk-forward folds", _at(payload, ("walk_forward", "folds"))),
        ("Optimizer agreement", _at(payload, ("optimizer_agreement", "runs"))),
        ("Model ensemble", _at(payload, ("model_ensemble", "models"))),
        ("Ensemble projection", _at(payload, ("model_ensemble", "combined_projection"))),
    ]
    for heading, rows in tables:
        if isinstance(rows, list) and rows:
            sections.append(f"<section><h2>{html.escape(heading)}</h2>{_table(rows)}</section>")

    created = datetime.now().isoformat(timespec="seconds")
    html_path = folder / "QUANT_LAB_REPORT.html"
    html_text = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#172126;--muted:#66767c;--line:#ced8da;--panel:#fff;--bg:#f4f7f5;--green:#136f63;--red:#9b2d21;--gold:#a66a1f}}
*{{box-sizing:border-box}} body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}}
header{{padding:22px 28px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}}
header h1{{margin:0 0 5px;font-size:24px}} header p{{margin:0;color:var(--muted)}}
main{{max-width:1500px;margin:0 auto;padding:20px}} section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}}
h2{{margin:0 0 13px;font-size:18px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:9px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:10px;min-height:70px}} .card span{{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}} .card strong{{font-size:16px;word-break:break-word}}
.scroll{{overflow:auto;max-height:620px}} table{{width:100%;border-collapse:collapse;font-size:12px}} th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}} th{{position:sticky;top:0;background:#eaf1ef;color:#43555b}}
footer{{padding:20px;color:var(--muted);font-size:12px}} .warning{{border-left:4px solid var(--gold);padding:10px 12px;background:#fff8e9;margin:12px 0}}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>Created {html.escape(created)}</p></header>
<main>
<div class='warning'><strong>Interpretation:</strong> Optimisation, candidate weights and simulation quantiles do not prove that the assessment model is correct or that uncertainty is fully characterised. Use predictive validation, identifiability diagnostics, stress tests and model disagreement together.</div>
{''.join(sections)}
</main>
<footer>Omega FISH Model — local analysis package. Raw JSON and CSV tables are stored beside this report.</footer>
</body></html>"""
    html_path.write_text(html_text, encoding="utf-8")
    return {
        "folder": str(folder),
        "html": str(html_path),
        "json": str(json_path),
        "csv": csv_files,
    }
