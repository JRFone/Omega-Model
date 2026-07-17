from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _write(figure: go.Figure, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.update_layout(
        hovermode="x unified",
        dragmode="zoom",
        template="plotly_white",
        font={"family": "Segoe UI, Arial", "size": 13},
        margin={"l": 70, "r": 35, "t": 80, "b": 60},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True, config={"scrollZoom": True, "displaylogo": False, "editable": True})
    return output


def write_biomass_truth_dashboard(result: Any, path: str | Path) -> Path:
    trajectory = result.trajectory if hasattr(result, "trajectory") else result["trajectory"]
    candidates = result.candidates if hasattr(result, "candidates") else result["candidates"]
    summary = result.summary if hasattr(result, "summary") else result["summary"]
    years = [row["year"] for row in trajectory]
    median = [row["biomass_median"] for row in trajectory]
    p10 = [row["biomass_p10"] for row in trajectory]
    p90 = [row["biomass_p90"] for row in trajectory]

    figure = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], subplot_titles=("Best-supported biomass trajectory", "Candidate evidence weights", "Terminal depletion by candidate"), vertical_spacing=0.14)
    figure.add_trace(go.Scatter(x=years, y=p90, mode="lines", line={"width": 0}, name="P90", hovertemplate="%{x}: %{y:,.2f}"), row=1, col=1)
    figure.add_trace(go.Scatter(x=years, y=p10, mode="lines", fill="tonexty", line={"width": 0}, name="P10–P90", hovertemplate="%{x}: %{y:,.2f}"), row=1, col=1)
    figure.add_trace(go.Scatter(x=years, y=median, mode="lines+markers", name="Evidence-weighted median", hovertemplate="%{x}: %{y:,.2f}"), row=1, col=1)
    figure.add_trace(go.Bar(x=[row["candidate"] for row in candidates], y=[row["weight"] for row in candidates], name="Weight"), row=2, col=1)
    figure.add_trace(go.Bar(x=[row["candidate"] for row in candidates], y=[row["terminal_depletion"] for row in candidates], name="Terminal depletion"), row=2, col=2)
    figure.update_xaxes(rangeslider={"visible": True}, row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=2)
    figure.update_yaxes(title_text="Biomass", row=1, col=1)
    figure.update_yaxes(title_text="Evidence weight", row=2, col=1)
    figure.update_yaxes(title_text="Relative depletion", row=2, col=2)
    figure.update_layout(
        title=f"Omega Biomass Evidence Synthesis — grade {summary.get('identifiability_grade', '')}<br><sup>{summary.get('statement', '')}</sup>",
        height=920,
    )
    return _write(figure, path)


def write_advanced_mse_dashboard(result: Mapping[str, Any], path: str | Path) -> Path:
    procedures = result["procedure_results"]
    scenario_rows = result["scenario_results"]
    trajectories = result.get("sample_trajectories", [])
    decision = result.get("decision_analysis", {})
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Risk–yield frontier",
            "Risk-adjusted utility",
            "Scenario risk",
            "Scenario regret",
            "Sample closed-loop trajectories",
            "Worst-case and lower-tail decision performance",
        ),
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.11,
    )
    figure.add_trace(
        go.Scatter(
            x=[row["weighted_probability_ever_below_limit"] for row in procedures],
            y=[row["weighted_median_annual_catch"] for row in procedures],
            mode="markers+text",
            text=[row["procedure"] for row in procedures],
            textposition="top center",
            marker={"size": [12 + 25 * max(row["safety_score"], 0.0) for row in procedures]},
            customdata=[
                [
                    row["weighted_catch_cv"],
                    row["weighted_assessment_rmse"],
                    row["meets_safety_constraint"],
                    row.get("maximum_regret"),
                ]
                for row in procedures
            ],
            hovertemplate=(
                "%{text}<br>P ever below limit=%{x:.3f}<br>Median catch=%{y:,.2f}"
                "<br>Catch CV=%{customdata[0]:.3f}<br>Assessment RMSE=%{customdata[1]:.3f}"
                "<br>Meets safety=%{customdata[2]}<br>Maximum regret=%{customdata[3]:.3f}<extra></extra>"
            ),
            name="Procedures",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(x=[row["procedure"] for row in procedures], y=[row["risk_adjusted_utility"] for row in procedures], name="Utility"),
        row=1,
        col=2,
    )
    procedures_names = sorted({row["procedure"] for row in scenario_rows})
    scenario_names = sorted({row["scenario"] for row in scenario_rows})
    risk_z = []
    regret_z = []
    for scenario in scenario_names:
        risk_z.append(
            [
                next(row["prob_ever_below_limit"] for row in scenario_rows if row["scenario"] == scenario and row["procedure"] == procedure)
                for procedure in procedures_names
            ]
        )
        regret_z.append(
            [
                next(float(row.get("relative_regret", float("nan"))) for row in scenario_rows if row["scenario"] == scenario and row["procedure"] == procedure)
                for procedure in procedures_names
            ]
        )
    figure.add_trace(
        go.Heatmap(
            x=procedures_names,
            y=scenario_names,
            z=risk_z,
            colorbar={"title": "P below limit", "x": 0.46},
            hovertemplate="%{y}<br>%{x}<br>Risk=%{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=procedures_names,
            y=scenario_names,
            z=regret_z,
            colorbar={"title": "Relative regret"},
            hovertemplate="%{y}<br>%{x}<br>Relative regret=%{z:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    for key in sorted({(row["procedure"], row["scenario"], row["simulation"]) for row in trajectories})[:12]:
        rows = [row for row in trajectories if (row["procedure"], row["scenario"], row["simulation"]) == key]
        figure.add_trace(
            go.Scatter(
                x=[row["year"] for row in rows],
                y=[row["true_depletion"] for row in rows],
                mode="lines",
                name=f"{key[0]} | {key[1]} | {key[2]}",
                hovertemplate="%{x}: %{y:.3f}",
            ),
            row=3,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=[row["procedure"] for row in procedures],
            y=[row.get("maximum_regret", float("nan")) for row in procedures],
            name="Maximum regret",
        ),
        row=3,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=[row["procedure"] for row in procedures],
            y=[row.get("lower_tail_scenario_utility", float("nan")) for row in procedures],
            mode="lines+markers",
            name="Lower-tail utility",
            yaxis="y6",
        ),
        row=3,
        col=2,
    )
    figure.update_xaxes(title_text="Probability ever below limit", row=1, col=1)
    figure.update_yaxes(title_text="Median annual catch", row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=1, col=2)
    figure.update_yaxes(title_text="Risk-adjusted utility", row=1, col=2)
    figure.update_xaxes(tickangle=-35, row=2, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=2)
    figure.update_yaxes(title_text="True depletion", row=3, col=1)
    figure.update_xaxes(tickangle=-35, row=3, col=2)
    figure.update_yaxes(title_text="Maximum regret / lower-tail utility", row=3, col=2)
    minimax = decision.get("minimax_regret_procedure", "not calculated")
    evpi = decision.get("expected_value_of_perfect_information", float("nan"))
    figure.update_layout(
        title=(
            f"Omega Advanced Age-Structured MSE — recommended: {result['summary']['recommended_procedure']}"
            f"<br><sup>Readiness {result['summary']['readiness_grade']} | Minimax regret: {minimax} | EVPI: {evpi:.3f}</sup>"
        ),
        height=1320,
        showlegend=True,
    )
    return _write(figure, path)


def write_experimental_diagnostics_dashboard(result: Mapping[str, Any], path: str | Path) -> Path:
    diagnostics = result["diagnostics"]
    names = []
    status_values = []
    detail_values = []
    mapping = {"PASS": 0, "CAUTION": 1, "FLAG": 2, "NOT_TESTED": -1}
    for name, value in diagnostics.items():
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status", "NOT_TESTED"))
        names.append(name.replace("_", " ").title())
        status_values.append(mapping.get(status, -1))
        detail_values.append(status)
    figure = make_subplots(rows=2, cols=1, subplot_titles=("Experimental diagnostic status", "Adversarial status sensitivity"), vertical_spacing=0.18)
    figure.add_trace(go.Bar(x=names, y=status_values, text=detail_values, textposition="outside", name="Status severity"), row=1, col=1)
    stress = diagnostics.get("adversarial_stress", {}).get("rows", [])
    figure.add_trace(go.Bar(x=[row["perturbation"] for row in stress], y=[row["relative_change"] for row in stress], name="Relative terminal-depletion change"), row=2, col=1)
    figure.update_xaxes(tickangle=-35, row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=1)
    figure.update_yaxes(title_text="-1 not tested, 0 pass, 1 caution, 2 flag", row=1, col=1)
    figure.update_yaxes(title_text="Relative change", row=2, col=1)
    figure.update_layout(title=f"Omega Experimental Diagnostics — grade {result['summary']['grade']}<br><sup>{result['summary']['boundary']}</sup>", height=820)
    return _write(figure, path)


__all__ = ["write_biomass_truth_dashboard", "write_advanced_mse_dashboard", "write_experimental_diagnostics_dashboard"]
