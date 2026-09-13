from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MODEL_PATH
from src.explainability.human_explanations import (
    build_human_explanation,
    maintenance_recommendation,
)
from src.preprocessing.data_loader import normalize_maintenance_columns
from src.preprocessing.preprocessor import add_engineered_features
from src.simulation.sensor_stream import generate_sensor_batch
from src.utils.io import load_model, model_path_candidates


st.set_page_config(page_title="Explainable Predictive Maintenance", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(37, 99, 235, 0.20), transparent 28%),
            radial-gradient(circle at 86% 12%, rgba(20, 184, 166, 0.16), transparent 26%),
            linear-gradient(135deg, #070b12 0%, #101827 48%, #07111f 100%);
        color: #f8fafc;
    }
    [data-testid="stHeader"] {
        background: rgba(7, 11, 18, 0.2);
    }
    .block-container {
        padding-top: 3.2rem;
        max-width: 1280px;
    }
    .glass-panel {
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.055));
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
        border-radius: 24px;
        padding: 26px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin: 18px 0 26px;
    }
    .glass-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 780;
        margin-bottom: 4px;
    }
    .glass-subtitle {
        color: rgba(226, 232, 240, 0.78);
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .machine-pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(125, 211, 252, 0.36);
        background: rgba(14, 165, 233, 0.14);
        color: #bae6fd;
        border-radius: 999px;
        padding: 8px 13px;
        font-weight: 750;
        letter-spacing: 0;
        margin-bottom: 16px;
    }
    .decision-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-bottom: 18px;
    }
    .decision-tile {
        border: 1px solid rgba(255, 255, 255, 0.13);
        background: rgba(15, 23, 42, 0.48);
        border-radius: 18px;
        padding: 16px;
        min-height: 96px;
    }
    .tile-label {
        color: rgba(203, 213, 225, 0.72);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .tile-value {
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .risk-high {
        color: #fecaca;
    }
    .risk-normal {
        color: #bbf7d0;
    }
    .action-strip {
        border-radius: 18px;
        padding: 18px;
        margin-top: 12px;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16), rgba(245, 158, 11, 0.10));
        border: 1px solid rgba(248, 113, 113, 0.22);
    }
    .action-strip.normal {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.14), rgba(20, 184, 166, 0.09));
        border: 1px solid rgba(74, 222, 128, 0.22);
    }
    .action-heading {
        color: #f8fafc;
        font-weight: 780;
        margin-bottom: 8px;
    }
    .action-body {
        color: rgba(241, 245, 249, 0.88);
        line-height: 1.55;
    }
    .analysis-hero {
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.16), rgba(20, 184, 166, 0.08));
        box-shadow: 0 18px 56px rgba(0, 0, 0, 0.28);
        border-radius: 24px;
        padding: 24px;
        margin: 16px 0 20px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }
    .analysis-kpis {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin-top: 18px;
    }
    .analysis-kpi {
        border: 1px solid rgba(255, 255, 255, 0.13);
        background: rgba(15, 23, 42, 0.42);
        border-radius: 18px;
        padding: 15px;
    }
    .chart-card {
        border: 1px solid rgba(255, 255, 255, 0.14);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.13), rgba(255, 255, 255, 0.05));
        border-radius: 22px;
        padding: 18px 18px 8px;
        box-shadow: 0 18px 52px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        margin-bottom: 18px;
    }
    .chart-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 780;
        margin-bottom: 2px;
    }
    .chart-caption {
        color: rgba(203, 213, 225, 0.76);
        font-size: 0.86rem;
        margin-bottom: 10px;
    }
    @media (max-width: 900px) {
        .decision-grid,
        .analysis-kpis {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 560px) {
        .decision-grid,
        .analysis-kpis {
            grid-template-columns: 1fr;
        }
        .glass-panel,
        .analysis-hero {
            padding: 18px;
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Explainable Predictive Maintenance")
st.caption("Industrial failure prediction with transparent maintenance reasoning")


@st.cache_resource
def get_model(path: Path):
    return load_model(path)


def failure_probabilities(model, features: pd.DataFrame) -> list[float | None]:
    if not hasattr(model, "predict_proba"):
        return [None] * len(features)

    probabilities = model.predict_proba(features)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        failure_index = classes.index(1)
        return probabilities[:, failure_index].tolist()
    if probabilities.shape[1] == 2:
        return probabilities[:, 1].tolist()
    return probabilities.max(axis=1).tolist()


def confidence_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def add_machine_ids(data: pd.DataFrame) -> pd.DataFrame:
    if "machine_id" in data.columns:
        return data
    output = data.copy()
    output.insert(0, "machine_id", [f"MCH-{idx + 1:03d}" for idx in range(len(output))])
    return output


def render_decision_card(
    machine_id: str,
    status: str,
    probability: str,
    recommendation: dict[str, str],
    explanation: str,
) -> None:
    is_risk = status == "Failure risk"
    status_class = "risk-high" if is_risk else "risk-normal"
    strip_class = "" if is_risk else " normal"
    st.markdown(
        f"""
        <div class="glass-panel">
            <div class="machine-pill">{machine_id}</div>
            <div class="glass-title">Selected Machine Maintenance Decision</div>
            <div class="glass-subtitle">Model prediction, likely fault area, and technician action in one view.</div>
            <div class="decision-grid">
                <div class="decision-tile">
                    <div class="tile-label">Predicted status</div>
                    <div class="tile-value {status_class}">{status}</div>
                </div>
                <div class="decision-tile">
                    <div class="tile-label">Failure probability</div>
                    <div class="tile-value">{probability}</div>
                </div>
                <div class="decision-tile">
                    <div class="tile-label">Priority</div>
                    <div class="tile-value">{recommendation["priority"]}</div>
                </div>
                <div class="decision-tile">
                    <div class="tile-label">Suspected issue</div>
                    <div class="tile-value" style="font-size:1.02rem;">{recommendation["suspected_issue"]}</div>
                </div>
            </div>
            <div class="action-strip{strip_class}">
                <div class="action-heading">What to repair or check</div>
                <div class="action-body">{recommendation["repair_focus"]}</div>
                <div class="action-heading" style="margin-top:14px;">Recommended technician action</div>
                <div class="action-body">{recommendation["recommended_action"]}</div>
                <div class="action-heading" style="margin-top:14px;">Why the model thinks this</div>
                <div class="action-body">{explanation}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_card(title: str, caption: str):
    st.markdown(
        f"""
        <div class="chart-card">
            <div class="chart-title">{title}</div>
            <div class="chart-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def close_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_graphs(result: pd.DataFrame) -> None:
    failure_count = int(result["prediction"].sum())
    normal_count = int(len(result) - failure_count)
    mean_risk = confidence_label(result["failure_probability"].dropna().mean())
    highest_risk = confidence_label(result["failure_probability"].dropna().max())

    st.markdown(
        f"""
        <div class="analysis-hero">
            <div class="glass-title">Maintenance Analytics Command View</div>
            <div class="glass-subtitle">A visual summary of machine health, failure probability, sensor behavior, and likely maintenance issues.</div>
            <div class="analysis-kpis">
                <div class="analysis-kpi">
                    <div class="tile-label">Machines analyzed</div>
                    <div class="tile-value">{len(result)}</div>
                </div>
                <div class="analysis-kpi">
                    <div class="tile-label">Failure-risk machines</div>
                    <div class="tile-value risk-high">{failure_count}</div>
                </div>
                <div class="analysis-kpi">
                    <div class="tile-label">Normal machines</div>
                    <div class="tile-value risk-normal">{normal_count}</div>
                </div>
                <div class="analysis-kpi">
                    <div class="tile-label">Highest risk</div>
                    <div class="tile-value">{highest_risk}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    graph_left, graph_right = st.columns(2)
    with graph_left:
        chart_card(
            "Fleet Health Split",
            f"Current batch risk distribution. Mean failure probability: {mean_risk}.",
        )
        status_counts = result["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.pie(
            status_counts,
            names="status",
            values="count",
            title="Normal vs Failure-Risk Machines",
            color="status",
            color_discrete_map={"Normal": "#22c55e", "Failure risk": "#ef4444"},
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            title_font_color="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
        close_card()

    with graph_right:
        chart_card(
            "Maintenance Issue Mix",
            "Counts the likely root maintenance areas among predicted failure-risk machines.",
        )
        issue_counts = (
            result[result["prediction"].astype(int) == 1]["suspected_issue"]
            .value_counts()
            .reset_index()
        )
        issue_counts.columns = ["suspected_issue", "count"]
        if issue_counts.empty:
            st.info("No failure-risk rows available for issue analysis.")
        else:
            fig = px.bar(
                issue_counts,
                x="count",
                y="suspected_issue",
                orientation="h",
                title="Likely Maintenance Issues",
                color="count",
                color_continuous_scale="Reds",
            )
            fig.update_layout(yaxis_title="", xaxis_title="Machines")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                title_font_color="#f8fafc",
            )
            st.plotly_chart(fig, use_container_width=True)
        close_card()

    sensor_cols = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    chart_card(
        "Sensor Condition by Prediction Status",
        "Compares average operating conditions for normal and failure-risk machines.",
    )
    sensor_summary = result.groupby("status")[sensor_cols].mean().reset_index()
    sensor_long = sensor_summary.melt(
        id_vars="status",
        value_vars=sensor_cols,
        var_name="sensor",
        value_name="average_value",
    )
    fig = px.bar(
        sensor_long,
        x="sensor",
        y="average_value",
        color="status",
        barmode="group",
        title="Average Sensor Values for Normal vs Failure-Risk Machines",
        color_discrete_map={"Normal": "#22c55e", "Failure risk": "#ef4444"},
    )
    fig.update_layout(xaxis_title="", yaxis_title="Average sensor value")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#f8fafc",
    )
    st.plotly_chart(fig, use_container_width=True)
    close_card()

    chart_card(
        "Highest Risk Machines",
        "Ranks machines by predicted failure probability for maintenance prioritization.",
    )
    top_risk = result.sort_values("failure_probability", ascending=False).head(15)
    fig = px.bar(
        top_risk,
        x="machine_id",
        y="failure_probability",
        color="status",
        title="Top Machines by Failure Probability",
        hover_data=["suspected_issue", "repair_focus"],
        color_discrete_map={"Normal": "#22c55e", "Failure risk": "#ef4444"},
    )
    fig.update_layout(xaxis_title="Machine ID", yaxis_title="Failure probability")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#f8fafc",
    )
    st.plotly_chart(fig, use_container_width=True)
    close_card()

    chart_card(
        "Sensor Trend Lines",
        "Shows how critical machine parameters vary across uploaded machine records.",
    )
    trend_data = result[["machine_id", *sensor_cols]].copy()
    trend_data["row"] = range(len(trend_data))
    trend_long = trend_data.melt(
        id_vars=["row", "machine_id"],
        value_vars=sensor_cols,
        var_name="sensor",
        value_name="value",
    )
    fig = px.line(
        trend_long,
        x="row",
        y="value",
        color="sensor",
        title="Sensor Trends Across Uploaded Machine Records",
        hover_data=["machine_id"],
    )
    fig.update_layout(xaxis_title="Dataset row", yaxis_title="Sensor value")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        title_font_color="#f8fafc",
    )
    st.plotly_chart(fig, use_container_width=True)
    close_card()

    if "failure_type" in result.columns:
        chart_card(
            "Failure Type Distribution",
            "Uses dataset labels, when available, to show the mix of fault categories.",
        )
        failure_type_counts = result["failure_type"].value_counts().reset_index()
        failure_type_counts.columns = ["failure_type", "count"]
        fig = px.bar(
            failure_type_counts,
            x="failure_type",
            y="count",
            title="Failure Type Mix in Uploaded Dataset",
            color="count",
            color_continuous_scale="Bluered",
        )
        fig.update_layout(xaxis_title="Failure type", yaxis_title="Records")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e5e7eb",
            title_font_color="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
        close_card()


MODEL_PATH = next((path for path in model_path_candidates(DEFAULT_MODEL_PATH) if path.exists()), None)

if MODEL_PATH is None:
    st.warning("No trained model found. Train a model first with `python main.py train ...`.")
    data = generate_sensor_batch(25)
else:
    model = get_model(MODEL_PATH)
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    data = (
        normalize_maintenance_columns(pd.read_csv(uploaded))
        if uploaded
        else generate_sensor_batch(25).drop(columns=["failure_binary"])
    )
    data = add_machine_ids(data)

    features = add_engineered_features(data)
    predictions = model.predict(features)
    probabilities = failure_probabilities(model, features)

    result = data.copy()
    result["prediction"] = predictions
    result["status"] = ["Failure risk" if int(pred) == 1 else "Normal" for pred in predictions]
    result["failure_probability"] = probabilities
    recommendations = [
        maintenance_recommendation(features.iloc[idx], int(pred))
        for idx, pred in enumerate(predictions)
    ]
    result["priority"] = [item["priority"] for item in recommendations]
    result["suspected_issue"] = [item["suspected_issue"] for item in recommendations]
    result["recommended_action"] = [item["recommended_action"] for item in recommendations]
    result["repair_focus"] = [item["repair_focus"] for item in recommendations]

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Rows scored", len(result))
    kpi2.metric("Failure-risk rows", int(result["prediction"].sum()))
    kpi3.metric("Mean failure probability", confidence_label(pd.Series(probabilities).dropna().mean()))

    prediction_tab, analysis_tab = st.tabs(["Prediction & Maintenance", "Analysis Graphs"])

    with prediction_tab:
        selected = st.slider("Inspect machine row", 0, max(len(result) - 1, 0), 0)
        selected_row = result.iloc[selected]
        selected_features = features.iloc[selected]
        selected_status = str(selected_row["status"])
        selected_probability = selected_row["failure_probability"]
        selected_recommendation = maintenance_recommendation(
            selected_features,
            int(selected_row["prediction"]),
        )

        selected_explanation = build_human_explanation(selected_features)
        render_decision_card(
            machine_id=str(selected_row["machine_id"]),
            status=selected_status,
            probability=confidence_label(selected_probability),
            recommendation=selected_recommendation,
            explanation=selected_explanation,
        )

        high_risk = result[result["prediction"].astype(int) == 1]
        st.subheader("Machines Requiring Maintenance")
        if high_risk.empty:
            st.info("No machines currently require maintenance based on this batch.")
        else:
            st.dataframe(
                high_risk[
                    [
                        "machine_id",
                        "status",
                        "failure_probability",
                        "priority",
                        "suspected_issue",
                        "repair_focus",
                        "recommended_action",
                    ]
                ],
                use_container_width=True,
            )

        st.subheader("Prediction Table")
        st.dataframe(result, use_container_width=True)

    with analysis_tab:
        render_analysis_graphs(result)
