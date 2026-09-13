from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


STYLE = {
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#d0d7de",
    "grid.color": "#eaeef2",
    "font.size": 11,
}


def _prepare(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", rc=STYLE)


def plot_failure_distribution(data: pd.DataFrame, output_path: Path) -> None:
    _prepare(output_path)
    plt.figure(figsize=(7, 4))
    sns.countplot(data=data, x="failure_type", color="#2867b2")
    plt.title("Failure Type Distribution")
    plt.xlabel("Failure type")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    _prepare(output_path)
    numeric = data.select_dtypes(include="number")
    plt.figure(figsize=(9, 6))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=False, linewidths=0.5)
    plt.title("Sensor Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    _prepare(output_path)
    plot_data = metrics.melt(id_vars="model", var_name="metric", value_name="score")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_data, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.title("Comparative Model Performance")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(matrix: list[list[int]], output_path: Path) -> None:
    _prepare(output_path)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sensor_trends(data: pd.DataFrame, output_path: Path) -> None:
    _prepare(output_path)
    sensor_cols = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]
    plt.figure(figsize=(12, 6))
    data[sensor_cols].reset_index(drop=True).plot(ax=plt.gca())
    plt.title("Sensor Trend Overview")
    plt.xlabel("Observation")
    plt.ylabel("Sensor value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
