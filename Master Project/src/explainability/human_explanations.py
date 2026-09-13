from __future__ import annotations

import pandas as pd


RISK_FEATURES = {
    "torque": "high torque may indicate excessive load or friction",
    "tool_wear": "tool wear suggests the cutting tool is near end-of-life",
    "temperature_delta": "a high temperature gap can indicate thermal stress",
    "rotational_speed": "abnormal speed can contribute to power or overstrain faults",
    "mechanical_power": "elevated mechanical power can indicate overload",
}


def build_human_explanation(row: pd.Series, top_features: list[str] | None = None) -> str:
    features = top_features or []
    reasons = []
    for feature in features:
        if feature in RISK_FEATURES:
            reasons.append(RISK_FEATURES[feature])

    if not reasons:
        if row.get("torque", 0) > 50 and row.get("tool_wear", 0) > 180:
            reasons.append("high torque combined with excessive tool wear increased risk")
        elif row.get("tool_wear", 0) > 180:
            reasons.append("tool wear is close to a maintenance threshold")
        else:
            reasons.append("the current sensor pattern is similar to historical failure cases")

    return "Machine failure risk is driven by " + "; ".join(reasons) + "."


def maintenance_recommendation(row: pd.Series, prediction: int) -> dict[str, str]:
    """Translate a model prediction and sensor state into maintenance guidance."""
    if int(prediction) == 0:
        return {
            "priority": "Low",
            "suspected_issue": "No immediate fault pattern detected",
            "recommended_action": "Continue monitoring",
            "repair_focus": "No repair required now",
        }

    torque = float(row.get("torque", 0))
    tool_wear = float(row.get("tool_wear", 0))
    speed = float(row.get("rotational_speed", 0))
    temp_delta = float(row.get("temperature_delta", 0))

    if tool_wear >= 220 and torque >= 55:
        issue = "Tool wear with mechanical overload"
        action = "Inspect or replace cutting tool; check load, alignment, and lubrication"
        focus = "Tool, spindle load, bearings, lubrication"
    elif tool_wear >= 200:
        issue = "Tool wear near end-of-life"
        action = "Schedule tool replacement before the next production cycle"
        focus = "Cutting tool or wear component"
    elif torque >= 58:
        issue = "Excessive torque or overstrain"
        action = "Check for overload, jammed workpiece, misalignment, or bearing friction"
        focus = "Drive train, bearings, shaft alignment, load balance"
    elif temp_delta >= 13:
        issue = "Thermal stress or heat dissipation issue"
        action = "Inspect cooling, airflow, lubrication, and heat dissipation path"
        focus = "Cooling system, lubrication, airflow, temperature control"
    elif speed >= 2700 or speed <= 1200:
        issue = "Abnormal rotational speed pattern"
        action = "Inspect motor controller, belt/gear condition, and speed feedback sensor"
        focus = "Motor, VFD/controller, belt, gear, speed sensor"
    else:
        issue = "Combined abnormal sensor pattern"
        action = "Perform detailed inspection and compare with recent maintenance history"
        focus = "General machine health inspection"

    return {
        "priority": "High",
        "suspected_issue": issue,
        "recommended_action": action,
        "repair_focus": focus,
    }
