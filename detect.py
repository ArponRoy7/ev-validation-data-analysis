"""
detect.py — simple rules + IsolationForest anomaly detection
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DEFAULT_RULES = {
    "max_temp_c": 55.0,          # °C
    "max_abs_current_a": 160.0,  # A
    "max_cell_delta_v": 0.25,    # V (cell_v_max - cell_v_min)
    "max_dt_rise_c": 0.6,        # °C per second (simple derivative threshold)
}

def rule_based_flags(df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["r_temp_high"]   = df["pack_temp"] > rules["max_temp_c"]
    out["r_over_current"] = np.abs(df["pack_current"]) > rules["max_abs_current_a"]
    out["r_v_imbalance"] = (df["cell_v_max"] - df["cell_v_min"]) > rules["max_cell_delta_v"]

    # simple dT/dt
    dt = df["time_s"].diff().replace(0, np.nan).fillna(1.0)
    dT = df["pack_temp"].diff().fillna(0.0)
    out["r_fast_temp_rise"] = (dT / dt) > rules["max_dt_rise_c"]

    out["rule_any"] = out.any(axis=1)
    return out

def isolation_forest_flags(df: pd.DataFrame, contamination: float = 0.03, random_state: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    features = df[["pack_voltage", "pack_current", "pack_temp", "cell_v_min", "cell_v_max"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(X)
    scores = iso.decision_function(X)  # higher = more normal
    preds = iso.predict(X)             # -1 = anomaly, 1 = normal
    flags = (preds == -1)
    return flags, scores

def summarize_flags(rule_df: pd.DataFrame, ml_flags: np.ndarray) -> Dict[str, int]:
    return {
        "rule_anomalies": int(rule_df["rule_any"].sum()),
        "ml_anomalies": int(ml_flags.sum()),
        "total_points": int(len(rule_df)),
    }
