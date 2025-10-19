"""
Streamlit dashboard for EV Validation Data Analysis
- Load CSV or generate new data
- Run rule-based + IsolationForest detection
- Visualize signals and anomalies
- Export anomaly report CSV
"""

import io
import time
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from detect import DEFAULT_RULES, rule_based_flags, isolation_forest_flags
from simulate import simulate

st.set_page_config(page_title="EV Validation Data Analysis", layout="wide")

st.title("🔧 EV Validation Data Analysis — Battery Telemetry & Fault Detection")

with st.sidebar:
    st.header("Data")
    mode = st.radio("Data Source", ["Load CSV", "Simulate"], index=1)

    if mode == "Load CSV":
        file = st.file_uploader("Upload CAN/telemetry CSV", type=["csv"])
    else:
        rows = st.slider("Rows", 500, 10000, 2000, 100)
        seed = st.number_input("Seed", value=7, step=1)

    st.header("Rule Thresholds")
    max_temp = st.number_input("Max Temp (°C)", value=float(DEFAULT_RULES["max_temp_c"]))
    max_abs_current = st.number_input("Max |Current| (A)", value=float(DEFAULT_RULES["max_abs_current_a"]))
    max_dv = st.number_input("Max Cell ΔV (V)", value=float(DEFAULT_RULES["max_cell_delta_v"]))
    max_dt = st.number_input("Max dT/dt (°C/s)", value=float(DEFAULT_RULES["max_dt_rise_c"]))

    st.header("ML (Isolation Forest)")
    use_ml = st.checkbox("Enable IsolationForest", value=True)
    contamination = st.slider("Contamination", 0.005, 0.1, 0.03, 0.005)

rules = {
    "max_temp_c": max_temp,
    "max_abs_current_a": max_abs_current,
    "max_cell_delta_v": max_dv,
    "max_dt_rise_c": max_dt,
}

# Load or simulate
if mode == "Load CSV" and "file" in locals() and file is not None:
    df = pd.read_csv(file)
    st.success(f"Loaded {len(df):,} rows from uploaded file.")
elif mode == "Simulate":
    df = simulate(rows=int(rows), seed=int(seed))
    st.success(f"Simulated {len(df):,} rows (seed={seed}).")
else:
    st.info("Upload a CSV or switch to Simulate.")
    st.stop()

# Ensure required columns
required_cols = ["time_s", "pack_voltage", "pack_current", "pack_temp", "cell_v_min", "cell_v_max"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Compute detections
rule_flags = rule_based_flags(df, rules)
if use_ml:
    ml_flags, ml_scores = isolation_forest_flags(df, contamination=contamination)
else:
    ml_flags = np.zeros(len(df), dtype=bool)
    ml_scores = np.zeros(len(df), dtype=float)

summary = {
    "Rule anomalies": int(rule_flags["rule_any"].sum()),
    "ML anomalies": int(ml_flags.sum()),
    "Total points": len(df),
}

col1, col2, col3 = st.columns(3)
col1.metric("Rule anomalies", f'{summary["Rule anomalies"]:,}')
col2.metric("ML anomalies", f'{summary["ML anomalies"]:,}')
col3.metric("Total points", f'{summary["Total points"]:,}')

# Merge results for export
out = df.copy()
out = out.join(rule_flags)
out["ml_anomaly"] = ml_flags
out["ml_score"] = ml_scores

# Plots
st.subheader("Signals")
sig_cols = ["pack_voltage", "pack_current", "pack_temp", "cell_v_min", "cell_v_max"]
sel = st.multiselect("Choose signals to plot:", sig_cols, default=["pack_voltage", "pack_current", "pack_temp"])

fig, ax = plt.subplots(figsize=(10, 4))
for c in sel:
    ax.plot(df["time_s"], df[c], label=c)
ax.set_xlabel("time_s")
ax.legend(loc="upper right")
st.pyplot(fig, clear_figure=True)

st.subheader("Anomaly Timeline")
timeline = pd.DataFrame({
    "time_s": df["time_s"],
    "Rule": rule_flags["rule_any"].astype(int),
    "ML": ml_flags.astype(int),
})
st.line_chart(timeline.set_index("time_s"))

# Export button
st.subheader("Export")
csv_bytes = out.to_csv(index=False).encode("utf-8")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    label="Download Anomaly Report (CSV)",
    data=csv_bytes,
    file_name=f"anomaly_report_{ts}.csv",
    mime="text/csv",
)

# Show sample anomalies
st.subheader("Sample Anomalies (Top 10)")
sample = out[(out["rule_any"] | out["ml_anomaly"])].head(10)
if len(sample) == 0:
    st.write("No anomalies flagged with current settings.")
else:
    st.dataframe(sample)
