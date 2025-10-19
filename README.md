# EV Validation Data Analysis (Battery Telemetry & Fault Detection)

A simple, explainable project to:
- **Simulate** CAN/sensor telemetry (voltage, current, temperature, cell voltages)
- **Detect anomalies** via **rules** (thresholds/trends) and **Isolation Forest**
- **Visualize** signals and flags in a lightweight **Streamlit** dashboard
- **Export** anomaly reports (CSV) for quality/validation reviews

**Live demo:** https://arponroy7-ev-validation-data-analysis-app-y9scra.streamlit.app/

> Built to align with validation/quality JD requirements: field data analysis, fault/failure pattern detection, dashboards, and metrics.

---

## 🔧 Quickstart

```bash
# 1) Create and activate a venv (recommended)
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Generate a fresh synthetic dataset
python simulate.py --rows 2000 --seed 7 --out data/sample_can.csv

# 4) Run the dashboard
streamlit run app.py
```
## 📁 Project Structure

```
ev-validation-data-analysis/
│
├─ app.py           # Streamlit dashboard (load → detect → visualize → export)
├─ simulate.py      # Synthetic data generator (CSV)
├─ detect.py        # Rules + IsolationForest anomaly detection
│
├─ data/
│  └─ sample_can.csv   # Created/overwritten by simulate.py
│
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---
## 📊 What It Detects (out-of-the-box)

### 🔹 Rule-based
- **High pack temperature:** `pack_temp > 55°C`  
- **Over-current:** `abs(pack_current) > 160A`  
- **Voltage imbalance:** `cell_v_max - cell_v_min > 0.25V`  
- **Rapid temperature rise:** large `ΔT/Δt` over short windows  

### 🔹 ML-based
- **Isolation Forest** on standardized features (unsupervised)  

> Tune thresholds and model parameters directly in the Streamlit sidebar.

---

## 📦 Data Columns

If you upload your own CSV file, ensure it includes the following columns:

| Column | Description | Unit |
|---------|--------------|------|
| `time_s` | Timestamp | seconds |
| `pack_voltage` | Battery pack voltage | V |
| `pack_current` | Battery pack current | A |
| `pack_temp` | Battery pack temperature | °C |
| `cell_v_min` | Minimum cell voltage | V |
| `cell_v_max` | Maximum cell voltage | V |

You can also use **Simulate mode** within the app to generate a synthetic dataset automatically.

---

## 🧰 How to Use

1. **Choose Data Source (sidebar)**
   - 🧪 **Simulate:** Generate telemetry data on the fly  
   - 📂 **Load CSV:** Upload your own dataset  

2. **Set Rule Thresholds (sidebar)**
   - Max Temp, Max \|Current\|, Max Cell ΔV, Max dT/dt  

3. **Enable ML (optional)**
   - Toggle **Isolation Forest**  
   - Adjust **contamination** (expected anomaly ratio)  

4. **Review Metrics**
   - Dashboard displays **Rule anomalies**, **ML anomalies**, and **Total flagged points**

5. **Visualize**
   - Select signals to plot (voltage/current/temp/cell voltages)  
   - See anomaly timeline (Rule vs ML)  

6. **Export**
   - Click **Download Anomaly Report (CSV)** to save a timestamped report  

---

## ❓ Troubleshooting

| Issue | Fix |
|-------|-----|
| `streamlit: command not found` | Activate your virtual environment and reinstall deps: `pip install -r requirements.txt` |
| Port already in use | Run on another port: `streamlit run app.py --server.port 8502` |
| “Missing required columns” | Ensure your CSV includes all six required columns listed above |
| Matplotlib warnings | Safe to ignore or fix via `pip install -U matplotlib` |

---
