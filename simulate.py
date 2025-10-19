"""
simulate.py — generate synthetic CAN/sensor telemetry for EV validation.

Features:
- pack_voltage (V), pack_current (A), pack_temp (°C)
- cell_v_min (V), cell_v_max (V)
- Optional injected anomalies: temp spikes, current surges, voltage imbalance.

Usage:
  python simulate.py --rows 2000 --seed 7 --out data/sample_can.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

def simulate(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    time_s = np.arange(rows)  # 1 Hz sampling for simplicity

    # Base signals
    pack_voltage = 360 + 5 * np.sin(time_s / 60) + rng.normal(0, 0.8, rows)   # ~360-370V
    pack_current = 50 * np.sin(time_s / 30) + rng.normal(0, 5, rows)           # charge/discharge oscillation
    pack_temp    = 30 + 3 * np.sin(time_s / 120) + rng.normal(0, 0.5, rows)    # °C
    cell_v_mean  = pack_voltage / 100.0                                        # assume 100 cells in series (~3.6V each)
    spread       = np.clip(rng.normal(0.03, 0.01, rows), 0.01, 0.08)           # cell spread

    cell_v_min = cell_v_mean - spread/2
    cell_v_max = cell_v_mean + spread/2

    df = pd.DataFrame({
        "time_s": time_s,
        "pack_voltage": pack_voltage,
        "pack_current": pack_current,
        "pack_temp": pack_temp,
        "cell_v_min": cell_v_min,
        "cell_v_max": cell_v_max,
    })

      # Inject anomalies
    def inject_spikes(col, count, magnitude, width=5):
        """Add symmetric spikes without shape mismatches (uses .iloc stop-exclusive)."""
        for _ in range(count):
            # ensure we have enough room to place a 2*width segment
            if rows <= (2 * width + 100):
                continue
            idx = int(rng.integers(50, rows - 2 * width))
            start = idx
            end = min(idx + 2 * width, rows)  # stop-exclusive
            seg_len = end - start
            if seg_len <= 0:
                continue

            # build spike of length seg_len
            full_spike = np.concatenate([
                np.linspace(0, magnitude, num=width, endpoint=False),
                np.linspace(magnitude, 0, num=width, endpoint=True),
            ])
            spike = full_spike[:seg_len]

            cidx = df.columns.get_loc(col)
            df.iloc[start:end, cidx] = df.iloc[start:end, cidx].values + spike

    # Temperature spikes
    inject_spikes("pack_temp", count=max(1, rows // 1200), magnitude=20, width=8)

    # Current surges
    inject_spikes("pack_current", count=max(1, rows // 1000), magnitude=200, width=5)

    # Voltage imbalance episodes (increase spread) — use .iloc (stop-exclusive)
    for _ in range(max(1, rows // 900)):
        if rows <= 200:
            break
        start = int(rng.integers(100, rows - 60))
        span = int(rng.integers(20, 60))
        end = min(start + span, rows)  # stop-exclusive
        df.iloc[start:end, df.columns.get_loc("cell_v_min")] -= 0.15
        df.iloc[start:end, df.columns.get_loc("cell_v_max")] += 0.15

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="data/sample_can.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = simulate(rows=args.rows, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
