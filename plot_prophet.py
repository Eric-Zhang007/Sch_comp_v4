import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_from_predictions_csv(csv_path: str, out_dir: str, only_observed: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 必需列
    need = ["ds", "y_obs", "y_pred"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"missing column {c}. got={df.columns.tolist()}")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

    if only_observed:
        if "eval_mask" in df.columns:
            m = df["eval_mask"].astype(int).astype(bool)
        else:
            m = df["y_obs"].notna()
        dfp = df.loc[m].copy()
    else:
        dfp = df.copy()

    if len(dfp) == 0:
        raise ValueError("No points to plot after filtering (only_observed).")

    t = dfp["ds"].values
    y_true = dfp["y_obs"].astype(float).values
    y_pred = dfp["y_pred"].astype(float).values
    has_baseline = "baseline" in dfp.columns
    baseline = dfp["baseline"].astype(float).values if has_baseline else None

    # 1) timeseries
    plt.figure(figsize=(14, 4))
    plt.plot(t, y_true, label="True", linewidth=1.2)
    plt.plot(t, y_pred, label="Pred", linewidth=1.2)
    if has_baseline:
        plt.plot(t, baseline, label="Baseline", linewidth=1.0, alpha=0.8)
    plt.title("Time Series (True vs Pred)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timeseries.png"), dpi=150)
    plt.close()

    # 2) residual ts
    resid = y_pred - y_true
    plt.figure(figsize=(14, 3))
    plt.plot(t, resid, linewidth=1.0)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.title("Residuals (Pred - True)")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_ts.png"), dpi=150)
    plt.close()

    # 3) residual hist
    plt.figure(figsize=(5, 4))
    plt.hist(resid, bins=50, alpha=0.85)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_hist.png"), dpi=150)
    plt.close()

    # 4) scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter.png"), dpi=150)
    plt.close()

    print(f"Saved 4 plots to: {out_dir}")
    print(f"Plotted rows: {len(dfp)} (only_observed={only_observed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True, help="path to predictions.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="output dir for plots")
    ap.add_argument("--all_rows", action="store_true", help="plot all rows (ignore eval_mask)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.pred_csv)), "plots")
    plot_from_predictions_csv(args.pred_csv, out_dir, only_observed=not args.all_rows)


if __name__ == "__main__":
    main()
