import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True, help="path to predictions.csv")
    ap.add_argument("--out_dir", type=str, default=None, help="where to save pngs")
    ap.add_argument("--only_observed", action="store_true", help="filter mask_missing=0 mask_hard=0 and y_obs notna")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)

    required = ["y_obs", "y_pred", "baseline", "mask_missing", "mask_hard"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}. got={df.columns.tolist()}")

    if args.only_observed:
        m = (df["mask_missing"] == 0) & (df["mask_hard"] == 0) & df["y_obs"].notna()
        d = df.loc[m].copy()
    else:
        d = df.copy()

    d = d.reset_index(drop=True)
    x = np.arange(len(d))

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.pred_csv)), "plots_rowindex")
    os.makedirs(out_dir, exist_ok=True)

    y_obs = d["y_obs"].astype(float).values
    y_pred = d["y_pred"].astype(float).values
    baseline = d["baseline"].astype(float).values
    err = y_pred - y_obs

    fig = plt.figure()
    plt.plot(x, y_obs, label="y_obs")
    plt.plot(x, y_pred, label="y_pred")
    plt.plot(x, baseline, label="baseline")
    plt.xlabel("row index")
    plt.ylabel("value")
    plt.title("y_obs vs y_pred vs baseline")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "timeseries_rowindex.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x, err)
    plt.xlabel("row index")
    plt.ylabel("y_pred - y_obs")
    plt.title("error timeseries")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_timeseries_rowindex.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(y_obs, y_pred, s=6)
    mn = float(min(np.nanmin(y_obs), np.nanmin(y_pred)))
    mx = float(max(np.nanmax(y_obs), np.nanmax(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("y_obs")
    plt.ylabel("y_pred")
    plt.title("scatter y_obs vs y_pred")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_rowindex.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.hist(err, bins=60)
    plt.xlabel("y_pred - y_obs")
    plt.ylabel("count")
    plt.title("error histogram")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_hist_rowindex.png"), dpi=150)
    plt.close(fig)

    print(f"saved plots to: {out_dir}")
    print(f"rows plotted: {len(d)} (only_observed={args.only_observed})")

if __name__ == "__main__":
    main()
