import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

def plot_node(df, node_name, out_dir, freq="5min"):
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    y_true = df[f"{node_name}_true"].astype(float).values
    y_pred = df[f"{node_name}_pred"].astype(float).values
    m = df[f"{node_name}_mask"].astype(float).values  # 1 valid, 0 invalid

    # invalid points become NaN so the line breaks
    y_true = np.where(m > 0.5, y_true, np.nan)
    y_pred = np.where(m > 0.5, y_pred, np.nan)

    s = pd.DataFrame(
        {"true": y_true, "pred": y_pred},
        index=df["ds"]
    )

    # reindex to full regular 5-min grid, missing timestamps become NaN and will create gaps
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    s = s.reindex(full_idx)

    # 1) time series with real time axis and gaps
    plt.figure(figsize=(14, 4))
    plt.plot(s.index, s["true"], label="True", linewidth=1.2)
    plt.plot(s.index, s["pred"], label="Pred", linewidth=1.2)
    plt.title(f"{node_name} | Step 1 Prediction")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_timeseries.png"), dpi=150)
    plt.close()

    # 2) residual time series, also breaks over gaps
    resid = s["pred"] - s["true"]
    plt.figure(figsize=(14, 3))
    plt.plot(resid.index, resid.values, linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(f"{node_name} | Residuals (Pred minus True)")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_residual_ts.png"), dpi=150)
    plt.close()

    # 3) residual distribution, drop NaNs
    r = resid.dropna().values
    plt.figure(figsize=(5, 4))
    plt.hist(r, bins=50, alpha=0.8)
    plt.title(f"{node_name} | Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_residual_hist.png"), dpi=150)
    plt.close()

    # 4) scatter, drop NaNs
    s2 = s.dropna()
    plt.figure(figsize=(5, 5))
    plt.scatter(s2["true"].values, s2["pred"].values, s=8, alpha=0.5)
    lims = [min(s2["true"].min(), s2["pred"].min()), max(s2["true"].max(), s2["pred"].max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{node_name} | True vs Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_scatter.png"), dpi=150)
    plt.close()

def main():
    csv_path = "./ST-GNN_out/results_step1.csv"
    out_dir = "./ST-GNN_out/plots_step1"

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 自动识别节点名
    node_names = sorted(
        set(col.replace("_true", "") for col in df.columns if col.endswith("_true"))
    )

    print("Found nodes:", node_names)

    for node in node_names:
        print(f"Plotting {node} ...")
        plot_node(df, node, out_dir)

    print(f"All plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
