import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

def plot_node(df, node_name, out_dir):
    y_true = df[f"{node_name}_true"].values
    y_pred = df[f"{node_name}_pred"].values
    mask = df[f"{node_name}_mask"].values.astype(bool)

    # 只画有效点
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    t = np.arange(len(y_true))

    # 1. 时间序列对比
    plt.figure(figsize=(14, 4))
    plt.plot(t, y_true, label="True", linewidth=1.2)
    plt.plot(t, y_pred, label="Pred", linewidth=1.2)
    plt.title(f"{node_name} – Step 1 Prediction")
    plt.xlabel("Time Index (5-min)")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_timeseries.png"), dpi=150)
    plt.close()

    # 2. 残差时间序列
    resid = y_pred - y_true
    plt.figure(figsize=(14, 3))
    plt.plot(t, resid, color="tab:red", linewidth=1.0)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(f"{node_name} – Residuals (Pred − True)")
    plt.xlabel("Time Index")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_residual_ts.png"), dpi=150)
    plt.close()

    # 3. 残差分布
    plt.figure(figsize=(5, 4))
    plt.hist(resid, bins=50, alpha=0.8)
    plt.title(f"{node_name} – Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_residual_hist.png"), dpi=150)
    plt.close()

    # 4. True vs Pred 散点
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{node_name} – True vs Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{node_name}_scatter.png"), dpi=150)
    plt.close()


def main():
    csv_path = "./results_step1.csv"
    out_dir = "plots_step1"

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
