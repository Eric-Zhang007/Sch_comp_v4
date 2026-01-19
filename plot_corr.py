import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
import itertools

# =========================
# Global style, clearer and cleaner
# =========================
sns.set_theme(style="ticks", context="paper", font_scale=1.15)
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300

BUILDINGS = ["Commercial", "Office", "Public", "Residential"]
PAIRS = list(itertools.combinations(BUILDINGS, 2))

SHORT = {"Commercial": "Com", "Office": "Off", "Public": "Pub", "Residential": "Res"}

RES_CONF = {
    "5min": {"slots": 288, "lag_max": 288},   # +/- 1 day
    "30min": {"slots": 48, "lag_max": 48},    # +/- 1 day
    "1hour": {"slots": 24, "lag_max": 24},    # +/- 1 day
}

# Color palette, stable and readable
PAIR_LABELS = [f"{SHORT[a]}-{SHORT[b]}" for a, b in PAIRS]
PAIR_COLORS = dict(zip(PAIR_LABELS, sns.color_palette("tab10", len(PAIRS))))

# Preferred residual columns from preprocess
PREFERRED_RESID_COLS = ["stl_resid", "y_stl_resid", "stl_residual", "resid_stl", "resid"]


def make_seasonal_key(index, res, slots_per_day):
    if res == "5min":
        slot = ((index.hour * 60 + index.minute) // 5).astype(int)
    elif res == "30min":
        slot = ((index.hour * 60 + index.minute) // 30).astype(int)
    else:
        slot = index.hour.astype(int)
    dow = index.dayofweek.astype(int)
    return dow * slots_per_day + slot


def get_residuals_fast(df, train_end, res):
    """
    Fallback residual construction via seasonal median baseline on train split.
    Used only if parquet does not provide STL residual.
    """
    train_mask = df.index <= train_end
    slots = RES_CONF[res]["slots"]
    residuals = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        series = df[col].astype(float)
        train_series = series[train_mask].dropna()
        if train_series.empty:
            continue

        key_train = make_seasonal_key(train_series.index, res, slots)
        baseline = train_series.groupby(key_train).median()

        key_all = make_seasonal_key(series.index, res, slots)
        residuals[col] = series - key_all.map(baseline)

    return residuals


def _pick_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_series_from_parquet(fp: Path, prefer_resid=True):
    """
    Load one building series from parquet.
    Priority: STL residual column if present.
    Fallback: y_imputed, then y_clean, then y_raw.
    """
    df = pd.read_parquet(fp)

    if prefer_resid:
        c = _pick_first_existing_col(df, PREFERRED_RESID_COLS)
        if c is not None:
            return df[c].astype(float)

    for c in ["y_imputed", "y_clean", "y_raw"]:
        if c in df.columns:
            return df[c].astype(float)

    # last resort
    return df.iloc[:, 0].astype(float)


def load_df_full(parquet_dir, res, prefer_resid=True):
    files = sorted(Path(parquet_dir).glob(f"*_{res}.parquet"))
    if len(files) < 2:
        return None

    data_map = {}
    for fp in files:
        bld = fp.stem.split("_")[0]
        if bld not in BUILDINGS:
            continue
        data_map[bld] = load_series_from_parquet(fp, prefer_resid=prefer_resid)

    if not data_map:
        return None

    df = pd.DataFrame(data_map).sort_index()
    # enforce consistent column order
    cols = [b for b in BUILDINGS if b in df.columns]
    return df[cols]


def compute_ccf_curve(s1, s2, max_lag):
    """
    Cross correlation curve.
    Returns lags, ccf, and effective sample size used.
    """
    common = pd.concat([s1, s2], axis=1).dropna()
    if len(common) < max_lag * 2 + 10:
        return None, None, 0

    v1 = common.iloc[:, 0].to_numpy(dtype=float)
    v2 = common.iloc[:, 1].to_numpy(dtype=float)

    v1 = v1 - np.mean(v1)
    v2 = v2 - np.mean(v2)

    n = len(v1)
    denom = np.std(v1) * np.std(v2) * n
    if denom == 0:
        return None, None, n

    corr = signal.correlate(v1, v2, mode="full") / denom
    lags = np.arange(-n + 1, n)

    mask = (lags >= -max_lag) & (lags <= max_lag)
    return lags[mask], corr[mask], n


def peak_from_curve(lags, ccf):
    if lags is None or ccf is None or len(ccf) == 0:
        return None
    idx = int(np.argmax(np.abs(ccf)))
    return int(lags[idx]), float(ccf[idx]), float(np.abs(ccf[idx]))


def split_df(df, train_end, valid_end):
    train_end = pd.Timestamp(train_end)
    valid_end = pd.Timestamp(valid_end)

    splits = {
        "Train": df[df.index <= train_end],
        "Valid": df[(df.index > train_end) & (df.index <= valid_end)],
        "Test": df[df.index > valid_end],
    }
    return splits


def summarize_top_pairs_from_matrix(mat: pd.DataFrame, top_k=3):
    """
    Return top_k off diagonal pairs by absolute value.
    """
    if mat is None or mat.empty:
        return []

    m = mat.copy()
    for c in m.columns:
        if c in m.index:
            m.loc[c, c] = np.nan

    best = []
    for a, b in PAIRS:
        if a not in m.index or b not in m.columns:
            continue
        v = m.loc[a, b]
        if pd.isna(v):
            continue
        best.append((a, b, float(v), float(abs(v))))

    best.sort(key=lambda x: x[3], reverse=True)
    return best[:top_k]


def plot_lag_curves(parquet_dir, out_dir, train_end, valid_end, top_k_pairs=2, prefer_stl_resid=True):
    """
    Cleaner Lag curve visualization:
    Each resolution produces a figure with Train, Valid, Test panels.
    All pairs are drawn in light gray.
    Top K pairs for each split get strong color, thicker lines, and annotations.
    Confidence band is shown using approximate 95 percent threshold.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for res, conf in RES_CONF.items():
        max_lag = conf["lag_max"]

        df_full = load_df_full(parquet_dir, res, prefer_resid=prefer_stl_resid)
        if df_full is None or df_full.shape[1] < 2:
            continue

        # If we did not load residuals, build residuals as fallback
        used_resid = False
        if any(c in PREFERRED_RESID_COLS for c in getattr(df_full, "columns", [])):
            used_resid = True

        # Here df_full already is the chosen signal. If it is y_imputed, derive residuals for fair lag structure.
        # We treat "prefer_stl_resid=True" as signal is residual, else fallback residual.
        if not prefer_stl_resid:
            df_signal = get_residuals_fast(df_full, pd.Timestamp(train_end), res)
            signal_name = "Seasonal Median Residual (Train baseline)"
        else:
            # assume parquet provides residuals, otherwise it may have loaded y_imputed
            # detect this by checking presence of extreme periodic structure via column name is not available
            # so we provide a safer rule: if parquet lacks stl_resid for most buildings, build fallback residuals
            # this keeps plots aligned with earlier analysis intent
            has_stl = True
            files = sorted(Path(parquet_dir).glob(f"*_{res}.parquet"))
            ok = 0
            for fp in files:
                d = pd.read_parquet(fp, columns=[c for c in PREFERRED_RESID_COLS if c in pd.read_parquet(fp).columns] or None)
                if any(c in d.columns for c in PREFERRED_RESID_COLS):
                    ok += 1
            if ok >= max(2, len(files) // 2):
                df_signal = df_full
                signal_name = "STL Residual (from preprocess)"
            else:
                df_signal = get_residuals_fast(df_full, pd.Timestamp(train_end), res)
                signal_name = "Seasonal Median Residual (fallback)"

        splits = split_df(df_signal, train_end, valid_end)

        fig, axes = plt.subplots(1, 3, figsize=(21, 6.3), sharey=True)
        fig.suptitle(
            f"Cross Correlation Lag Curves | {res} | Signal: {signal_name}",
            fontsize=16,
            y=1.02
        )

        # collect per split stats and save a small summary
        summary_rows = []

        for idx, (split_name, split_df_) in enumerate(splits.items()):
            ax = axes[idx]
            ax.set_title(split_name, fontsize=14)
            ax.set_xlabel("Lag (steps)")
            if idx == 0:
                ax.set_ylabel("Cross correlation")

            ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)
            ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.18)

            # compute peaks for all pairs first
            pair_stats = []
            n_list = []

            for b1, b2 in PAIRS:
                if b1 not in split_df_.columns or b2 not in split_df_.columns:
                    continue

                lags, ccf, n_eff = compute_ccf_curve(split_df_[b1], split_df_[b2], max_lag)
                if lags is None:
                    continue

                p = peak_from_curve(lags, ccf)
                if p is None:
                    continue

                peak_lag, peak_corr, peak_abs = p
                label = f"{SHORT[b1]}-{SHORT[b2]}"
                pair_stats.append((label, lags, ccf, peak_lag, peak_corr, peak_abs, n_eff))
                n_list.append(n_eff)

            if not pair_stats:
                ax.text(0.5, 0.5, "Not enough overlap for lag curves", ha="center", va="center")
                ax.set_axis_off()
                continue

            # significance band using a robust N estimate
            n_eff_global = int(np.median(n_list)) if n_list else 0
            if n_eff_global > 0:
                thr = 1.96 / np.sqrt(n_eff_global)
                ax.axhline(thr, color="black", linewidth=1.0, alpha=0.22)
                ax.axhline(-thr, color="black", linewidth=1.0, alpha=0.22)
                ax.text(
                    0.98,
                    0.02,
                    f"approx 95% band: ±{thr:.3f}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=9,
                    alpha=0.75
                )

            # select top pairs for this split
            pair_stats.sort(key=lambda x: x[5], reverse=True)
            top = pair_stats[:max(1, int(top_k_pairs))]
            top_labels = {t[0] for t in top}

            # draw all in gray first
            for label, lags, ccf, peak_lag, peak_corr, peak_abs, n_eff in pair_stats:
                ax.plot(
                    lags,
                    ccf,
                    color="0.55",
                    linewidth=1.0,
                    alpha=0.18,
                    zorder=1
                )

            # draw top pairs
            for rank, (label, lags, ccf, peak_lag, peak_corr, peak_abs, n_eff) in enumerate(top, start=1):
                color = PAIR_COLORS.get(label, "C0")
                ax.plot(
                    lags,
                    ccf,
                    label=f"Top {rank}: {label}",
                    color=color,
                    linewidth=2.4,
                    alpha=0.95,
                    zorder=3
                )

                ax.scatter([peak_lag], [peak_corr], color=color, s=55, zorder=4)

                # annotation with small offset
                y_off = 0.04 if peak_corr >= 0 else -0.06
                ax.text(
                    peak_lag,
                    peak_corr + y_off,
                    f"lag {peak_lag}, r {peak_corr:.2f}",
                    fontsize=9.5,
                    color=color,
                    ha="center",
                    va="center",
                    weight="bold",
                    zorder=5
                )

                summary_rows.append(
                    {
                        "resolution": res,
                        "split": split_name,
                        "pair": label,
                        "peak_lag": peak_lag,
                        "peak_corr": peak_corr,
                        "abs_peak_corr": peak_abs,
                        "n_eff": n_eff,
                    }
                )

            # tidy axes
            ax.set_xlim(-max_lag, max_lag)
            ax.grid(True, axis="y", alpha=0.18)
            sns.despine(ax=ax)

            # concise legend
            if idx == 2:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=True,
                    title="Key pairs"
                )

        plt.tight_layout()
        png_path = out_dir / f"Viz_LagCurves_{res}.png"
        svg_path = out_dir / f"Viz_LagCurves_{res}.svg"
        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()

        if summary_rows:
            pd.DataFrame(summary_rows).sort_values(
                ["resolution", "split", "abs_peak_corr"],
                ascending=[True, True, False]
            ).to_csv(out_dir / f"KeyPairs_LagPeaks_{res}.csv", index=False)

        print(f"[Generated] {png_path.name}")
        print(f"[Generated] {svg_path.name}")
        if summary_rows:
            print(f"[Generated] KeyPairs_LagPeaks_{res}.csv")


def _read_matrix(csv_path: Path, enforce_order=True):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)

    # normalize names if user used full building names in csv
    # keep original if already matches
    if enforce_order:
        cols = [b for b in BUILDINGS if b in df.columns]
        rows = [b for b in BUILDINGS if b in df.index]
        if cols and rows:
            df = df.loc[rows, cols]
    return df


def plot_heatmaps(csv_dir, out_dir, top_k_pairs=3):
    """
    Heatmaps with better readability:
    Uses upper triangle only, consistent ordering, and adds a concise key findings footer.
    """
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolutions = ["5min", "30min", "1hour"]
    splits = ["train", "valid", "test"]

    metrics = [
        ("Resid_Pearson", "corr_pearson", "Residual Pearson correlation"),
        ("XCorr_Max", "xcorr_max", "Max cross correlation"),
    ]

    for filename_prefix, file_pattern, title_prefix in metrics:
        fig, axes = plt.subplots(3, 3, figsize=(16.6, 14.2))
        fig.suptitle(f"{title_prefix} across resolutions and splits", fontsize=18, y=0.98)

        key_findings = []

        for i, res in enumerate(resolutions):
            for j, split in enumerate(splits):
                ax = axes[i, j]
                csv_path = csv_dir / f"{res}_{file_pattern}_{split}.csv"
                df = _read_matrix(csv_path, enforce_order=True)

                if df is None or df.empty:
                    ax.text(0.5, 0.5, "Data not found", ha="center", va="center")
                    ax.set_axis_off()
                    continue

                # upper triangle mask, keep diagonal
                mask = np.tril(np.ones_like(df, dtype=bool), k=-1)

                sns.heatmap(
                    df,
                    ax=ax,
                    cmap="RdBu_r",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                    square=True,
                    linewidths=0.6,
                    linecolor="white",
                    cbar=False,
                    annot=True,
                    fmt=".2f",
                    annot_kws={"size": 10}
                )

                if i == 0:
                    ax.set_title(split.capitalize(), fontsize=14, pad=10)

                if j == 0:
                    ax.set_ylabel(res, fontsize=14, rotation=90)
                else:
                    ax.set_ylabel("")

                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35)
                ax.tick_params(axis="y", rotation=0)

                # key findings from this matrix
                top_pairs = summarize_top_pairs_from_matrix(df, top_k=top_k_pairs)
                if top_pairs:
                    top_str = ", ".join([f"{SHORT[a]}-{SHORT[b]} {v:+.2f}" for a, b, v, _ in top_pairs[:2]])
                    key_findings.append(f"{res} {split}: {top_str}")

        # one shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.52])
        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Coefficient")

        # footer with key findings
        if key_findings:
            footer = "Key pairs (top by |value|):  " + "   |   ".join(key_findings)
            fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10.2, alpha=0.9)

        plt.subplots_adjust(wspace=0.12, hspace=0.28, right=0.90, bottom=0.08)
        png_path = out_dir / f"Viz_{filename_prefix}_Matrix.png"
        svg_path = out_dir / f"Viz_{filename_prefix}_Matrix.svg"
        plt.savefig(png_path, bbox_inches="tight")
        plt.savefig(svg_path, bbox_inches="tight")
        plt.close()
        print(f"[Generated] {png_path.name}")
        print(f"[Generated] {svg_path.name}")


def plot_best_lags_heatmap(csv_dir, out_dir, top_k_pairs=3):
    """
    Best lag matrices:
    Upper triangle, diverging colormap centered at 0.
    Adds footer with key lead lag pairs using xcorr_max if present.
    """
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolutions = ["5min", "30min", "1hour"]
    splits = ["train", "valid", "test"]

    fig, axes = plt.subplots(3, 3, figsize=(16.6, 14.2))
    fig.suptitle("Optimal lag steps (lead lag relationship)", fontsize=18, y=0.98)

    key_findings = []

    for i, res in enumerate(resolutions):
        limit = RES_CONF[res]["lag_max"]

        for j, split in enumerate(splits):
            ax = axes[i, j]
            lag_path = csv_dir / f"{res}_xcorr_lag_{split}.csv"
            df_lag = _read_matrix(lag_path, enforce_order=True)

            if df_lag is None or df_lag.empty:
                ax.set_axis_off()
                continue

            mask = np.tril(np.ones_like(df_lag, dtype=bool), k=-1)

            sns.heatmap(
                df_lag,
                ax=ax,
                cmap="PuOr",
                center=0,
                vmin=-limit,
                vmax=limit,
                mask=mask,
                square=True,
                linewidths=0.6,
                linecolor="white",
                cbar=False,
                annot=True,
                fmt=".0f",
                annot_kws={"size": 10}
            )

            if i == 0:
                ax.set_title(split.capitalize(), fontsize=14, pad=10)
            if j == 0:
                ax.set_ylabel(res, fontsize=14, rotation=90)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=35)
            ax.tick_params(axis="y", rotation=0)

            # try to use xcorr_max to state the most meaningful lead lag relation
            xcorr_path = csv_dir / f"{res}_xcorr_max_{split}.csv"
            df_xcorr = _read_matrix(xcorr_path, enforce_order=True)

            if df_xcorr is not None and not df_xcorr.empty:
                top_pairs = summarize_top_pairs_from_matrix(df_xcorr, top_k=top_k_pairs)
                if top_pairs:
                    a, b, v, _ = top_pairs[0]
                    lag_ab = df_lag.loc[a, b] if (a in df_lag.index and b in df_lag.columns) else np.nan
                    key_findings.append(f"{res} {split}: {SHORT[a]} leads {SHORT[b]} at lag {int(lag_ab)} with r {v:+.2f}")

    # shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.52])
    sm = plt.cm.ScalarMappable(cmap="PuOr")
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Lag steps (positive: row leads column)")

    if key_findings:
        footer = "Key lead lag statements:  " + "   |   ".join(key_findings)
        fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=10.2, alpha=0.9)

    plt.subplots_adjust(wspace=0.12, hspace=0.28, right=0.90, bottom=0.08)
    png_path = out_dir / "Viz_BestLag_Matrix.png"
    svg_path = out_dir / "Viz_BestLag_Matrix.svg"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {png_path.name}")
    print(f"[Generated] {svg_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True, help="Directory containing .parquet files")
    ap.add_argument("--csv_dir", required=True, help="Directory containing .csv analysis results")
    ap.add_argument("--out_dir", required=True, help="Output directory for figures")
    ap.add_argument("--train_end", default="2017-12-31 23:59:59")
    ap.add_argument("--valid_end", default="2018-06-30 23:59:59")
    ap.add_argument("--top_k_pairs", type=int, default=2, help="Top K pairs to highlight in lag curves")
    ap.add_argument("--prefer_stl_resid", action="store_true", help="Prefer STL residual from preprocess if available")
    args = ap.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(">>> Generating lag curves with highlighted key pairs")
    plot_lag_curves(
        parquet_dir=args.parquet_dir,
        out_dir=args.out_dir,
        train_end=args.train_end,
        valid_end=args.valid_end,
        top_k_pairs=args.top_k_pairs,
        prefer_stl_resid=args.prefer_stl_resid
    )

    print(">>> Generating correlation matrices with key findings")
    plot_heatmaps(args.csv_dir, args.out_dir, top_k_pairs=3)

    print(">>> Generating best lag matrices with key lead lag statements")
    plot_best_lags_heatmap(args.csv_dir, args.out_dir, top_k_pairs=3)

    print("All visualizations complete.")


if __name__ == "__main__":
    main()
