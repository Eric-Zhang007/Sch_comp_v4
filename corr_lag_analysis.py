import argparse
import multiprocessing
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Optional

warnings.filterwarnings("ignore")

BUILDINGS = ["Commercial", "Office", "Public", "Residential"]
RES_CONF = {
    "5min": {"slots": 288, "lag_max": 288, "granger_max": 24},
    "30min": {"slots": 48, "lag_max": 48, "granger_max": 6},
    "1hour": {"slots": 24, "lag_max": 24, "granger_max": 4},
}

def _compute_stl_residual(series: pd.Series, period: int) -> pd.Series:
    s = series.astype(float)
    s_nonan = s.dropna()
    if len(s_nonan) < max(period * 2 + 10, 50):
        return pd.Series(np.nan, index=s.index)

    try:
        from statsmodels.tsa.seasonal import STL
        resid = STL(s_nonan, period=period, robust=True).fit().resid
        return resid.reindex(s.index)
    except Exception:
        return pd.Series(np.nan, index=s.index)

def _pick_residual_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["stl_resid", "y_stl_resid", "stl_residual", "resid_stl", "resid"]:
        if cand in df.columns:
            return cand
    return None

def build_residual_frame(files, res: str) -> pd.DataFrame:
    period = RES_CONF[res]["slots"]
    data_map = {}

    for fp in files:
        bld = fp.stem.split("_")[0]
        df = pd.read_parquet(fp)

        col = _pick_residual_column(df)
        if col is not None:
            data_map[bld] = df[col]
            continue

        base_col = "y_clean" if "y_clean" in df.columns else ("y_imputed" if "y_imputed" in df.columns else None)
        if base_col is None:
            data_map[bld] = pd.Series(np.nan, index=df.index)
            continue

        data_map[bld] = _compute_stl_residual(df[base_col], period=period)

    return pd.DataFrame(data_map)

def calculate_best_lag(s1, s2, max_lag):
    common = pd.concat([s1, s2], axis=1).dropna()
    if len(common) < max_lag * 2 + 10:
        return np.nan, 0

    common = common.rank()

    v1 = common.iloc[:, 0].values
    v2 = common.iloc[:, 1].values

    import scipy.signal

    n = len(v1)
    corr = scipy.signal.correlate(v1 - v1.mean(), v2 - v2.mean(), mode="full")
    denom = np.std(v1) * np.std(v2) * n

    if denom == 0:
        return 0.0, 0

    corr_norm = corr / denom
    lags = np.arange(-n + 1, n)

    mask = (lags >= -max_lag) & (lags <= max_lag)
    valid_lags = lags[mask]
    valid_corr = corr_norm[mask]

    idx_max = np.argmax(np.abs(valid_corr))
    best_corr = valid_corr[idx_max]
    best_lag = valid_lags[idx_max]

    return float(best_corr), int(best_lag)

def run_granger_pair(s_target, s_source, max_lag):
    df_pair = pd.concat([s_target, s_source], axis=1).dropna()
    if len(df_pair) < 500:
        return None

    dt = df_pair.index.to_series().diff().dt.total_seconds()
    mode_dt = dt.mode()
    if len(mode_dt) == 0:
        return None
    step = mode_dt.iloc[0]

    breaks = dt != step
    groups = breaks.cumsum()
    counts = groups.value_counts()
    if len(counts) == 0:
        return None

    best_grp = counts.idxmax()
    subset = df_pair[groups == best_grp]
    if len(subset) < 100:
        return None

    data = subset.iloc[:, :2].values

    try:
        res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        chosen_lag = 1
        chosen_p = 1.0
        chosen_f = 0.0

        for lg, val in res.items():
            f_stat = val[0]["params_ftest"][0]
            p_val = val[0]["params_ftest"][1]
            if f_stat > chosen_f:
                chosen_f = f_stat
                chosen_p = p_val
                chosen_lag = lg

        return {"lag": chosen_lag, "f_stat": chosen_f, "p_value": chosen_p}
    except Exception:
        return None

def process_resolution(res, parquet_dir, out_dir, train_end, valid_end):
    files = sorted(Path(parquet_dir).glob(f"*_{res}.parquet"))
    if len(files) < 2:
        return

    df_resid = build_residual_frame(files, res)

    train_mask = df_resid.index <= train_end
    valid_mask = (df_resid.index > train_end) & (df_resid.index <= valid_end)
    test_mask = df_resid.index > valid_end

    splits = {
        "train": df_resid[train_mask],
        "valid": df_resid[valid_mask],
        "test": df_resid[test_mask],
    }

    blds = df_resid.columns.tolist()
    n_blds = len(blds)
    lag_max = RES_CONF[res]["lag_max"]

    for name, part_df in splits.items():
        if part_df.empty:
            continue

        corr_s = part_df.corr(method="spearman")
        corr_s.to_csv(out_dir / f"{res}_corr_spearman_{name}.csv")

        corr_p = part_df.corr(method="pearson")
        corr_p.to_csv(out_dir / f"{res}_corr_pearson_{name}.csv")

        mat_val = pd.DataFrame(np.zeros((n_blds, n_blds)), index=blds, columns=blds)
        mat_lag = pd.DataFrame(np.zeros((n_blds, n_blds)), index=blds, columns=blds)

        for i in range(n_blds):
            for j in range(n_blds):
                if i == j:
                    mat_val.iloc[i, j] = 1.0
                    mat_lag.iloc[i, j] = 0
                    continue

                b1, b2 = blds[i], blds[j]
                val, lag = calculate_best_lag(part_df[b1], part_df[b2], lag_max)
                mat_val.iloc[i, j] = val
                mat_lag.iloc[i, j] = lag

        mat_val.to_csv(out_dir / f"{res}_xcorr_max_{name}.csv")
        mat_lag.to_csv(out_dir / f"{res}_xcorr_lag_{name}.csv")

    granger_res = []
    df_train = splits["train"]
    granger_limit = RES_CONF[res]["granger_max"]

    for i in range(n_blds):
        for j in range(n_blds):
            if i == j:
                continue

            source = blds[i]
            target = blds[j]

            ret = run_granger_pair(df_train[target], df_train[source], granger_limit)
            if ret:
                granger_res.append(
                    {
                        "source": source,
                        "target": target,
                        "best_lag": ret["lag"],
                        "f_stat": ret["f_stat"],
                        "p_value": ret["p_value"],
                    }
                )

    if granger_res:
        pd.DataFrame(granger_res).to_csv(out_dir / f"{res}_granger_train.csv", index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_end", default="2017-12-31 23:59:59")
    ap.add_argument("--valid_end", default="2018-06-30 23:59:59")
    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_end = pd.Timestamp(args.train_end)
    valid_end = pd.Timestamp(args.valid_end)

    resolutions = ["5min", "30min", "1hour"]

    max_workers = min(len(resolutions), multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for res in resolutions:
            futures.append(pool.submit(process_resolution, res, parquet_dir, out_dir, train_end, valid_end))

        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()