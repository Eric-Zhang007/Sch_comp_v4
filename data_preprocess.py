import argparse
import re
import json
import warnings
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

RES_MAP = {
    "5_minutes": ("5min", "5min", 288),
    "30_minutes": ("30min", "30min", 48),
    "1_hour": ("1hour", "1h", 24),
}

BLD_SET = {"Commercial", "Office", "Public", "Residential"}
RAMP_Q = 0.999
STL_K = 8.0

def infer_meta_from_path(fp: Path):
    parts = fp.parts
    year = None
    res_key = None
    bld = None

    for p in parts:
        if re.fullmatch(r"\d{4}", p):
            year = p
        if p in RES_MAP:
            res_key = p

    leaf = fp.parent.name
    for cand in BLD_SET:
        if leaf.lower().endswith(cand.lower()):
            bld = cand
            break

    if year is None or res_key is None or bld is None:
        raise ValueError(f"Cannot infer meta from path: {fp}")

    res_tag, freq, slots_per_day = RES_MAP[res_key]
    return year, res_tag, freq, slots_per_day, bld

def read_day_file(fp: Path):
    df = pd.read_excel(fp)
    cols = {c.strip(): c for c in df.columns}
    time_col = cols.get("Time", None)
    val_col = cols.get("Power (kW)", None)
    if time_col is None or val_col is None:
        raise ValueError(f"Unexpected columns in {fp}: {list(df.columns)}")

    d = df[[time_col, val_col]].copy()
    d.columns = ["Time", "Power"]
    d["Time"] = pd.to_datetime(d["Time"])
    d["Power"] = pd.to_numeric(d["Power"], errors="coerce")
    d = d.dropna(subset=["Time"]).sort_values("Time")

    if len(d) == 0:
        return d

    day = d["Time"].min().floor("D")
    end = day + pd.Timedelta(days=1)
    d = d[d["Time"] < end]
    return d

def build_continuous_series(files, freq):
    parts = []
    dup_within = 0
    for fp in sorted(files):
        try:
            d = read_day_file(fp)
        except Exception:
            continue
        if len(d) == 0:
            continue
        dup_within += int(d["Time"].duplicated().sum())
        d = d.drop_duplicates(subset=["Time"], keep="last")
        parts.append(d)

    if len(parts) == 0:
        return None, {"n_files": len(files)}

    all_df = pd.concat(parts, ignore_index=True).sort_values("Time")
    dup_across = int(all_df["Time"].duplicated().sum())
    all_df = all_df.drop_duplicates(subset=["Time"], keep="last").set_index("Time").sort_index()

    full_idx = pd.date_range(all_df.index.min(), all_df.index.max(), freq=freq)
    aligned = all_df.reindex(full_idx)
    aligned.index.name = "Time"

    report = {
        "start": str(all_df.index.min()),
        "end": str(all_df.index.max()),
        "n_files": len(files),
        "dup_within_file": dup_within,
        "dup_across_files": dup_across,
    }
    return aligned, report

def add_time_features(df):
    dt = df.index

    min_of_day = dt.hour * 60 + dt.minute
    df["day_sin"] = np.sin(2 * np.pi * min_of_day / 1440)
    df["day_cos"] = np.cos(2 * np.pi * min_of_day / 1440)

    week_pos = dt.dayofweek + min_of_day / 1440.0
    df["week_sin"] = np.sin(2 * np.pi * week_pos / 7)
    df["week_cos"] = np.cos(2 * np.pi * week_pos / 7)

    days_in_year = 365 + dt.is_leap_year
    day_of_year = dt.dayofyear

    df["year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df["year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    return df

def robust_scale(train_values, all_values):
    med = np.nanmedian(train_values)
    iqr = np.nanpercentile(train_values, 75) - np.nanpercentile(train_values, 25)
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0
    return (all_values - med) / iqr, {"median": float(med), "iqr": float(iqr)}

def process_and_enrich(aligned: pd.DataFrame, freq: str, train_end: str, meta_info: dict):
    y = aligned["Power"].copy()
    hard = (~np.isfinite(y)) | (y < 0)
    missing = y.isna()
    y_clean = y.mask(hard)

    step = {"5min": 2, "30min": 1, "1h": 1}[freq]
    y_short_imp = y_clean.interpolate(limit=step, limit_direction="both")

    train_end_dt = pd.to_datetime(train_end)
    train_mask = aligned.index <= train_end_dt

    if freq == "5min":
        period = 288
    elif freq == "30min":
        period = 48
    else:
        period = 24

    y_filled_linear = y_short_imp.interpolate(method="linear", limit_direction="both").fillna(method="bfill").fillna(method="ffill")
    
    resid_full = None
    try:
        stl = STL(y_filled_linear, period=period, robust=True)
        res = stl.fit()
        trend = res.trend
        seasonal = res.seasonal
        resid_full = res.resid

        valid_train_resid = resid_full[train_mask & (~missing)]
        if len(valid_train_resid) > 10:
            mu_noise = valid_train_resid.mean()
            std_noise = valid_train_resid.std()
        else:
            mu_noise = 0.0
            std_noise = 0.0

        noise = np.random.normal(mu_noise, std_noise, size=len(aligned))
        y_stl_generated = trend + seasonal + noise
        y_stl_generated = np.maximum(y_stl_generated, 0.0)
        
        y_imputed = y_short_imp.copy()
        mask_still_missing = y_imputed.isna()
        y_imputed[mask_still_missing] = y_stl_generated[mask_still_missing]
        
    except Exception:
        y_imputed = y_filled_linear
        resid_full = pd.Series(0, index=aligned.index) if resid_full is None else resid_full

    df = pd.DataFrame({
        "y_raw": y,
        "y_clean": y_clean,
        "y_imputed": y_imputed,
        "mask_missing": missing.astype(int),
        "mask_hard": hard.astype(int),
    }, index=aligned.index)

    df = add_time_features(df)

    ramp = df["y_clean"].diff().abs()
    ramp_train = ramp[train_mask].dropna()
    if len(ramp_train) > 10:
        ramp_thr = np.nanquantile(ramp_train, RAMP_Q)
    else:
        ramp_thr = np.nanmax(ramp_train) if len(ramp_train) else np.inf

    mask_ramp = ramp > ramp_thr
    df["mask_ramp"] = mask_ramp.astype(int)

    if resid_full is not None:
        df["stl_resid"] = resid_full
        resid_train = resid_full[train_mask & (~missing)].dropna()
        if len(resid_train) > 0:
            med_resid = np.nanmedian(resid_train)
            mad_resid = np.nanmedian(np.abs(resid_train - med_resid))
        else:
            med_resid = 0.0
            mad_resid = 0.0

        if not np.isfinite(mad_resid) or mad_resid == 0:
            mad_resid = 1.0

        mask_stl = np.abs(resid_full - med_resid) > STL_K * 1.4826 * mad_resid
        mask_stl = mask_stl.fillna(False)
    else:
        mask_stl = pd.Series(False, index=df.index)
        df["stl_resid"] = np.nan

    df["mask_stl"] = mask_stl.astype(int)

    y_imp_vals = df["y_imputed"].values
    y_imp_train = df.loc[train_mask, "y_imputed"].values

    scaled, scaler_param = robust_scale(y_imp_train, y_imp_vals)
    df["y_scaled"] = scaled

    quality_metrics = {
        "missing_points": int(missing.sum()),
        "neg_points": int((y < 0).sum()),
        "zero_points": int((y == 0).sum()),
    }

    scaler_meta = {
        "file": f"{meta_info['building']}_{meta_info['resolution']}.parquet",
        "building": meta_info['building'],
        "resolution": meta_info['resolution'],
        "train_end": train_end,
        "ramp_thr": float(ramp_thr),
        **scaler_param
    }

    return df, quality_metrics, scaler_meta

def process_single_bucket(bld, res_tag, freq, files, out_dir, train_end):
    aligned, file_rep = build_continuous_series(files, freq=freq)
    if aligned is None:
        return None

    meta_info = {"building": bld, "resolution": res_tag}
    df_final, quality_metrics, scaler_meta = process_and_enrich(aligned, freq, train_end, meta_info)

    out_path = Path(out_dir)
    save_path = out_path / f"{bld}_{res_tag}.parquet"
    df_final.to_parquet(save_path)

    full_rep = {**file_rep, **quality_metrics, "building": bld, "resolution": res_tag, "freq": freq}

    return full_rep, scaler_meta

def main(root_dir: str, out_dir: str, train_end: str = "2017-12-31 23:59:59"):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "quality_report.csv"
    param_path = out / "scaler_params.json"

    buckets = {}
    for fp in root.rglob("*.xlsx"):
        if fp.name.startswith("~$"):
            continue
        try:
            year, res_tag, freq, slots, bld = infer_meta_from_path(fp)
            key = (bld, res_tag, freq)
            buckets.setdefault(key, []).append(fp)
        except ValueError:
            continue

    tasks = []
    for (bld, res_tag, freq), files in sorted(buckets.items()):
        tasks.append((bld, res_tag, freq, files))

    all_reports = []
    all_params = []

    max_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_bucket, bld, res, freq, files, out, train_end): (bld, res)
            for (bld, res, freq, files) in tasks
        }

        for future in as_completed(futures):
            bld, res = futures[future]
            try:
                result = future.result()
                if result is None:
                    continue
                rep, param = result
                all_reports.append(rep)
                all_params.append(param)
                print(f"[Done] {bld} {res}")
            except Exception as e:
                print(f"[Error] {bld} {res}: {e}")

    report_cols = [
        "building", "resolution", "freq", "start", "end", "n_files",
        "dup_within_file", "dup_across_files", "missing_points",
        "neg_points", "zero_points"
    ]

    if all_reports:
        pd.DataFrame(all_reports, columns=report_cols).sort_values(["building", "resolution"]).to_csv(report_path, index=False)

    if all_params:
        all_params.sort(key=lambda x: x["file"])
        with open(param_path, "w") as f:
            json.dump(all_params, f, indent=2)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_end", default="2017-12-31 23:59:59")
    args = ap.parse_args()
    main(args.root, args.out, args.train_end)