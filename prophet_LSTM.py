import argparse
import os
import random
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import logging
import warnings
from typing import Tuple, Dict, Any
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def safe_log1p(s: pd.Series) -> pd.Series:
    return np.log1p(pd.to_numeric(s, errors="coerce").clip(lower=0.0))

def safe_expm1(a: np.ndarray) -> np.ndarray:
    return np.expm1(a)


warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _get_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter, "torch"
    except Exception:
        pass
    try:
        from tensorboardX import SummaryWriter  # type: ignore
        return SummaryWriter, "tensorboardX"
    except Exception:
        return None, "none"


def print_device_info():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"Current CUDA Device: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("Current CUDA Device: <unknown>")

    has_xpu = hasattr(torch, "xpu")
    xpu_avail = has_xpu and torch.xpu.is_available()
    print(f"XPU Available: {xpu_avail}")
    if xpu_avail:
        try:
            print(f"Current XPU Device: {torch.xpu.get_device_name(0)}")
        except Exception:
            print("Current XPU Device: <unknown>")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed_all(seed)
        except Exception:
            pass
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> Tuple[torch.device, str]:
    req = str(requested).lower().strip()
    has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()

    if req == "cpu":
        return torch.device("cpu"), "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if req == "xpu":
        if has_xpu:
            return torch.device("xpu"), "xpu"
        raise RuntimeError("Requested --device xpu, but XPU is not available.")

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if has_xpu:
        return torch.device("xpu"), "xpu"
    return torch.device("cpu"), "cpu"


def configure_threads(cpu_threads: int):
    n = max(1, int(cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))
    os.environ.setdefault("STAN_NUM_THREADS", str(n))
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(2)
    except Exception:
        pass


def file_fingerprint(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        return {"path": os.path.abspath(path), "size": None, "mtime": None}


def check_fingerprint(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    for k in ("size", "mtime"):
        if expected.get(k) is None or actual.get(k) is None:
            continue
        if expected.get(k) != actual.get(k):
            return False
    return True


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    smape = float(100.0 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / denom))
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": rmse, "sMAPE": smape}


class ResidualLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        return self.fc(out)


def infer_steps_per_day(index: pd.DatetimeIndex):
    if len(index) < 10:
        raise ValueError("Index too short to infer frequency.")
    unique_index = index.unique().sort_values()
    if len(unique_index) < 2:
        raise ValueError("Not enough unique timestamps to infer frequency.")
    diffs = unique_index.to_series().diff().dropna()
    step = diffs.median()
    if not pd.notna(step) or step <= pd.Timedelta(0):
        raise ValueError("Failed to infer step size from index.")
    steps_per_day = int(round(pd.Timedelta(days=1) / step))
    steps_per_hour = max(1, int(round(pd.Timedelta(hours=1) / step)))
    return step, steps_per_hour, steps_per_day


def add_features(df: pd.DataFrame, steps_per_hour: int, steps_per_day: int, resid_col: str = "resid"):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    dt = df.index
    timestamp_s = dt.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    week = 7 * day
    year = 365.2425 * day

    df["day_sin"] = np.sin(timestamp_s * (2 * np.pi / day))
    df["day_cos"] = np.cos(timestamp_s * (2 * np.pi / day))
    df["week_sin"] = np.sin(timestamp_s * (2 * np.pi / week))
    df["week_cos"] = np.cos(timestamp_s * (2 * np.pi / week))
    df["year_sin"] = np.sin(timestamp_s * (2 * np.pi / year))
    df["year_cos"] = np.cos(timestamp_s * (2 * np.pi / year))

    lag_24h = steps_per_day
    roll_6h = max(1, 6 * steps_per_hour)
    df["lag_24h"] = df[resid_col].shift(lag_24h)
    df["rolling_mean_6h"] = df[resid_col].rolling(window=roll_6h, min_periods=1).mean()
    return df


def get_data_splits(df: pd.DataFrame):
    idx = df.index
    train_mask = idx < "2018-01-01"
    val_mask = (idx >= "2018-01-01") & (idx < "2018-07-01")

    test_mask_jul = (idx >= "2018-07-01") & (idx < "2018-08-01")
    test_mask_aug = (idx >= "2018-08-01") & (idx <= "2018-08-10 23:59:59")
    test_mask_nov = (idx >= "2018-11-20") & (idx < "2018-12-01")
    test_mask_dec = (idx >= "2018-12-01") & (idx <= "2018-12-31 23:59:59")

    test_mask = test_mask_jul | test_mask_aug | test_mask_nov | test_mask_dec
    return (
        np.asarray(train_mask, dtype=bool),
        np.asarray(val_mask, dtype=bool),
        np.asarray(test_mask, dtype=bool),
    )


class ImputationDataset(Dataset):
    def __init__(self, data, split_mask, obs_present, seq_len, mode="train", mask_prob=0.5):
        self.seq_len = int(seq_len)
        self.mode = mode
        self.mask_prob = float(mask_prob)

        split_mask = np.asarray(split_mask, dtype=int)
        obs_present = np.asarray(obs_present, dtype=int)

        split_s = pd.Series(split_mask)
        valid_windows = split_s.rolling(window=self.seq_len + 1).min() == 1
        candidate = np.where(valid_windows.values)[0]
        candidate = candidate[obs_present[candidate] == 1]

        self.valid_indices = candidate
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        target_idx = int(self.valid_indices[idx])
        start_idx = target_idx - self.seq_len

        window = self.data[start_idx:target_idx].clone()
        target = self.data[target_idx, -1].clone()

        mask_indicator = torch.zeros((self.seq_len, 1))

        if self.mode == "train" and random.random() < self.mask_prob:
            max_gap = max(1, self.seq_len // 2)
            gap_len = random.randint(1, max_gap)
            gap_start = random.randint(0, self.seq_len - gap_len)
            window[gap_start:gap_start + gap_len, -1] = 0.0
            mask_indicator[gap_start:gap_start + gap_len, 0] = 1.0

        x_input = torch.cat([window, mask_indicator], dim=1)
        return x_input, target, target_idx


def build_prophet_model(cps, sps, hps, seasonality_mode):
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=float(cps),
        seasonality_prior_scale=float(sps),
        holidays_prior_scale=float(hps),
        n_changepoints=100,
        changepoint_range=0.95,
        seasonality_mode=str(seasonality_mode),
        uncertainty_samples=0,
    )
    m.add_seasonality(name="daily", period=1, fourier_order=20)
    m.add_seasonality(name="weekly", period=7, fourier_order=15)
    m.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    m.add_seasonality(name="hourly", period=1 / 24, fourier_order=8)
    m.add_seasonality(name="half_hourly", period=1 / 48, fourier_order=4)
    try:
        m.add_country_holidays(country_name="CN")
    except Exception:
        pass
    return m


def _eval_prophet_combo(cps, sps, hps, seasonality_mode, df_train, df_val):
    try:
        m = build_prophet_model(cps, sps, hps, seasonality_mode)
        m.fit(df_train)
        future = pd.DataFrame({"ds": df_val["ds"]})
        fc = m.predict(future)
        val_y = df_val.set_index("ds")["y"]
        pred_y = fc.set_index("ds")["yhat"]
        common = val_y.index.intersection(pred_y.index)
        if len(common) == 0:
            return None
        rmse = float(np.sqrt(mean_squared_error(val_y.loc[common], pred_y.loc[common])))
        return rmse, (cps, sps, hps)
    except Exception:
        return None


def tune_prophet(df_train, df_val, seasonality_mode="multiplicative", prophet_jobs: int = 12):
    cps_list = [0.05, 0.1, 0.3, 0.5, 1.0]
    sps_list = [10, 20, 50, 100]
    hps_list = [10, 20]
    combos = list(product(cps_list, sps_list, hps_list))
    n_jobs = max(1, int(prophet_jobs))

    results = joblib.Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        joblib.delayed(_eval_prophet_combo)(cps, sps, hps, seasonality_mode, df_train, df_val)
        for cps, sps, hps in combos
    )

    best = None
    best_rmse = float("inf")
    for r in results:
        if r is None:
            continue
        rmse, params = r
        if rmse < best_rmse:
            best_rmse = rmse
            best = params
    if best is None:
        best = (0.3, 50, 10)

    print(f">>> Best Prophet params: cps={best[0]}, sps={best[1]}, hps={best[2]}  val_RMSE={best_rmse:.4f}")
    return best


def _autocast_ctx(device: torch.device, amp: bool, amp_dtype: str):
    if not amp:
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = amp_dtype.lower().strip()
    if device.type == "cuda":
        chosen = torch.float16 if dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=chosen, enabled=True)
    if device.type == "xpu":
        chosen = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
        return torch.autocast(device_type="xpu", dtype=chosen, enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)


def phase1_paths(out_dir: str) -> Dict[str, str]:
    return {
        "meta": os.path.join(out_dir, "phase1_meta.json"),
        "df": os.path.join(out_dir, "phase1_df.parquet"),
        "prophet_model": os.path.join(out_dir, "prophet_model.json"),
    }


def phase2_paths(out_dir: str) -> Dict[str, str]:
    return {
        "meta": os.path.join(out_dir, "phase2_meta.json"),
        "df": os.path.join(out_dir, "phase2_df.parquet"),
        "scaler_feat": os.path.join(out_dir, "scaler_feat.pkl"),
        "scaler_target": os.path.join(out_dir, "scaler_target.pkl"),
    }


def phase3_paths(out_dir: str) -> Dict[str, str]:
    return {
        "best_model": os.path.join(out_dir, "best_model.pth"),
        "ckpt_last": os.path.join(out_dir, "checkpoint_last.pth"),
        "training_log": os.path.join(out_dir, "training_log.csv"),
    }


# 2) 修改 _load_and_prepare_raw_df：增加 log 列
def _load_and_prepare_raw_df(data_path: str, target_col: str) -> pd.DataFrame:
    df_all = pd.read_parquet(data_path).sort_index()
    if not isinstance(df_all.index, pd.DatetimeIndex):
        df_all.index = pd.to_datetime(df_all.index)

    if target_col not in df_all.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df_all.columns.tolist()}")

    if "mask_missing" in df_all.columns:
        df_all["mask_missing"] = pd.to_numeric(df_all["mask_missing"], errors="coerce").fillna(0).astype(int)
    else:
        df_all["mask_missing"] = 0

    if "mask_hard" in df_all.columns:
        df_all["mask_hard"] = pd.to_numeric(df_all["mask_hard"], errors="coerce").fillna(0).astype(int)
    else:
        df_all["mask_hard"] = 0

    df_all["y_imp"] = pd.to_numeric(df_all[target_col], errors="coerce").clip(lower=0.0)

    if "y_clean" in df_all.columns:
        df_all["y_obs"] = pd.to_numeric(df_all["y_clean"], errors="coerce")
    elif "y_raw" in df_all.columns:
        df_all["y_obs"] = pd.to_numeric(df_all["y_raw"], errors="coerce")
    else:
        df_all["y_obs"] = df_all["y_imp"]

    df_all.loc[df_all["y_obs"].notna(), "y_obs"] = df_all.loc[df_all["y_obs"].notna(), "y_obs"].clip(lower=0.0)

    if (df_all["mask_missing"] == 0).all():
        df_all["mask_missing"] = df_all["y_obs"].isna().astype(int)

    # 新增：log1p 版本（y_obs 缺失会保留 NaN）
    df_all["y_imp_log"] = safe_log1p(df_all["y_imp"])
    df_all["y_obs_log"] = safe_log1p(df_all["y_obs"])

    return df_all


def run_phase1_or_load(args) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = phase1_paths(args.out_dir)
    fp_now = file_fingerprint(args.data_path)
    want_resume = args.resume_from in ("phase2", "phase3", "phase4")

    if want_resume and os.path.exists(p["meta"]) and os.path.exists(p["df"]):
        meta = load_json(p["meta"])
        fp_old = meta.get("data_fingerprint", {})
        if not args.ignore_data_mismatch and not check_fingerprint(fp_old, fp_now):
            raise RuntimeError("Data file fingerprint changed since last run. Use --ignore_data_mismatch to force resume.")
        df_all = pd.read_parquet(p["df"]).sort_index()
        if not isinstance(df_all.index, pd.DatetimeIndex):
            df_all.index = pd.to_datetime(df_all.index)
        print("Phase 1 cache found: loaded phase1_df.parquet, skipping Prophet fitting.")
        return df_all, meta

    if want_resume and os.path.exists(p["prophet_model"]):
        print("Phase 1 cache missing, prophet_model.json exists. Rebuilding phase1 cache from saved Prophet model...")
        df_all = _load_and_prepare_raw_df(args.data_path, args.target_col)
        step, steps_per_hour, steps_per_day = infer_steps_per_day(df_all.index)

        with open(p["prophet_model"], "r") as fin:
            m = model_from_json(fin.read())

        future_full = pd.DataFrame({"ds": df_all.index})
        fc_full = m.predict(future_full)

        df_all["prophet_yhat"] = fc_full["yhat"].values
        df_all["prophet_trend"] = fc_full["trend"].values
        df_all["baseline"] = df_all["prophet_yhat"]
        df_all["resid"] = df_all["y_imp"] - df_all["baseline"]

        meta = {
            "phase": "phase1",
            "data_fingerprint": fp_now,
            "step": str(step),
            "steps_per_hour": int(steps_per_hour),
            "steps_per_day": int(steps_per_day),
            "prophet_loaded_from_json": True,
            "prophet_seasonality_mode": str(args.prophet_seasonality_mode),
        }

        df_all.to_parquet(p["df"])
        save_json(meta, p["meta"])
        print("Phase 1 cache rebuilt: wrote phase1_df.parquet and phase1_meta.json.")
        return df_all, meta

    print(f"Loading data from {args.data_path}...")
    df_all = _load_and_prepare_raw_df(args.data_path, args.target_col)
    step, steps_per_hour, steps_per_day = infer_steps_per_day(df_all.index)
    print(f"Inferred step={step}, steps_per_hour={steps_per_hour}, steps_per_day={steps_per_day}")

    print(">>> Phase 1: Prophet baseline extraction (log space)...")
    # 关键：拟合用 y_obs_log
    df_fit_1h = df_all["y_obs_log"].to_frame("y").resample(args.prophet_resample).mean()
    df_fit_1h["ds"] = df_fit_1h.index
    df_fit_1h = df_fit_1h.dropna(subset=["y"]).copy()

    # 轻微 epsilon，避免 y 全 0 时的数值尴尬
    df_fit_1h["y"] = df_fit_1h["y"].clip(lower=0.0) + 1e-9

    train_mask_p = df_fit_1h["ds"] < "2018-01-01"
    val_mask_p = (df_fit_1h["ds"] >= "2018-01-01") & (df_fit_1h["ds"] < "2018-07-01")

    df_p_train = df_fit_1h.loc[train_mask_p, ["ds", "y"]].copy()
    df_p_val = df_fit_1h.loc[val_mask_p, ["ds", "y"]].copy()

    if len(df_p_train) < 200:
        raise ValueError("Prophet training data too small after dropping missing. Check y_clean and masks.")

    best_cps, best_sps, best_hps = tune_prophet(
        df_p_train,
        df_p_val,
        seasonality_mode=args.prophet_seasonality_mode,
        prophet_jobs=args.prophet_jobs,
    )

    m = build_prophet_model(best_cps, best_sps, best_hps, args.prophet_seasonality_mode)
    m.fit(df_p_train)

    with open(p["prophet_model"], "w") as fout:
        fout.write(model_to_json(m))

    future_full = pd.DataFrame({"ds": df_all.index})
    fc_full = m.predict(future_full)

    # 关键：baseline 在 log 空间
    df_all["baseline_log"] = fc_full["yhat"].values
    df_all["prophet_trend_log"] = fc_full["trend"].values

    # resid 在 log 空间
    df_all["resid_log"] = df_all["y_imp_log"] - df_all["baseline_log"]

    # 可选：同时给原始尺度 baseline，方便直观检查
    df_all["baseline"] = np.clip(safe_expm1(df_all["baseline_log"].values), 0.0, None)

    meta = {
        "phase": "phase1",
        "data_fingerprint": fp_now,
        "step": str(step),
        "steps_per_hour": int(steps_per_hour),
        "steps_per_day": int(steps_per_day),
        "prophet_best": {"cps": float(best_cps), "sps": float(best_sps), "hps": float(best_hps)},
        "prophet_seasonality_mode": str(args.prophet_seasonality_mode),
        "prophet_loaded_from_json": False,
        "log_space": True,
    }

    df_all.to_parquet(p["df"])
    save_json(meta, p["meta"])
    return df_all, meta


def run_phase2_or_load(args, df_all: pd.DataFrame, meta1: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p2 = phase2_paths(args.out_dir)
    fp_now = file_fingerprint(args.data_path)

    want_resume = args.resume_from in ("phase3", "phase4")
    if want_resume and os.path.exists(p2["meta"]) and os.path.exists(p2["df"]):
        meta = load_json(p2["meta"])
        fp_old = meta.get("data_fingerprint", {})
        if not args.ignore_data_mismatch and not check_fingerprint(fp_old, fp_now):
            raise RuntimeError("Data file fingerprint changed since last run. Use --ignore_data_mismatch to force resume.")
        df = pd.read_parquet(p2["df"]).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        print("Phase 2 cache found: loaded phase2_df.parquet, skipping feature engineering.")
        return df, meta

    steps_per_hour = int(meta1["steps_per_hour"])
    steps_per_day = int(meta1["steps_per_day"])

    print(">>> Phase 2: LSTM feature engineering (log space residuals)...")
    df = df_all[
        ["y_imp", "y_obs", "y_imp_log", "y_obs_log", "mask_missing", "mask_hard", "baseline", "baseline_log", "resid_log"]
    ].copy()

    df = add_features(df, steps_per_hour=steps_per_hour, steps_per_day=steps_per_day, resid_col="resid_log")
    df = df.dropna().copy()

    feat_cols = [
        "day_sin", "day_cos", "week_sin", "week_cos", "year_sin", "year_cos",
        "lag_24h", "rolling_mean_6h",
        "mask_missing",
    ]

    meta = {
        "phase": "phase2",
        "data_fingerprint": fp_now,
        "steps_per_hour": steps_per_hour,
        "steps_per_day": steps_per_day,
        "feat_cols": feat_cols,
        "log_space": True,
        "target_residual_col": "resid_log",
    }

    df.to_parquet(p2["df"])
    save_json(meta, p2["meta"])
    return df, meta


def build_loaders(args, device: torch.device, train_dataset: Dataset, val_dataset: Dataset):
    use_pin = bool(args.pin_memory) and device.type == "cuda"

    persistent_ok = (args.num_workers > 0) and (os.name != "nt")
    dl_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent_ok,
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    return train_loader, val_loader


def warmup_one_batch(device: torch.device, model: nn.Module, criterion, train_loader):
    it = iter(train_loader)
    bx, by, _ = next(it)
    bx, by = bx.to(device), by.to(device)
    with torch.no_grad():
        out = model(bx).view(-1)
        _ = criterion(out, by)


def train_phase3(args, device: torch.device, writer, df: pd.DataFrame, meta2: Dict[str, Any]):
    p2 = phase2_paths(args.out_dir)
    p3 = phase3_paths(args.out_dir)
    feat_cols = meta2["feat_cols"]
    target_col = meta2.get("target_residual_col", "resid_log")

    obs_present_aligned = ((df["mask_missing"] == 0) & (df["mask_hard"] == 0) & (df["y_obs"].notna())).values.astype(bool)
    train_mask, val_mask, test_mask = get_data_splits(df)

    scaler_feat = StandardScaler()
    scaler_target = StandardScaler()
    scaler_feat.fit(df.loc[train_mask, feat_cols])

    target_fit_mask = train_mask & obs_present_aligned
    if int(target_fit_mask.sum()) < 200:
        target_fit_mask = train_mask
    scaler_target.fit(df.loc[target_fit_mask, [target_col]])

    joblib.dump(scaler_feat, p2["scaler_feat"])
    joblib.dump(scaler_target, p2["scaler_target"])

    data_feat = scaler_feat.transform(df[feat_cols])
    data_target = scaler_target.transform(df[[target_col]])
    combined_data = np.hstack([data_feat, data_target])
    model_input_dim = combined_data.shape[1] + 1

    train_dataset = ImputationDataset(combined_data, train_mask, obs_present_aligned, args.seq_len, mode="train", mask_prob=args.mask_prob)
    val_dataset = ImputationDataset(combined_data, val_mask, obs_present_aligned, args.seq_len, mode="val", mask_prob=0.0)
    test_dataset = ImputationDataset(combined_data, test_mask, obs_present_aligned, args.seq_len, mode="test", mask_prob=0.0)

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. Check split dates, seq_len, or observed mask.")

    print(f">>> Phase 3: Training LSTM (input_dim={model_input_dim})...")
    print(f"Train windows: {len(train_dataset)}  Val windows: {len(val_dataset)}  Test windows: {len(test_dataset)}")
    print(f"DataLoader: num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} pin_memory={args.pin_memory}")

    model = ResidualLSTM(
        input_dim=model_input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss()

    use_grad_scaler = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    start_epoch = 0
    best_val = float("inf")
    early_stop_cnt = 0
    patience = 8
    train_log = []

    if args.resume_training and os.path.exists(p3["ckpt_last"]):
        ckpt = torch.load(p3["ckpt_last"], map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if use_grad_scaler and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_val = float(ckpt.get("best_val", best_val))
            early_stop_cnt = int(ckpt.get("early_stop_cnt", 0))
            if os.path.exists(p3["training_log"]):
                try:
                    train_log = pd.read_csv(p3["training_log"]).to_dict(orient="records")
                except Exception:
                    train_log = []
            print(f"Resumed training: start_epoch={start_epoch} best_val={best_val:.6f}")
        except Exception as e:
            print(f"Failed to load checkpoint, starting fresh. Reason: {e}")

    train_loader, val_loader = build_loaders(args, device, train_dataset, val_dataset)

    if len(train_loader) == 0:
        raise RuntimeError("train_loader is empty. batch_size may be too large or dataset windows too small.")
    if len(val_loader) == 0:
        print("Warning: val_loader is empty. Validation will be skipped.")

    print("Warmup: fetching one batch and running one forward pass...")
    warmup_one_batch(device, model, criterion, train_loader)
    print("Warmup done. Training starts.")

    total_epochs = args.epochs
    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc="Epochs", dynamic_ncols=True)

    global_step = start_epoch * len(train_loader)

    for epoch in epoch_pbar:
        model.train()
        running = []

        batch_iter = train_loader
        if args.batch_progress:
            batch_iter = tqdm(train_loader, desc=f"Train e{epoch+1}", leave=False, dynamic_ncols=True)

        for bi, (bx, by, _) in enumerate(batch_iter):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx(device, args.amp, args.amp_dtype):
                out = model(bx).view(-1)
                loss = criterion(out, by)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            lv = float(loss.item())
            running.append(lv)

            if writer is not None:
                writer.add_scalar("loss/train_step", lv, global_step)
            global_step += 1

            if args.batch_progress and bi % max(1, args.log_every) == 0:
                try:
                    batch_iter.set_postfix({"loss": f"{lv:.4f}"})
                except Exception:
                    pass

        avg_train = float(np.mean(running)) if len(running) > 0 else float("nan")
        if writer is not None:
            writer.add_scalar("loss/train_epoch", avg_train, int(epoch + 1))

        val_mae = None
        val_rmse = None
        if len(val_loader) > 0:
            model.eval()
            vp, vt = [], []
            with torch.no_grad():
                for bx, by, _ in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    with _autocast_ctx(device, args.amp, args.amp_dtype):
                        out = model(bx).view(-1)
                    vp.append(out.float().cpu().numpy())
                    vt.append(by.float().cpu().numpy())
            vp = np.concatenate(vp)
            vt = np.concatenate(vt)
            val_mae = float(mean_absolute_error(vt, vp))
            metrics = calculate_metrics(vt, vp)
            val_rmse = float(metrics["RMSE"])
            scheduler.step(val_mae)

            if writer is not None:
                writer.add_scalar("metrics/val_mae", val_mae, int(epoch + 1))
                writer.add_scalar("metrics/val_rmse", val_rmse, int(epoch + 1))
                writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), int(epoch + 1))

            if val_mae < best_val:
                best_val = val_mae
                early_stop_cnt = 0
                torch.save(model.state_dict(), p3["best_model"])
            else:
                early_stop_cnt += 1
        else:
            scheduler.step(avg_train)

        epoch_pbar.set_postfix({
            "train": f"{avg_train:.4f}",
            "val_mae": f"{val_mae:.4f}" if val_mae is not None else "NA",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        train_log.append({
            "epoch": int(epoch + 1),
            "train_loss": avg_train,
            "val_mae": float(val_mae) if val_mae is not None else np.nan,
            "val_rmse": float(val_rmse) if val_rmse is not None else np.nan,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        pd.DataFrame(train_log).to_csv(p3["training_log"], index=False)

        ckpt = {
            "epoch": int(epoch + 1),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_grad_scaler else None,
            "best_val": float(best_val),
            "early_stop_cnt": int(early_stop_cnt),
            "args": vars(args),
        }
        torch.save(ckpt, p3["ckpt_last"])

        if writer is not None:
            writer.flush()

        if early_stop_cnt >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, test_dataset, scaler_target, df

def plot_predictions_df(df: pd.DataFrame, out_dir: str, prefix: str = "step1"):
    os.makedirs(out_dir, exist_ok=True)

    if "ds" not in df.columns:
        raise ValueError("plot requires a 'ds' column in predictions dataframe.")
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds")

    y_true_col = "y_obs"
    y_pred_col = "y_pred"
    mask_col = "eval_mask"

    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError("plot requires columns y_obs and y_pred in predictions dataframe.")

    if mask_col in df.columns:
        mask = df[mask_col].astype(int).astype(bool).values
    else:
        mask = df[y_true_col].notna().values

    dfp = df.loc[mask].copy()
    if len(dfp) == 0:
        raise ValueError("No valid points to plot after applying eval_mask or y_obs.notna().")

    t = dfp["ds"].values
    y_true = dfp[y_true_col].astype(float).values
    y_pred = dfp[y_pred_col].astype(float).values

    baseline_col = "baseline" if "baseline" in dfp.columns else None

    # 1) 时间序列对比，横轴用时间轴
    plt.figure(figsize=(14, 4))
    plt.plot(t, y_true, label="True", linewidth=1.2)
    plt.plot(t, y_pred, label="Pred", linewidth=1.2)
    if baseline_col is not None:
        plt.plot(t, dfp[baseline_col].astype(float).values, label="Baseline", linewidth=1.0, alpha=0.8)
    plt.title(f"{prefix} Time Series")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_timeseries.png"), dpi=150)
    plt.close()

    # 2) 残差时间序列
    resid = y_pred - y_true
    plt.figure(figsize=(14, 3))
    plt.plot(t, resid, linewidth=1.0)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.title(f"{prefix} Residuals (Pred - True)")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_residual_ts.png"), dpi=150)
    plt.close()

    # 3) 残差分布
    plt.figure(figsize=(5, 4))
    plt.hist(resid, bins=50, alpha=0.8)
    plt.title(f"{prefix} Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_residual_hist.png"), dpi=150)
    plt.close()

    # 4) True vs Pred 散点
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.5)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{prefix} True vs Pred")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_scatter.png"), dpi=150)
    plt.close()

# 7) 修改 phase4：pred_resid 变成 pred_resid_log，最后 expm1 回原始尺度
def run_phase4(args, device: torch.device, writer, model: nn.Module, test_dataset: Dataset, scaler_target: StandardScaler, df_feat: pd.DataFrame):
    p3 = phase3_paths(args.out_dir)
    best_model_path = p3["best_model"]
    print(">>> Phase 4: Final prediction and evaluation (log space)...")

    if not os.path.exists(best_model_path):
        print("No saved best model found. Exiting.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_preds, all_indices = [], []

    with torch.no_grad():
        for bx, _, idx in test_loader:
            bx = bx.to(device)
            with _autocast_ctx(device, args.amp, args.amp_dtype):
                out = model(bx).view(-1)
            all_preds.append(out.float().cpu().numpy())
            all_indices.append(idx.numpy())

    if len(all_preds) == 0:
        print("No predictions generated.")
        return

    all_preds = np.concatenate(all_preds)
    all_indices = np.concatenate(all_indices)

    pred_resid_log = scaler_target.inverse_transform(all_preds.reshape(-1, 1)).flatten()

    final_df = df_feat.iloc[all_indices].copy()
    final_df["resid_pred_log"] = pred_resid_log

    # log 空间合成
    if "baseline_log" not in final_df.columns:
        raise ValueError("baseline_log not found. Phase1/2 must generate baseline_log.")
    final_df["y_pred_log"] = final_df["baseline_log"] + final_df["resid_pred_log"]

    # 回到原始尺度
    final_df["y_pred"] = np.clip(safe_expm1(final_df["y_pred_log"].values), 0.0, None)

    final_df = final_df.sort_index()
    eval_mask = (final_df["mask_missing"] == 0) & (final_df["mask_hard"] == 0) & final_df["y_obs"].notna()

    y_true = final_df.loc[eval_mask, "y_obs"].values
    y_pred = final_df.loc[eval_mask, "y_pred"].values

    if len(y_true) == 0:
        print("No observed points in test slice for evaluation.")
        final_metrics = {}
    else:
        final_metrics = calculate_metrics(y_true, y_pred)

    final_df = final_df.copy()
    final_df["ds"] = final_df.index
    final_df["eval_mask"] = eval_mask.astype(int)

    out_df = final_df.reset_index(drop=True)

    keep_cols = [
        "ds",
        "y_imp", "y_obs",
        "y_imp_log", "y_obs_log",
        "baseline", "baseline_log",
        "resid_log",
        "y_pred", "y_pred_log",
        "resid_pred_log",
        "mask_missing", "mask_hard", "eval_mask",
    ]

    out_csv = os.path.join(args.out_dir, "predictions.csv")
    out_df[keep_cols].to_csv(out_csv, index=False)

    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)

    print("Test metrics on observed points:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    if getattr(args, "plot_results", False):
        plot_dir = getattr(args, "plot_dir", None) or os.path.join(args.out_dir, "plots")
        plot_predictions_df(out_df[keep_cols], plot_dir, prefix="pred")
        print(f"Plots saved to: {plot_dir}")

def run_pipeline(args):
    configure_threads(args.cpu_threads)
    setup_seed(args.seed)

    device, device_tag = select_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    SummaryWriter, tb_src = _get_summary_writer()
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        tb_dir = args.tb_logdir or os.path.join(args.out_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard logging enabled: {tb_dir} ({tb_src})")
        writer.add_text("run/status", f"started {datetime.now().isoformat()} device={device_tag}", 0)
        writer.flush()
    elif not args.no_tensorboard:
        print("TensorBoard disabled: install tensorboard or tensorboardX to enable it.")

    print(f"Using device: {device_tag}")

    try:
        df_all, meta1 = run_phase1_or_load(args)
        df_feat, meta2 = run_phase2_or_load(args, df_all, meta1)
        model, test_dataset, scaler_target, df_feat_for_eval = train_phase3(args, device, writer, df_feat, meta2)
        run_phase4(args, device, writer, model, test_dataset, scaler_target, df_feat_for_eval)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()


if __name__ == "__main__":
    print_device_info()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="y_imputed")
    parser.add_argument("--prophet_seasonality_mode", type=str, default="additive", choices=["additive", "multiplicative"])
    parser.add_argument("--mask_prob", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=48)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"])

    parser.add_argument("--cpu_threads", type=int, default=max(1, (os.cpu_count() or 14) - 2))
    parser.add_argument("--prophet_jobs", type=int, default=max(1, (os.cpu_count() or 14) - 2))

    default_workers = 0 if os.name == "nt" else min(12, max(2, (os.cpu_count() or 14) - 2))
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--batch_progress", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)

    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--tb_logdir", type=str, default=None)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--resume_from", type=str, default="phase1", choices=["phase1", "phase2", "phase3", "phase4"])
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ignore_data_mismatch", action="store_true")
    parser.add_argument("--plot_results", action="store_true")
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--prophet_resample", type=str, default="30min")



    args = parser.parse_args()
    run_pipeline(args)
