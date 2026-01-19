import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# TensorBoard (optional)
# -----------------------------
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


# -----------------------------
# Repro + device + threads
# -----------------------------
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed_all(seed)  # type: ignore
        except Exception:
            pass
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
    try:
        torch.set_num_threads(n)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(2)
    except Exception:
        pass


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


# -----------------------------
# Fingerprint for data safety
# -----------------------------
def file_fingerprint(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        return {"path": os.path.abspath(path), "size": None, "mtime": None}


def dir_fingerprint(root_dir: str) -> Dict[str, Any]:
    root = Path(root_dir)
    items = []
    for p in sorted(root.glob("*.parquet")):
        items.append(file_fingerprint(str(p)))
    sp = root / "scaler_params.json"
    items.append(file_fingerprint(str(sp)))
    return {"root": str(root.resolve()), "files": items}


def _check_fingerprint(expected: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    exp_files = expected.get("files", [])
    act_files = actual.get("files", [])
    if len(exp_files) != len(act_files):
        return False
    for e, a in zip(exp_files, act_files):
        if e.get("path") != a.get("path"):
            return False
        if e.get("size") is not None and a.get("size") is not None and e.get("size") != a.get("size"):
            return False
        if e.get("mtime") is not None and a.get("mtime") is not None and e.get("mtime") != a.get("mtime"):
            return False
    return True


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=NumpyEncoder)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Time features (same semantics as your earlier pipeline)
# -----------------------------
def get_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    min_of_day = dt_index.hour * 60 + dt_index.minute
    day_sin = np.sin(2 * np.pi * min_of_day / 1440)
    day_cos = np.cos(2 * np.pi * min_of_day / 1440)

    week_pos = dt_index.dayofweek + min_of_day / 1440.0
    week_sin = np.sin(2 * np.pi * week_pos / 7)
    week_cos = np.cos(2 * np.pi * week_pos / 7)

    days_in_year = 365 + dt_index.is_leap_year
    day_of_year = dt_index.dayofyear
    year_sin = np.sin(2 * np.pi * day_of_year / days_in_year)
    year_cos = np.cos(2 * np.pi * day_of_year / days_in_year)

    return np.stack([day_sin, day_cos, week_sin, week_cos, year_sin, year_cos], axis=1).astype(np.float32)


# -----------------------------
# Data loading with strict 5min grid reindex
# Expected per-node parquet columns:
#   y_scaled, mask_missing, mask_hard, optional mask_stl
# Scaler file: scaler_params.json list with fields: file, building, resolution, median, iqr
# -----------------------------
def load_data_and_params(root_dir: str, target_resolution: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    root = Path(root_dir)
    files = sorted(list(root.glob("*.parquet")))
    param_path = root / "scaler_params.json"

    if not files or not param_path.exists():
        raise FileNotFoundError(f"Data not found in {root_dir}")

    with open(param_path, "r") as f:
        params_list = json.load(f)
    params_map = {p["file"]: p for p in params_list}

    dfs_scaled = []
    dfs_mask_input = []
    dfs_mask_target = []
    node_names = []
    medians, iqrs = [], []

    for fp in files:
        if fp.name not in params_map:
            continue
        p = params_map[fp.name]
        if p.get("resolution") != target_resolution:
            continue

        node_key = f"{p.get('building')}_{p.get('resolution')}"
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        if "y_scaled" not in df.columns:
            raise ValueError(f"{fp.name} missing column y_scaled")
        if "mask_missing" not in df.columns or "mask_hard" not in df.columns:
            raise ValueError(f"{fp.name} missing mask_missing/mask_hard")

        dfs_scaled.append(df[["y_scaled"]].rename(columns={"y_scaled": node_key}))

        m_miss = pd.to_numeric(df["mask_missing"], errors="coerce").fillna(0).astype(int)
        m_hard = pd.to_numeric(df["mask_hard"], errors="coerce").fillna(0).astype(int)
        if "mask_stl" in df.columns:
            m_stl = pd.to_numeric(df["mask_stl"], errors="coerce").fillna(0).astype(int)
        else:
            m_stl = pd.Series(0, index=df.index, dtype=int)

        # 输入侧无效点: missing 或 hard
        invalid_input = (m_miss | m_hard).astype(int)
        # 监督侧无效点: missing 或 hard 或 stl
        invalid_target = (m_miss | m_hard | m_stl).astype(int)

        dfs_mask_input.append(invalid_input.rename(node_key))
        dfs_mask_target.append(invalid_target.rename(node_key))

        node_names.append(node_key)
        medians.append(float(p["median"]))
        iqrs.append(float(p["iqr"]))

    if not dfs_scaled:
        raise ValueError(f"No files found for resolution={target_resolution}")

    merged_scaled = pd.concat(dfs_scaled, axis=1).sort_index()
    merged_in = pd.concat(dfs_mask_input, axis=1).sort_index()
    merged_tgt = pd.concat(dfs_mask_target, axis=1).sort_index()

    # Outer join already done by concat, now fill
    merged_scaled = merged_scaled.fillna(0.0)
    merged_in = merged_in.fillna(1).astype(np.float32)
    merged_tgt = merged_tgt.fillna(1).astype(np.float32)

    # Full regular grid
    freq_map = {"5min": "5min", "30min": "30min", "1hour": "1h"}
    if target_resolution not in freq_map:
        raise ValueError(f"Unknown resolution: {target_resolution}")
    full_idx = pd.date_range(merged_scaled.index.min(), merged_scaled.index.max(), freq=freq_map[target_resolution])

    merged_scaled = merged_scaled.reindex(full_idx).fillna(0.0).astype(np.float32)
    merged_in = merged_in.reindex(full_idx).fillna(1.0).astype(np.float32)
    merged_tgt = merged_tgt.reindex(full_idx).fillna(1.0).astype(np.float32)

    # Column order alignment
    merged_scaled = merged_scaled.reindex(columns=node_names)
    merged_in = merged_in.reindex(columns=node_names)
    merged_tgt = merged_tgt.reindex(columns=node_names)

    stats = {
        "nodes": node_names,
        "medians": np.array(medians, dtype=np.float32),
        "iqrs": np.array(iqrs, dtype=np.float32),
        "resolution": target_resolution,
        "full_index_start": str(full_idx.min()),
        "full_index_end": str(full_idx.max()),
        "n_steps": int(len(full_idx)),
    }

    return merged_scaled, merged_in, merged_tgt, stats


# -----------------------------
# Split protocol you specified
# Train: 2016-2017
# Val: 2018-01-01 to 2018-06-30
# Test: 2018-07, 2018-08-01..10, 2018-11-20..30, 2018-12
# -----------------------------
def build_split_masks(index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = index

    train_mask = idx < pd.Timestamp("2018-01-01")
    val_mask = (idx >= pd.Timestamp("2018-01-01")) & (idx < pd.Timestamp("2018-07-01"))

    test_mask_jul = (idx >= pd.Timestamp("2018-07-01")) & (idx < pd.Timestamp("2018-08-01"))
    test_mask_aug = (idx >= pd.Timestamp("2018-08-01")) & (idx <= pd.Timestamp("2018-08-10 23:59:59"))
    test_mask_nov = (idx >= pd.Timestamp("2018-11-20")) & (idx < pd.Timestamp("2018-12-01"))
    test_mask_dec = (idx >= pd.Timestamp("2018-12-01")) & (idx <= pd.Timestamp("2018-12-31 23:59:59"))

    test_mask = test_mask_jul | test_mask_aug | test_mask_nov | test_mask_dec

    return train_mask.astype(bool), val_mask.astype(bool), test_mask.astype(bool)


# -----------------------------
# Index generation
# strict: [i-lookback, i+horizon) all inside split
# deploy: only [i, i+horizon) inside split, history only requires physical availability
# -----------------------------
def get_indices_strict(split_mask: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    split = split_mask.astype(np.int32)
    n = len(split)
    w = lookback + horizon
    if w <= 0 or n <= w:
        return np.array([], dtype=np.int64)
    prefix = np.zeros(n + 1, dtype=np.int32)
    prefix[1:] = np.cumsum(split)
    sums = prefix[w:] - prefix[:-w]
    ok = np.where(sums == w)[0]
    return (ok + lookback).astype(np.int64)


def get_indices_deploy(split_mask: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    mask_int = split_mask.astype(np.int32)
    n = len(mask_int)
    if horizon <= 0 or n <= horizon:
        return np.array([], dtype=np.int64)

    prefix = np.zeros(n + 1, dtype=np.int32)
    prefix[1:] = np.cumsum(mask_int)

    all_i = np.arange(lookback, n - horizon + 1, dtype=np.int64)
    sums = prefix[all_i + horizon] - prefix[all_i]
    valid = (sums == horizon)
    return all_i[valid]


def filter_indices_with_targets(indices: np.ndarray, mask_tgt: np.ndarray, horizon: int) -> np.ndarray:
    if len(indices) == 0:
        return indices
    # mask_tgt: (T, N) 1 invalid
    valid_per_t = (1.0 - mask_tgt).sum(axis=1)  # (T,)
    prefix = np.zeros(len(valid_per_t) + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(valid_per_t)
    counts = prefix[indices + horizon] - prefix[indices]
    return indices[counts > 0]


# -----------------------------
# Correlation prior, computed only on training timestamps, using valid input points
# -----------------------------
def compute_corr_prior(values: np.ndarray, mask_in: np.ndarray, train_mask: np.ndarray, min_points: int = 200) -> np.ndarray:
    """
    values: (T, N) scaled
    mask_in: (T, N) 1 invalid input
    train_mask: (T,) bool
    Return prior_logits: (N, N), nonnegative, diag=1
    """
    T, N = values.shape
    prior = np.zeros((N, N), dtype=np.float32)

    valid_global = (train_mask[:, None]) & (mask_in == 0)

    for i in range(N):
        prior[i, i] = 1.0

    for i in range(N):
        for j in range(i + 1, N):
            valid_ij = valid_global[:, i] & valid_global[:, j]
            if valid_ij.sum() < min_points:
                r = 0.0
            else:
                xi = values[valid_ij, i]
                xj = values[valid_ij, j]
                si = float(np.std(xi))
                sj = float(np.std(xj))
                if si < 1e-8 or sj < 1e-8:
                    r = 0.0
                else:
                    r = float(np.corrcoef(xi, xj)[0, 1])
                    if not np.isfinite(r):
                        r = 0.0
            r = max(r, 0.0)
            prior[i, j] = r
            prior[j, i] = r

    return prior

def compute_tail_quantiles_per_node(
    values: np.ndarray,         # (T,N) scaled
    mask_tgt: np.ndarray,       # (T,N) invalid=1
    train_mask: np.ndarray,     # (T,)
    q_low: float,
    q_high: float,
    min_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    T, N = values.shape
    ql = np.zeros(N, dtype=np.float32)
    qh = np.zeros(N, dtype=np.float32)

    for n in range(N):
        valid = train_mask & (mask_tgt[:, n] == 0)
        data = values[valid, n]
        if data.size < min_points:
            # fallback: 只按 train_mask 取，避免极端情况下崩掉
            data = values[train_mask, n]
        ql[n] = float(np.quantile(data, q_low))
        qh[n] = float(np.quantile(data, q_high))
    return ql, qh

# -----------------------------
# Dataset: lazy window extraction, optional return of start index for exporting timestamp
# -----------------------------
class WindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        mask_in: np.ndarray,
        mask_tgt: np.ndarray,
        time_feats: np.ndarray,
        indices: np.ndarray,
        lookback: int,
        horizon: int,
        return_index: bool = False,
    ):
        self.values = values
        self.mask_in = mask_in
        self.mask_tgt = mask_tgt
        self.time_feats = time_feats
        self.indices = indices.astype(np.int64)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.n_nodes = int(values.shape[1])
        self.return_index = bool(return_index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        w_start = i - self.lookback
        w_end = i
        pred_end = i + self.horizon

        v_win = self.values[w_start:w_end]          # (L, N)
        m_in_win = self.mask_in[w_start:w_end]      # (L, N), 1 invalid input
        t_win = self.time_feats[w_start:w_end]      # (L, 6)

        v_win_clean = v_win * (1.0 - m_in_win)

        x = np.concatenate(
            [
                v_win_clean[..., None],  # (L, N, 1)
                m_in_win[..., None],     # (L, N, 1)
                np.repeat(t_win[:, None, :], self.n_nodes, axis=1),  # (L, N, 6)
            ],
            axis=2,
        ).astype(np.float32)  # (L, N, 8)

        y = self.values[w_end:pred_end].T.astype(np.float32)  # (N, H)
        m_t = self.mask_tgt[w_end:pred_end].T.astype(np.float32)  # (N, H), 1 invalid
        y_mask = (1.0 - m_t).astype(np.float32)  # (N, H), 1 valid

        if self.return_index:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask), torch.tensor(i, dtype=torch.long)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask)


# -----------------------------
# Loss
# -----------------------------
class MaskedCombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        mse_part = (diff ** 2) * mask
        mae_part = diff.abs() * mask
        denom = mask.sum().clamp(min=self.eps)
        return self.alpha * (mse_part.sum() / denom) + (1.0 - self.alpha) * (mae_part.sum() / denom)

class QuantileWeightedMaskedCombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        q_low: np.ndarray,     # (N,)
        q_high: np.ndarray,    # (N,)
        w_low: float = 4.0,
        w_high: float = 4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.w_low = float(w_low)
        self.w_high = float(w_high)
        self.eps = float(eps)

        ql = torch.tensor(q_low, dtype=torch.float32).view(1, -1, 1)   # (1,N,1)
        qh = torch.tensor(q_high, dtype=torch.float32).view(1, -1, 1)  # (1,N,1)
        self.register_buffer("q_low", ql)
        self.register_buffer("q_high", qh)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred, target, mask: (B, N, H), mask=1 means valid supervision
        ql = self.q_low.to(target.device)
        qh = self.q_high.to(target.device)

        w = torch.ones_like(target)
        w = torch.where(target < ql, w * (1.0 + self.w_low), w)
        w = torch.where(target > qh, w * (1.0 + self.w_high), w)

        eff = mask * w
        denom = eff.sum().clamp(min=self.eps)

        diff = pred - target
        mse_part = (diff ** 2) * eff
        mae_part = diff.abs() * eff

        return self.alpha * (mse_part.sum() / denom) + (1.0 - self.alpha) * (mae_part.sum() / denom)

# -----------------------------
# Model with correlation prior initialization
# -----------------------------
class TGCNGraphLearner(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_feats: int,
        hidden_dim: int,
        horizon: int,
        dropout: float = 0.2,
        top_k: Optional[int] = None,
        prior_logits: Optional[torch.Tensor] = None,
        prior_beta: float = 0.0,
        prior_learnable: bool = False,
        self_loop_w: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.top_k = top_k if top_k is None else int(top_k)
        self.self_loop_w = float(self_loop_w)

        self.node_emb1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.node_emb2 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)

        self.gcn_weights = nn.Linear(in_feats, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, horizon)
        self.dropout = nn.Dropout(dropout)

        if prior_logits is None:
            prior_logits = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        self.register_buffer("prior_logits", prior_logits.float())

        if prior_learnable:
            self.prior_beta = nn.Parameter(torch.tensor(float(prior_beta), dtype=torch.float32), requires_grad=True)
        else:
            self.register_buffer("prior_beta_const", torch.tensor(float(prior_beta), dtype=torch.float32))
            self.prior_beta = None

    def _beta(self) -> torch.Tensor:
        if self.prior_beta is not None:
            return self.prior_beta
        return self.prior_beta_const

    def compute_adj(self) -> torch.Tensor:
        with torch.autocast(device_type="cpu", enabled=False):
            emb1 = self.node_emb1.float()
            emb2 = self.node_emb2.float()
            learned = torch.mm(emb1, emb2.t())  # (N,N)

            beta = self._beta().float()
            logits = learned + beta * self.prior_logits.to(learned.device)
            logits = F.relu(logits)

            # numerical guard
            logits = logits + 1e-6 * torch.eye(self.num_nodes, device=logits.device, dtype=logits.dtype)

            if self.top_k is not None:
                k = min(max(int(self.top_k), 1), self.num_nodes)
                _, idx = torch.topk(logits, k=k, dim=1)
                keep = torch.zeros_like(logits, dtype=torch.bool)
                keep.scatter_(1, idx, True)
                logits = logits.masked_fill(~keep, -1e9)

            adj = F.softmax(logits, dim=1)
            if self.self_loop_w > 0:
                adj = adj + self.self_loop_w * torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
                adj = adj / adj.sum(dim=1, keepdim=True)
            return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F)
        B, T, N, n_feat = x.shape

        with torch.cuda.amp.autocast(enabled=False):
            adj = self.compute_adj()

        x_reshaped = x.view(B * T, N, n_feat)
        gcn_in = self.gcn_weights(x_reshaped)  # (B*T, N, H)

        adj_expand = adj.unsqueeze(0)  # (1,N,N)
        gcn_out = torch.matmul(adj_expand, gcn_in.view(B * T, N, self.hidden_dim))  # (B*T,N,H)
        gcn_out = gcn_out.view(B, T, N, self.hidden_dim)
        gcn_out = F.leaky_relu(gcn_out)

        gcn_flat = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        lstm_out, _ = self.lstm(gcn_flat)
        last_step = lstm_out[:, -1, :]

        out = self.fc(self.dropout(last_step))
        return out.view(B, N, self.horizon)


# -----------------------------
# Online metric trackers
# -----------------------------
class MetricTracker:
    def __init__(self, horizon: int):
        self.horizon = int(horizon)
        self.reset()

    def reset(self):
        self.total_mse = np.zeros(self.horizon, dtype=np.float64)
        self.total_mae = np.zeros(self.horizon, dtype=np.float64)
        self.total_count = np.zeros(self.horizon, dtype=np.float64)
        self.overall_mse_sum = 0.0
        self.overall_mae_sum = 0.0
        self.overall_count = 0.0

    def update(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor):
        if torch.is_tensor(pred):
            pred = pred.detach().float().cpu()
        if torch.is_tensor(true):
            true = true.detach().float().cpu()
        if torch.is_tensor(mask):
            mask = mask.detach().float().cpu()

        diff = pred - true
        mse = (diff ** 2) * mask
        mae = diff.abs() * mask

        mse_sum = mse.sum(dim=(0, 1)).numpy()
        mae_sum = mae.sum(dim=(0, 1)).numpy()
        cnt_sum = mask.sum(dim=(0, 1)).numpy()

        self.total_mse += mse_sum
        self.total_mae += mae_sum
        self.total_count += cnt_sum

        self.overall_mse_sum += float(mse.sum().item())
        self.overall_mae_sum += float(mae.sum().item())
        self.overall_count += float(mask.sum().item())

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.overall_count > 0:
            mse_all = self.overall_mse_sum / self.overall_count
            metrics["Overall_RMSE"] = float(np.sqrt(mse_all))
            metrics["Overall_MAE"] = float(self.overall_mae_sum / self.overall_count)

        for h in range(self.horizon):
            denom = self.total_count[h]
            if denom > 0:
                mse_h = self.total_mse[h] / denom
                mae_h = self.total_mae[h] / denom
                metrics[f"Step_{h+1}_RMSE"] = float(np.sqrt(mse_h))
                metrics[f"Step_{h+1}_MAE"] = float(mae_h)
        return metrics


# -----------------------------
# Train pipeline
# -----------------------------
@dataclass
class Paths:
    run_meta: str
    best_model: str
    ckpt_last: str
    training_log: str
    metrics: str
    node_metrics: str
    results_step1: str


def make_paths(out_dir: str) -> Paths:
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    return Paths(
        run_meta=str(od / "run_meta.json"),
        best_model=str(od / "best_model.pth"),
        ckpt_last=str(od / "checkpoint_last.pth"),
        training_log=str(od / "training_log.csv"),
        metrics=str(od / "metrics.json"),
        node_metrics=str(od / "node_metrics.json"),
        results_step1=str(od / "results_step1.csv"),
    )


def log_adj_stats(adj: torch.Tensor) -> Dict[str, float]:
    # adj: (N,N) row-stochastic
    a = adj.detach().float().cpu().numpy()
    eps = 1e-12
    row_entropy = -(a * np.log(a + eps)).sum(axis=1)  # (N,)
    row_max = a.max(axis=1)
    diag = np.diag(a)
    return {
        "adj_entropy_mean": float(row_entropy.mean()),
        "adj_max_mean": float(row_max.mean()),
        "adj_diag_mean": float(diag.mean()),
    }


def save_training_log_csv(rows: list, path: str):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def train_model(args):
    configure_threads(args.cpu_threads)
    setup_seed(args.seed)
    device, device_tag = select_device(args.device)

    paths = make_paths(args.out)
    fp_now = dir_fingerprint(args.root)

    # TensorBoard
    SummaryWriter, tb_src = _get_summary_writer()
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        tb_dir = args.tb_logdir or os.path.join(args.out, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard enabled: {tb_dir} ({tb_src})")

    # Load data
    df_scaled, df_in, df_tgt, stats = load_data_and_params(args.root, args.resolution)
    index = df_scaled.index
    values = df_scaled.values.astype(np.float32)
    mask_in = df_in.values.astype(np.float32)
    mask_tgt = df_tgt.values.astype(np.float32)
    time_feats = get_time_features(index)

    # Split masks
    train_mask, val_mask, test_mask = build_split_masks(index)

    # Correlation prior from training segment only
    prior_np = compute_corr_prior(values, mask_in, train_mask, min_points=args.prior_min_points)
    prior_t = torch.from_numpy(prior_np).float()

    
    # Index generation
    if args.split_mode == "deploy":
        idx_train = get_indices_strict(train_mask, args.lookback, args.horizon)
        idx_val = get_indices_deploy(val_mask, args.lookback, args.horizon)
        idx_test = get_indices_deploy(test_mask, args.lookback, args.horizon)
    else:
        idx_train = get_indices_strict(train_mask, args.lookback, args.horizon)
        idx_val = get_indices_strict(val_mask, args.lookback, args.horizon)
        idx_test = get_indices_strict(test_mask, args.lookback, args.horizon)

    # Filter samples with no valid supervision in target horizon
    idx_train = filter_indices_with_targets(idx_train, mask_tgt, args.horizon)
    idx_val = filter_indices_with_targets(idx_val, mask_tgt, args.horizon)
    idx_test = filter_indices_with_targets(idx_test, mask_tgt, args.horizon)

    # Stride
    idx_train = idx_train[:: args.stride]
    idx_val = idx_val[:: args.stride]
    idx_test = idx_test[:: args.stride]

    print(f"Split mode: {args.split_mode}")
    print(f"Windows - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        raise ValueError("One of the splits has zero valid windows. Check lookback/horizon/split dates.")

    # Datasets
    train_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_train, args.lookback, args.horizon, return_index=False)
    val_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_val, args.lookback, args.horizon, return_index=False)
    test_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_test, args.lookback, args.horizon, return_index=True)

    # DataLoader
    use_pin = device.type in ("cuda", "xpu")
    persistent_ok = (args.num_workers > 0) and (os.name != "nt")
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent_ok,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    # Model
    model = TGCNGraphLearner(
        num_nodes=len(stats["nodes"]),
        in_feats=8,
        hidden_dim=args.hidden,
        horizon=args.horizon,
        dropout=args.dropout,
        top_k=args.top_k,
        prior_logits=prior_t,
        prior_beta=args.prior_beta,
        prior_learnable=args.prior_learnable,
        self_loop_w=args.self_loop_w,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile failed, continuing without it: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    q_low_arr, q_high_arr = compute_tail_quantiles_per_node(
    values=values,
        mask_tgt=mask_tgt,
        train_mask=train_mask,
        q_low=args.tail_q_low,
        q_high=args.tail_q_high,
        min_points=500,
    )

    criterion = QuantileWeightedMaskedCombinedLoss(
        alpha=args.loss_alpha,
        q_low=q_low_arr,
        q_high=q_high_arr,
        w_low=args.tail_w_low,
        w_high=args.tail_w_high,
    )


    use_grad_scaler = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    # Inverse transform broadcast tensors
    medians = torch.tensor(stats["medians"], device=device).view(1, -1, 1)
    iqrs = torch.tensor(stats["iqrs"], device=device).view(1, -1, 1)

    # Resume support
    best_val = float("inf")
    early_cnt = 0
    start_epoch = 0
    train_log_rows = []

    if args.resume_training and os.path.exists(paths.ckpt_last):
        meta_old = load_json(paths.run_meta) if os.path.exists(paths.run_meta) else {}
        fp_old = meta_old.get("data_fingerprint", {})
        if (not args.ignore_data_mismatch) and fp_old and (not _check_fingerprint(fp_old, fp_now)):
            raise RuntimeError("Data fingerprint changed since last run. Use --ignore_data_mismatch to force resume.")

        ckpt = torch.load(paths.ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if use_grad_scaler and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        best_val = float(ckpt.get("best_val", best_val))
        early_cnt = int(ckpt.get("early_cnt", 0))
        start_epoch = int(ckpt.get("epoch", 0))

        # RNG states (best effort)
        rng = ckpt.get("rng_state", None)
        if rng is not None:
            try:
                random.setstate(rng["py"])
                np.random.set_state(rng["np"])
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception:
                pass

        if os.path.exists(paths.training_log):
            try:
                train_log_rows = pd.read_csv(paths.training_log).to_dict(orient="records")
            except Exception:
                train_log_rows = []

        print(f"Resumed from checkpoint, start_epoch={start_epoch}, best_val={best_val:.6f}")

    # Run meta
    run_meta = {
        "args": vars(args),
        "data_fingerprint": fp_now,
        "stats": stats,
        "device": device_tag,
        "prior": {
            "type": "corr_train_only",
            "prior_beta": float(args.prior_beta),
            "prior_learnable": bool(args.prior_learnable),
            "prior_min_points": int(args.prior_min_points),
        },
        "split": {
            "train": "< 2018-01-01",
            "val": "2018-01-01 .. 2018-06-30",
            "test": "2018-07, 2018-08-01..10, 2018-11-20..30, 2018-12",
            "split_mode": args.split_mode,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_meta, paths.run_meta)

    # Training
    pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs")
    for epoch in pbar:
        model.train()
        train_losses = []
        t0 = time.time()

        for batch in train_loader:
            x, y, m = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, args.amp, args.amp_dtype):
                pred = model(x)
                loss = criterion(pred, y, m)

            if use_grad_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()

            train_losses.append(float(loss.item()))

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        # Validation loss and optional physical metrics (online)
        model.eval()
        val_losses = []
        val_tracker = MetricTracker(args.horizon)

        with torch.no_grad():
            for batch in val_loader:
                x, y, m = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)

                with _autocast_ctx(device, args.amp, args.amp_dtype):
                    pred = model(x)
                    vloss = criterion(pred, y, m)
                val_losses.append(float(vloss.item()))

                # physical metrics
                pred_inv = pred * iqrs + medians
                y_inv = y * iqrs + medians
                val_tracker.update(pred_inv, y_inv, m)

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(avg_val)
        lr = float(optimizer.param_groups[0]["lr"])

        # Adjacency stats
        with torch.no_grad():
            adj = model.compute_adj()
        adj_stats = log_adj_stats(adj)

        # Early stopping on val loss
        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            early_cnt = 0
            torch.save(model.state_dict(), paths.best_model)
        else:
            early_cnt += 1

        epoch_time = time.time() - t0
        pbar.set_postfix(
            {
                "train": f"{avg_train:.5f}",
                "val": f"{avg_val:.5f}",
                "lr": f"{lr:.2e}",
                "pat": f"{early_cnt}/{args.patience}",
            }
        )

        val_metrics = val_tracker.get_metrics()
        row = {
            "epoch": int(epoch + 1),
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": lr,
            "grad_norm": float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm),
            "time_sec": float(epoch_time),
            "best_val": float(best_val),
            "early_cnt": int(early_cnt),
            **adj_stats,
        }
        # log a few key val physical metrics
        if "Overall_RMSE" in val_metrics:
            row["val_overall_rmse"] = val_metrics["Overall_RMSE"]
            row["val_overall_mae"] = val_metrics["Overall_MAE"]

        train_log_rows.append(row)
        save_training_log_csv(train_log_rows, paths.training_log)

        if writer is not None:
            writer.add_scalar("loss/train", avg_train, int(epoch + 1))
            writer.add_scalar("loss/val", avg_val, int(epoch + 1))
            writer.add_scalar("lr", lr, int(epoch + 1))
            writer.add_scalar("grad_norm", row["grad_norm"], int(epoch + 1))
            writer.add_scalar("adj/entropy_mean", adj_stats["adj_entropy_mean"], int(epoch + 1))
            writer.add_scalar("adj/max_mean", adj_stats["adj_max_mean"], int(epoch + 1))
            writer.add_scalar("adj/diag_mean", adj_stats["adj_diag_mean"], int(epoch + 1))
            if "Overall_RMSE" in val_metrics:
                writer.add_scalar("val_phys/overall_rmse", val_metrics["Overall_RMSE"], int(epoch + 1))
                writer.add_scalar("val_phys/overall_mae", val_metrics["Overall_MAE"], int(epoch + 1))

        # Save checkpoint_last
        rng_state = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        ckpt = {
            "epoch": int(epoch + 1),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_grad_scaler else None,
            "best_val": float(best_val),
            "early_cnt": int(early_cnt),
            "args": vars(args),
            "rng_state": rng_state,
        }
        torch.save(ckpt, paths.ckpt_last)

        if early_cnt >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # -----------------------------
    # Final test evaluation
    # -----------------------------
    print("Evaluating on Test Set...")
    if os.path.exists(paths.best_model):
        model.load_state_dict(torch.load(paths.best_model, map_location=device))
    model.eval()

    test_tracker = MetricTracker(args.horizon)

    step1_rows = []
    node_names = stats["nodes"]

    with torch.no_grad():
        for batch in test_loader:
            x, y, m, idx_i = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            pred = model(x)
            pred_inv = pred * iqrs + medians
            y_inv = y * iqrs + medians

            test_tracker.update(pred_inv, y_inv, m)

            # Export only step 1
            p1 = pred_inv[:, :, 0].detach().cpu().numpy()
            t1 = y_inv[:, :, 0].detach().cpu().numpy()
            m1 = m[:, :, 0].detach().cpu().numpy()
            idx_np = idx_i.detach().cpu().numpy()

            # Timestamp for each sample refers to prediction start time i
            ds = index[idx_np].astype("datetime64[ns]")

            for b in range(len(idx_np)):
                row = {"ds": str(pd.Timestamp(ds[b]))}
                for n, name in enumerate(node_names):
                    row[f"{name}_true"] = float(t1[b, n])
                    row[f"{name}_pred"] = float(p1[b, n])
                    row[f"{name}_mask"] = float(m1[b, n])
                step1_rows.append(row)

    metrics = test_tracker.get_metrics()
    save_json(metrics, paths.metrics)

    # Node-wise metrics (N is small, safe)
    # We compute per-node per-step RMSE using online accumulation over test set
    # Here we reuse aggregated sums by recomputing with a second pass if needed.
    # N=4 so we just do one more streaming pass for clarity.
    node_mse = {name: np.zeros(args.horizon, dtype=np.float64) for name in node_names}
    node_cnt = {name: np.zeros(args.horizon, dtype=np.float64) for name in node_names}

    with torch.no_grad():
        for batch in test_loader:
            x, y, m, _ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            pred = model(x)

            pred_inv = (pred * iqrs + medians).detach().float().cpu().numpy()
            y_inv = (y * iqrs + medians).detach().float().cpu().numpy()
            mask_np = m.detach().float().cpu().numpy()

            diff = (pred_inv - y_inv)
            for n, name in enumerate(node_names):
                mse = (diff[:, n, :] ** 2) * mask_np[:, n, :]
                node_mse[name] += mse.sum(axis=0)
                node_cnt[name] += mask_np[:, n, :].sum(axis=0)

    node_metrics: Dict[str, Dict[str, float]] = {}
    for name in node_names:
        node_metrics[name] = {}
        for h in range(args.horizon):
            denom = node_cnt[name][h]
            if denom > 0:
                rmse = float(np.sqrt(node_mse[name][h] / denom))
                node_metrics[name][f"Step_{h+1}_RMSE"] = rmse
    save_json(node_metrics, paths.node_metrics)

    # Export step1 csv
    if step1_rows:
        pd.DataFrame(step1_rows).to_csv(paths.results_step1, index=False)

    print("Done.")
    print(json.dumps(metrics, indent=2))

    if writer is not None:
        writer.flush()
        writer.close()


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--root", required=True, help="Directory containing per-node parquet and scaler_params.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--resolution", type=str, default="5min", choices=["5min", "30min", "1hour"])
    parser.add_argument("--split_mode", type=str, default="strict", choices=["strict", "deploy"])

    # Model
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=288, help="24h @ 5min")
    parser.add_argument("--horizon", type=int, default=12, help="1h @ 5min")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sparsity; for 4 nodes you can leave None")
    parser.add_argument("--self_loop_w", type=float, default=0.2, help="Self-loop weight; 0.2 is usually stable")

    # Correlation prior
    parser.add_argument("--prior_beta", type=float, default=0.5, help="Strength of correlation prior (train-only)")
    parser.add_argument("--prior_learnable", action="store_true", help="Make prior_beta learnable")
    parser.add_argument("--prior_min_points", type=int, default=200, help="Min overlap points for correlation")

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tail_q_low", type=float, default=0.1)
    parser.add_argument("--tail_q_high", type=float, default=0.9)
    parser.add_argument("--tail_w_low", type=float, default=4.0)
    parser.add_argument("--tail_w_high", type=float, default=4.0)


    # Device and perf
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--compile", action="store_true")

    # Loader
    parser.add_argument("--num_workers", type=int, default=min(12, max(2, (os.cpu_count() or 8) - 2)))
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # Repro and system
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))

    # Logging and resume
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--tb_logdir", type=str, default=None)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ignore_data_mismatch", action="store_true")

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
