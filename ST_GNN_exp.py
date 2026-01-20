# train.py
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
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
            raise ValueError(f"{fp.name} missing mask_missing or mask_hard")

        dfs_scaled.append(df[["y_scaled"]].rename(columns={"y_scaled": node_key}))

        m_miss = pd.to_numeric(df["mask_missing"], errors="coerce").fillna(0).astype(int)
        m_hard = pd.to_numeric(df["mask_hard"], errors="coerce").fillna(0).astype(int)
        if "mask_stl" in df.columns:
            m_stl = pd.to_numeric(df["mask_stl"], errors="coerce").fillna(0).astype(int)
        else:
            m_stl = pd.Series(0, index=df.index, dtype=int)

        invalid_input = (m_miss | m_hard).astype(int)
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

    merged_scaled = merged_scaled.fillna(0.0)
    merged_in = merged_in.fillna(1).astype(np.float32)
    merged_tgt = merged_tgt.fillna(1).astype(np.float32)

    freq_map = {"5min": "5min", "30min": "30min", "1hour": "1h"}
    if target_resolution not in freq_map:
        raise ValueError(f"Unknown resolution: {target_resolution}")

    full_idx = pd.date_range(merged_scaled.index.min(), merged_scaled.index.max(), freq=freq_map[target_resolution])

    merged_scaled = merged_scaled.reindex(full_idx).fillna(0.0).astype(np.float32)
    merged_in = merged_in.reindex(full_idx).fillna(1.0).astype(np.float32)
    merged_tgt = merged_tgt.reindex(full_idx).fillna(1.0).astype(np.float32)

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
    valid_per_t = (1.0 - mask_tgt).sum(axis=1)
    prefix = np.zeros(len(valid_per_t) + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(valid_per_t)
    counts = prefix[indices + horizon] - prefix[indices]
    return indices[counts > 0]


def compute_corr_prior(
    values: np.ndarray,
    mask_in: np.ndarray,
    train_mask: np.ndarray,
    min_points: int = 200,
    zero_diag: bool = False,
) -> np.ndarray:
    _, N = values.shape
    prior = np.zeros((N, N), dtype=np.float32)
    valid_global = (train_mask[:, None]) & (mask_in == 0)

    if not zero_diag:
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
    values: np.ndarray,
    mask_tgt: np.ndarray,
    train_mask: np.ndarray,
    q_low: float,
    q_high: float,
    min_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    _, N = values.shape
    ql = np.zeros(N, dtype=np.float32)
    qh = np.zeros(N, dtype=np.float32)

    for n in range(N):
        valid = train_mask & (mask_tgt[:, n] == 0)
        data = values[valid, n]
        if data.size < min_points:
            data = values[train_mask, n]
        ql[n] = float(np.quantile(data, q_low))
        qh[n] = float(np.quantile(data, q_high))
    return ql, qh


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

        v_win = self.values[w_start:w_end]
        m_in_win = self.mask_in[w_start:w_end]
        t_win = self.time_feats[w_start:w_end]

        v_win_clean = v_win * (1.0 - m_in_win)

        x = np.concatenate(
            [
                v_win_clean[..., None],
                m_in_win[..., None],
                np.repeat(t_win[:, None, :], self.n_nodes, axis=1),
            ],
            axis=2,
        ).astype(np.float32)

        y = self.values[w_end:pred_end].T.astype(np.float32)
        m_t = self.mask_tgt[w_end:pred_end].T.astype(np.float32)
        y_mask = (1.0 - m_t).astype(np.float32)

        if self.return_index:
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask), torch.tensor(i, dtype=torch.long)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask)


class SignedQuantileWeightedMaskedCombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float,
        q_low: np.ndarray,
        q_high: np.ndarray,
        w_low_over: float = 0.0,
        w_low_under: float = 0.0,
        w_high_under: float = 0.0,
        w_high_over: float = 0.0,
        eps: float = 1e-6,
        tail_lambda_init: float = 0.0,
        node_weights: Optional[np.ndarray] = None,
        sign_tol: float = 1e-4,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.w_low_over = float(w_low_over)
        self.w_low_under = float(w_low_under)
        self.w_high_under = float(w_high_under)
        self.w_high_over = float(w_high_over)
        self.eps = float(eps)
        self.sign_tol = float(sign_tol)

        ql = torch.tensor(q_low, dtype=torch.float32).view(1, -1, 1)
        qh = torch.tensor(q_high, dtype=torch.float32).view(1, -1, 1)
        self.register_buffer("q_low", ql)
        self.register_buffer("q_high", qh)
        self.register_buffer("tail_lambda", torch.tensor(float(tail_lambda_init), dtype=torch.float32))

        if node_weights is None:
            nw = torch.ones(len(q_low), dtype=torch.float32)
        else:
            if len(node_weights) != len(q_low):
                raise ValueError("node_weights length must match number of nodes")
            nw = torch.tensor(node_weights, dtype=torch.float32)
        self.register_buffer("node_w", nw.view(1, -1, 1))

    def set_tail_lambda(self, value: float):
        self.tail_lambda.fill_(float(value))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        ql = self.q_low.to(target.device)
        qh = self.q_high.to(target.device)
        lam = self.tail_lambda.to(target.device)
        nw = self.node_w.to(target.device)

        diff = pred - target
        tol = self.sign_tol
        over = diff > tol
        under = diff < -tol

        low = target < ql
        high = target > qh

        w = torch.ones_like(target)

        if self.w_low_over != 0.0:
            w = torch.where(low & over, w * (1.0 + lam * self.w_low_over), w)
        if self.w_low_under != 0.0:
            w = torch.where(low & under, w * (1.0 + lam * self.w_low_under), w)
        if self.w_high_under != 0.0:
            w = torch.where(high & under, w * (1.0 + lam * self.w_high_under), w)
        if self.w_high_over != 0.0:
            w = torch.where(high & over, w * (1.0 + lam * self.w_high_over), w)

        eff = mask * w * nw
        denom = eff.sum().clamp(min=self.eps)

        mse_part = (diff ** 2) * eff
        mae_part = diff.abs() * eff
        return self.alpha * (mse_part.sum() / denom) + (1.0 - self.alpha) * (mae_part.sum() / denom)


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
        self_loop_w: float = 0.2,
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
            learned = torch.mm(emb1, emb2.t())

            beta = self._beta().float()
            logits = learned + beta * self.prior_logits.to(learned.device)
            logits = F.relu(logits)

            # =======================================================
            # 【核弹级操作】物理屏蔽对角线
            # 无论 logits 算出来是多少，强制把对角线变成负无穷
            # 这样 Softmax 之后，对角线权重 100% 为 0
            # =======================================================
            mask = torch.eye(self.num_nodes, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(mask, -1e9)
            # =======================================================

            if self.top_k is not None:
                # 注意：这里 top_k 可能会选到 -1e9 的值（如果邻居不够多），但没关系
                k = min(max(int(self.top_k), 1), self.num_nodes)
                _, idx = torch.topk(logits, k=k, dim=1)
                keep = torch.zeros_like(logits, dtype=torch.bool)
                keep.scatter_(1, idx, True)
                logits = logits.masked_fill(~keep, -1e9)

            adj = F.softmax(logits, dim=1)
            
            # 确保这里没有 self_loop_w 的干扰
            if self.self_loop_w > 0:
                 adj = adj + self.self_loop_w * torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
                 adj = adj / adj.sum(dim=1, keepdim=True)
            
            return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, n_feat = x.shape
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            adj = self.compute_adj()

        x_reshaped = x.view(B * T, N, n_feat)
        gcn_in = self.gcn_weights(x_reshaped)

        adj_expand = adj.unsqueeze(0)
        gcn_out = torch.matmul(adj_expand, gcn_in.view(B * T, N, self.hidden_dim))
        gcn_out = gcn_out.view(B, T, N, self.hidden_dim)
        gcn_out = F.leaky_relu(gcn_out)

        gcn_flat = gcn_out.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        lstm_out, _ = self.lstm(gcn_flat)
        last_step = lstm_out[:, -1, :]

        out = self.fc(self.dropout(last_step))
        return out.view(B, N, self.horizon)


class MetricTracker:
    def __init__(self, horizon: int, mape_eps: float = 1e-3):
        self.horizon = int(horizon)
        self.mape_eps = float(mape_eps)
        self.reset()

    def reset(self):
        H = self.horizon
        self.total_mse = np.zeros(H, dtype=np.float64)
        self.total_mae = np.zeros(H, dtype=np.float64)
        self.total_count = np.zeros(H, dtype=np.float64)

        self.overall_mse_sum = 0.0
        self.overall_mae_sum = 0.0
        self.overall_count = 0.0

        self.total_sse = np.zeros(H, dtype=np.float64)
        self.total_sum_y = np.zeros(H, dtype=np.float64)
        self.total_sum_y2 = np.zeros(H, dtype=np.float64)
        self.total_r2_count = np.zeros(H, dtype=np.float64)

        self.overall_sse = 0.0
        self.overall_sum_y = 0.0
        self.overall_sum_y2 = 0.0
        self.overall_r2_count = 0.0

        self.total_mape_sum = np.zeros(H, dtype=np.float64)
        self.total_mape_count = np.zeros(H, dtype=np.float64)

        self.overall_mape_sum = 0.0
        self.overall_mape_count = 0.0

    def _r2_from_sums(self, sse: float, sum_y: float, sum_y2: float, cnt: float) -> Optional[float]:
        if cnt <= 0:
            return None
        sst = sum_y2 - (sum_y * sum_y) / cnt
        if sst <= 1e-12:
            return None
        return float(1.0 - (sse / sst))

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

        sse_sum = ((diff ** 2) * mask).sum(dim=(0, 1)).numpy()
        self.total_sse += sse_sum

        y_sum = (true * mask).sum(dim=(0, 1)).numpy()
        y2_sum = ((true ** 2) * mask).sum(dim=(0, 1)).numpy()
        self.total_sum_y += y_sum
        self.total_sum_y2 += y2_sum
        self.total_r2_count += cnt_sum

        self.overall_sse += float(((diff ** 2) * mask).sum().item())
        self.overall_sum_y += float((true * mask).sum().item())
        self.overall_sum_y2 += float(((true ** 2) * mask).sum().item())
        self.overall_r2_count += float(mask.sum().item())

        eps = self.mape_eps
        denom_ok = (true.abs() > eps).float()
        mape_mask = mask * denom_ok
        mape = (diff.abs() / true.abs().clamp(min=eps)) * mape_mask

        self.total_mape_sum += mape.sum(dim=(0, 1)).numpy()
        self.total_mape_count += mape_mask.sum(dim=(0, 1)).numpy()

        self.overall_mape_sum += float(mape.sum().item())
        self.overall_mape_count += float(mape_mask.sum().item())

    def get_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self.overall_count > 0:
            mse_all = self.overall_mse_sum / self.overall_count
            metrics["Overall_RMSE"] = float(np.sqrt(mse_all))
            metrics["Overall_MAE"] = float(self.overall_mae_sum / self.overall_count)

        r2_all = self._r2_from_sums(
            sse=self.overall_sse,
            sum_y=self.overall_sum_y,
            sum_y2=self.overall_sum_y2,
            cnt=self.overall_r2_count,
        )
        if r2_all is not None:
            metrics["Overall_R2"] = float(r2_all)

        if self.overall_mape_count > 0:
            metrics["Overall_MAPE"] = float(self.overall_mape_sum / self.overall_mape_count)

        for h in range(self.horizon):
            denom = self.total_count[h]
            if denom > 0:
                mse_h = self.total_mse[h] / denom
                mae_h = self.total_mae[h] / denom
                metrics[f"Step_{h+1}_RMSE"] = float(np.sqrt(mse_h))
                metrics[f"Step_{h+1}_MAE"] = float(mae_h)

            r2_h = self._r2_from_sums(
                sse=float(self.total_sse[h]),
                sum_y=float(self.total_sum_y[h]),
                sum_y2=float(self.total_sum_y2[h]),
                cnt=float(self.total_r2_count[h]),
            )
            if r2_h is not None:
                metrics[f"Step_{h+1}_R2"] = float(r2_h)

            if self.total_mape_count[h] > 0:
                metrics[f"Step_{h+1}_MAPE"] = float(self.total_mape_sum[h] / self.total_mape_count[h])

        return metrics


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
    a = adj.detach().float().cpu().numpy()
    eps = 1e-12
    row_entropy = -(a * np.log(a + eps)).sum(axis=1)
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
    pd.DataFrame(rows).to_csv(path, index=False)


def train_model(args):
    configure_threads(args.cpu_threads)
    setup_seed(args.seed)
    device, device_tag = select_device(args.device)

    paths = make_paths(args.out)
    fp_now = dir_fingerprint(args.root)

    SummaryWriter, tb_src = _get_summary_writer()
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        tb_dir = args.tb_logdir or os.path.join(args.out, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard enabled: {tb_dir} ({tb_src})")

    df_scaled, df_in, df_tgt, stats = load_data_and_params(args.root, args.resolution)
    index = df_scaled.index
    values = df_scaled.values.astype(np.float32)
    mask_in = df_in.values.astype(np.float32)
    mask_tgt = df_tgt.values.astype(np.float32)
    time_feats = get_time_features(index)

    train_mask, val_mask, test_mask = build_split_masks(index)

    q_low_arr, q_high_arr = compute_tail_quantiles_per_node(
        values=values,
        mask_tgt=mask_tgt,
        train_mask=train_mask,
        q_low=args.tail_q_low,
        q_high=args.tail_q_high,
        min_points=500,
    )

    prior_np = compute_corr_prior(
        values=values,
        mask_in=mask_in,
        train_mask=train_mask,
        min_points=args.prior_min_points,
        zero_diag=args.zero_prior_diag,
    )
    prior_t = torch.from_numpy(prior_np).float()

    if args.split_mode == "deploy":
        idx_train = get_indices_deploy(train_mask, args.lookback, args.horizon)
        idx_val = get_indices_deploy(val_mask, args.lookback, args.horizon)
        idx_test = get_indices_deploy(test_mask, args.lookback, args.horizon)
    else:
        idx_train = get_indices_strict(train_mask, args.lookback, args.horizon)
        idx_val = get_indices_strict(val_mask, args.lookback, args.horizon)
        idx_test = get_indices_strict(test_mask, args.lookback, args.horizon)

    idx_train = filter_indices_with_targets(idx_train, mask_tgt, args.horizon)
    idx_val = filter_indices_with_targets(idx_val, mask_tgt, args.horizon)
    idx_test = filter_indices_with_targets(idx_test, mask_tgt, args.horizon)

    idx_train = idx_train[:: args.stride]
    idx_val = idx_val[:: args.stride]
    idx_test = idx_test[:: args.stride]

    print(f"Split mode: {args.split_mode}")
    print(f"Windows: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        raise ValueError("One of the splits has zero valid windows. Check lookback, horizon, split dates.")

    train_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_train, args.lookback, args.horizon, return_index=False)
    val_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_val, args.lookback, args.horizon, return_index=False)
    test_ds = WindowDataset(values, mask_in, mask_tgt, time_feats, idx_test, args.lookback, args.horizon, return_index=True)

    use_pin = device.type in ("cuda", "xpu")
    persistent_ok = (args.num_workers > 0) and (os.name != "nt")
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent_ok,
        drop_last=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

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
            print(f"torch.compile failed: {e}")

    if args.init_from_best_path is not None and str(args.init_from_best_path).strip() != "":
        init_path = str(args.init_from_best_path)
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"init_from_best_path not found: {init_path}")

        checkpoint = torch.load(init_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
            print(f"Extracting model_state from full checkpoint: {init_path}")
        else:
            state_dict = checkpoint
            print(f"Loading weights directly: {init_path}")

        # --- 核心修改：删除 Prior 相关和 Embedding 相关的权重 ---
        keys_to_reset = ["prior_beta", "node_emb1", "node_emb2"] # 强制重置图结构的学习参数
        for k in keys_to_reset:
            if k in state_dict:
                del state_dict[k]
                print(f"Force reset parameter: {k}")
        # -----------------------------------------------------

        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded init weights (strict=False). Graph Learner reset to random.")

        args.resume_training = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    node_weights = None
    if args.node_weights is not None and str(args.node_weights).strip() != "":
        try:
            arr = np.array([float(x) for x in str(args.node_weights).split(",")], dtype=np.float32)
            if len(arr) != len(stats["nodes"]):
                raise ValueError("node_weights length mismatch")
            node_weights = arr
        except Exception as e:
            raise ValueError(f"Failed to parse --node_weights: {e}")

    criterion = SignedQuantileWeightedMaskedCombinedLoss(
        alpha=args.loss_alpha,
        q_low=q_low_arr,
        q_high=q_high_arr,
        w_low_over=args.w_low_over,
        w_low_under=args.w_low_under,
        w_high_under=args.w_high_under,
        w_high_over=args.w_high_over,
        tail_lambda_init=1.0 if args.finetune_mode else 0.0,
        node_weights=node_weights,
        sign_tol=args.sign_tol,
    )

    use_grad_scaler = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    medians = torch.tensor(stats["medians"], device=device).view(1, -1, 1)
    iqrs = torch.tensor(stats["iqrs"], device=device).view(1, -1, 1)

    best_val = float("inf")
    early_cnt = 0
    start_epoch = 0
    train_log_rows = []

    scheduler_type = "plateau" if args.finetune_mode else "onecycle"
    scheduler = None

    if args.resume_training and os.path.exists(paths.ckpt_last):
        meta_old = load_json(paths.run_meta) if os.path.exists(paths.run_meta) else {}
        fp_old = meta_old.get("data_fingerprint", {})
        if (not args.ignore_data_mismatch) and fp_old and (not _check_fingerprint(fp_old, fp_now)):
            raise RuntimeError("Data fingerprint changed since last run. Use --ignore_data_mismatch to force resume.")

        ckpt = torch.load(paths.ckpt_last, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))

        if use_grad_scaler and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        best_val = float(ckpt.get("best_val", best_val))
        early_cnt = int(ckpt.get("early_cnt", 0))
        start_epoch = int(ckpt.get("epoch", 0))

        sch_info = ckpt.get("scheduler_info", {})
        ckpt_sch_type = sch_info.get("type", None)
        if ckpt_sch_type is not None and ckpt_sch_type != scheduler_type:
            raise RuntimeError(f"Scheduler type mismatch in resume: ckpt={ckpt_sch_type}, current={scheduler_type}")

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

        print(f"Resumed weights from checkpoint_last.pth, next epoch={start_epoch + 1}, best_val={best_val:.6f}")

    if args.finetune_mode:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            min_lr=args.plateau_min_lr,
        )
    else:
        steps_per_epoch = len(train_loader)
        if steps_per_epoch <= 0:
            raise RuntimeError("train_loader is empty, cannot build OneCycleLR")
        remaining_epochs = max(1, args.epochs - start_epoch)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=remaining_epochs * steps_per_epoch,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
        )

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
            "zero_prior_diag": bool(args.zero_prior_diag),
            "self_loop_w": float(args.self_loop_w),
        },
        "loss": {
            "type": "SignedQuantileWeightedMaskedCombinedLoss",
            "alpha": float(args.loss_alpha),
            "tail_q_low": float(args.tail_q_low),
            "tail_q_high": float(args.tail_q_high),
            "w_low_over": float(args.w_low_over),
            "w_low_under": float(args.w_low_under),
            "w_high_under": float(args.w_high_under),
            "w_high_over": float(args.w_high_over),
            "tail_warmup_epochs": int(args.tail_warmup_epochs),
            "finetune_mode": bool(args.finetune_mode),
            "sign_tol": float(args.sign_tol),
            "node_weights": (node_weights.tolist() if node_weights is not None else None),
        },
        "scheduler": {"type": scheduler_type},
        "split": {
            "train": "< 2018-01-01",
            "val": "2018-01-01 .. 2018-06-30",
            "test": "2018-07, 2018-08-01..10, 2018-11-20..30, 2018-12",
            "split_mode": args.split_mode,
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_meta, paths.run_meta)

    pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs")
    for epoch in pbar:
        local_epoch = epoch - start_epoch
        if args.finetune_mode:
            tail_lambda = 1.0
        else:
            warm = max(1, int(args.tail_warmup_epochs))
            tail_lambda = min(1.0, float(local_epoch + 1) / float(warm))
        criterion.set_tail_lambda(tail_lambda)

        if writer is not None:
            writer.add_scalar("tail/lambda", tail_lambda, int(epoch + 1))

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

            if not args.finetune_mode:
                scheduler.step()

            train_losses.append(float(loss.item()))

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses = []
        val_tracker = MetricTracker(args.horizon, mape_eps=args.mape_eps)

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

                pred_inv = pred * iqrs + medians
                y_inv = y * iqrs + medians
                val_tracker.update(pred_inv, y_inv, m)

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")

        if args.finetune_mode:
            scheduler.step(avg_val)

        lr = float(optimizer.param_groups[0]["lr"])

        with torch.no_grad():
            adj = model.compute_adj()
        adj_stats = log_adj_stats(adj)

        improved = avg_val < best_val
        if improved:
            best_val = avg_val
            early_cnt = 0
            torch.save(model.state_dict(), paths.best_model)
            if args.backup_best_each_improve:
                backup_path = paths.best_model.replace(".pth", f"_e{epoch+1}.pth")
                torch.save(model.state_dict(), backup_path)
        else:
            early_cnt += 1

        epoch_time = time.time() - t0
        pbar.set_postfix({"train": f"{avg_train:.5f}", "val": f"{avg_val:.5f}", "lr": f"{lr:.2e}", "pat": f"{early_cnt}/{args.patience}"})

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
            "tail_lambda": float(tail_lambda),
            **adj_stats,
        }
        for k, v in val_metrics.items():
            row[f"val_{k.lower()}"] = float(v)

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
            if "Overall_MAE" in val_metrics:
                writer.add_scalar("val_phys/overall_mae", val_metrics["Overall_MAE"], int(epoch + 1))
            if "Overall_R2" in val_metrics:
                writer.add_scalar("val_phys/overall_r2", val_metrics["Overall_R2"], int(epoch + 1))
            if "Overall_MAPE" in val_metrics:
                writer.add_scalar("val_phys/overall_mape", val_metrics["Overall_MAPE"], int(epoch + 1))

        rng_state = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        sch_state = scheduler.state_dict() if scheduler is not None else None
        ckpt = {
            "epoch": int(epoch + 1),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": sch_state,
            "scheduler_info": {"type": scheduler_type},
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

    print("Evaluating on Test Set...")
    if os.path.exists(paths.best_model):
        model.load_state_dict(torch.load(paths.best_model, map_location=device))
    model.eval()

    test_tracker = MetricTracker(args.horizon, mape_eps=args.mape_eps)
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

            p1 = pred_inv[:, :, 0].detach().cpu().numpy()
            t1 = y_inv[:, :, 0].detach().cpu().numpy()
            m1 = m[:, :, 0].detach().cpu().numpy()
            idx_np = idx_i.detach().cpu().numpy()

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

            diff = pred_inv - y_inv
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

    if step1_rows:
        pd.DataFrame(step1_rows).to_csv(paths.results_step1, index=False)

    print("Done.")
    print(json.dumps(metrics, indent=2))

    if writer is not None:
        writer.flush()
        writer.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True, help="Directory containing per-node parquet and scaler_params.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--resolution", type=str, default="5min", choices=["5min", "30min", "1hour"])
    parser.add_argument("--split_mode", type=str, default="strict", choices=["strict", "deploy"])

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lookback", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--self_loop_w", type=float, default=0.2)

    parser.add_argument("--prior_beta", type=float, default=0.5)
    parser.add_argument("--prior_learnable", action="store_true")
    parser.add_argument("--prior_min_points", type=int, default=200)
    parser.add_argument("--zero_prior_diag", action="store_true")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_alpha", type=float, default=0.7)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--tail_q_low", type=float, default=0.1)
    parser.add_argument("--tail_q_high", type=float, default=0.9)
    parser.add_argument("--w_low_over", type=float, default=0.0)
    parser.add_argument("--w_low_under", type=float, default=0.0)
    parser.add_argument("--w_high_under", type=float, default=0.0)
    parser.add_argument("--w_high_over", type=float, default=0.0)
    parser.add_argument("--tail_warmup_epochs", type=int, default=10)
    parser.add_argument("--sign_tol", type=float, default=1e-4)
    parser.add_argument("--node_weights", type=str, default=None, help="Comma-separated floats, length must equal num_nodes")

    parser.add_argument("--finetune_mode", action="store_true")
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--plateau_threshold", type=float, default=1e-4)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-6)

    parser.add_argument("--onecycle_pct_start", type=float, default=0.1)
    parser.add_argument("--onecycle_div_factor", type=float, default=10.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=100.0)

    parser.add_argument("--mape_eps", type=float, default=1e-3)
    parser.add_argument("--backup_best_each_improve", action="store_true")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--num_workers", type=int, default=min(12, max(2, (os.cpu_count() or 8) - 2)))
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_threads", type=int, default=max(1, (os.cpu_count() or 8) - 2))

    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--tb_logdir", type=str, default=None)
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ignore_data_mismatch", action="store_true")
    parser.add_argument("--init_from_best_path", type=str, default=None)

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
