#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從原始井資料 TXT 構建 aux_vec（44 維）並離線保存：
- 僅用 train_wells 擬合特徵標準化（feature-wise mean/std），對 val_wells 僅做 transform
- 與子圖譜對齊：讀取 train/valid 子圖譜 .npz 的 well_name 與 window_start/window_end
- 生成檔案：optimized_patch_data/precomputed/{train|valid}/aux_vec/{stem}_aux_vec.npy
- 標準化參數：optimized_patch_data/meta/aux_vec_scaler.json

假設曲線列名存在：GR, SP, RT, AC, CNL, DEN 以及深度列（含「深度」字樣）。
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

TRY_PANDAS = True
try:
    import pandas as pd  # type: ignore
except Exception:
    TRY_PANDAS = False


CURVE_NAMES = ["GR", "SP", "RT", "AC", "CNL", "DEN"]


def find_depth_column(df_like) -> str:
    candidates = ["深度", "DEPTH", "Depth", "depth", "MD"]
    for c in candidates:
        if c in df_like:
            return c
    # 寬鬆匹配
    for c in df_like:
        if str(c).find("深") >= 0 or str(c).lower().find("depth") >= 0:
            return c
    raise ValueError("無法在 TXT 中找到深度列名")


def read_well_txt(fp: Path) -> Dict[str, np.ndarray]:
    if TRY_PANDAS:
        try:
            df = pd.read_csv(fp, sep=None, engine="python", encoding="utf-8", na_values=["", " ", "NA", "NaN"])
        except Exception:
            df = pd.read_csv(fp, sep=None, engine="python", encoding="gbk", na_values=["", " ", "NA", "NaN"])
        depth_col = find_depth_column(df.columns)
        out = {k: pd.to_numeric(df[k], errors='coerce').astype(float).to_numpy() for k in df.columns}
        out["__depth_key__"] = depth_col
        return out
    # fallback: numpy
    arr = np.genfromtxt(fp, delimiter=None, dtype=float, encoding=None)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"無法解析 TXT: {fp}")
    # 假設第一列為深度，其餘列按固定順序嘗試映射
    out = {"__depth_key__": "DEPTH_NP"}
    out["DEPTH_NP"] = arr[:, 0]
    for i, name in enumerate(CURVE_NAMES, start=1):
        if i < arr.shape[1]:
            out[name] = arr[:, i]
        else:
            out[name] = np.full(arr.shape[0], np.nan)
    return out


def align_depth_uniform(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    depth_key = raw["__depth_key__"]
    depth = np.asarray(raw[depth_key], dtype=float)
    m = np.isfinite(depth)
    depth = depth[m]
    order = np.argsort(depth)
    depth = depth[order]
    if depth.size < 3:
        raise ValueError("深度點不足以對齊")
    diffs = np.diff(depth)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    step = float(np.median(diffs)) if diffs.size > 0 else 1.0
    new_depth = np.arange(depth.min(), depth.max() + step * 0.5, step, dtype=float)
    aligned = {"深度": new_depth}
    # 對每條曲線做線性插值/近鄰回退
    for name in CURVE_NAMES:
        v = np.asarray(raw.get(name, np.full_like(depth, np.nan)), dtype=float)
        v = v[m][order] if v.shape[0] == m.shape[0] else v
        try:
            # 線性插值（缺值回退最近鄰）
            mask = np.isfinite(v)
            if np.count_nonzero(mask) >= 2:
                aligned[name] = np.interp(new_depth, depth[mask], v[mask])
            elif np.count_nonzero(mask) == 1:
                aligned[name] = np.full_like(new_depth, float(v[mask][0]))
            else:
                aligned[name] = np.full_like(new_depth, 0.0)
        except Exception:
            aligned[name] = np.full_like(new_depth, 0.0)
    return aligned


def compute_window_features(win: Dict[str, np.ndarray]) -> np.ndarray:
    """構建 44 維特徵（統計 + 物理先驗代理）。"""
    feats: List[float] = []
    # 每曲線：mean/std/min/max/p25/p50/p75 -> 7*6 = 42
    for name in CURVE_NAMES:
        x = np.asarray(win[name], dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            x = np.array([0.0])
        q25, q50, q75 = np.percentile(x, [25, 50, 75])
        feats.extend([
            float(np.mean(x)),
            float(np.std(x) + 1e-8),
            float(np.min(x)),
            float(np.max(x)),
            float(q25), float(q50), float(q75)
        ])
    # 物理先驗代理：AI ≈ DEN * (1 / AC) 的 mean/std -> 2，共 44 維
    den = np.asarray(win.get("DEN", np.array([0.0])), dtype=float)
    ac = np.asarray(win.get("AC", np.array([1.0])), dtype=float)
    ai = den * (1.0 / (ac + 1e-6))
    feats.extend([float(np.mean(ai)), float(np.std(ai) + 1e-8)])
    return np.asarray(feats, dtype=np.float32)


def load_npz_meta(fp: Path) -> Tuple[str, int, int]:
    with np.load(fp, mmap_mode='r', allow_pickle=False) as d:
        well = str(d.get('well_name', ''))
        ws = int(d.get('window_start', 0))
        we = int(d.get('window_end', 0))
    return well, ws, we


def build_aux_for_split(base_dir: Path, split: str, scaler_stats: Dict[str, np.ndarray] | None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    spectro_dir = base_dir / ("train_spectrograms" if split == 'train' else "valid_spectrograms")
    wells_dir = base_dir / ("train_wells" if split == 'train' else "val_wells")
    out_dir = base_dir / "precomputed" / split / "aux_vec"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每井讀一次並對齊
    well_cache: Dict[str, Dict[str, np.ndarray]] = {}
    # 暫存特徵以便擬合 scaler
    collected: List[np.ndarray] = []

    files = sorted(list(spectro_dir.glob("*_patch_6ch_32x32_*.npz")))
    for fp in files:
        try:
            stem = fp.stem
            well, ws, we = load_npz_meta(fp)
            if not well:
                # 退化：以檔名前綴猜測井名（不建議，但避免整批失敗）
                well = stem.split("_")[0]
            if well not in well_cache:
                cand = list(wells_dir.glob(f"{well}.*"))
                if not cand:
                    # 寬鬆：嘗試中文名去副檔名匹配
                    cand = list(wells_dir.glob(f"{well}*.txt"))
                if not cand:
                    continue
                raw = read_well_txt(cand[0])
                well_cache[well] = align_depth_uniform(raw)
            aligned = well_cache[well]
            # 取窗口
            start = max(0, ws)
            end = min(len(aligned["深度"]), we)
            if end <= start:
                continue
            win = {k: aligned[k][start:end] for k in CURVE_NAMES}
            feat = compute_window_features(win)
            collected.append(feat)
            # 暫存未標準化，稍後統一寫檔（為減少記憶體可直接寫，這裡保持簡潔）
            np.save(out_dir / f"{stem}_aux_vec.npy", feat)
        except Exception:
            continue

    stats = {"mean": None, "std": None}
    if split == 'train':
        if collected:
            feats = np.stack(collected, axis=0)
            mean = feats.mean(axis=0)
            std = feats.std(axis=0) + 1e-6
            stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
            # 覆寫檔為標準化版本
            for fp in out_dir.glob("*_aux_vec.npy"):
                try:
                    v = np.load(fp).astype(np.float32)
                    v = (v - stats["mean"]) / stats["std"]
                    np.save(fp, v)
                except Exception:
                    pass
        return stats["mean"], stats
    else:
        # 使用給定 scaler_stats 做 transform
        if scaler_stats and scaler_stats.get("mean") is not None:
            mean = scaler_stats["mean"].astype(np.float32)
            std = scaler_stats["std"].astype(np.float32)
            for fp in out_dir.glob("*_aux_vec.npy"):
                try:
                    v = np.load(fp).astype(np.float32)
                    v = (v - mean) / std
                    np.save(fp, v)
                except Exception:
                    pass
        return None, scaler_stats or {"mean": None, "std": None}


def main():
    base_dir = Path("optimized_patch_data")
    meta_dir = base_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # 先處理 train 並擬合 scaler
    train_mean, train_stats = build_aux_for_split(base_dir, 'train', None)
    # 再處理 val 並使用 train 的 scaler
    _, _ = build_aux_for_split(base_dir, 'valid', train_stats)
    # 保存 scaler
    scaler_fp = meta_dir / 'aux_vec_scaler.json'
    payload = {
        'feature_names': [f'{n}_{m}' for n in CURVE_NAMES for m in ['mean','std','min','max','p25','p50','p75']] + ['AI_mean','AI_std'],
        'mean': train_stats['mean'].astype(float).tolist() if train_stats['mean'] is not None else None,
        'std': train_stats['std'].astype(float).tolist() if train_stats['std'] is not None else None,
    }
    with open(scaler_fp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成並保存 aux_vec 與標準化參數: {scaler_fp}")


if __name__ == '__main__':
    main()






