#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
離線批量生成 wpt_vec（2級 2D DWT 子帶能量與熵統計），對齊子圖譜檔名：
- 輸入：optimized_patch_data/{train_spectrograms|valid_spectrograms}/*.npz
- 輸出：optimized_patch_data/precomputed/{train|valid}/wpt_vec/{stem}_wpt_vec.npy

備註：與 data/six_channel_loader.py 的在線計算一致（haar，小波層級=2，對每通道 8 個子帶提取 [能量, 熵]）。
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np

try:
    import pywt
except Exception as e:
    raise SystemExit("需要安裝 pywt，請先 pip install PyWavelets")


def compute_wpt_vec(spec6: np.ndarray) -> np.ndarray:
    # spec6: (6, 32, 32)
    feats = []
    for c in range(spec6.shape[0]):
        arr = spec6[c]
        cA1, (cH1, cV1, cD1) = pywt.dwt2(arr, 'haar')
        cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, 'haar')
        subbands = [cA1, cH1, cV1, cD1, cA2, cH2, cV2, cD2]
        for sb in subbands:
            e = float((sb * sb).mean())
            sbp = np.abs(sb)
            s = sbp.sum()
            if s <= 1e-8:
                h = 0.0
            else:
                p = (sbp / s).ravel() + 1e-12
                h = float(-(p * np.log(p)).sum())
            feats.extend([e, h])
    wpt_vec = np.asarray(feats, dtype=np.float32)
    wpt_vec = np.nan_to_num(wpt_vec, nan=0.0, posinf=0.0, neginf=0.0)
    wpt_vec = np.clip(wpt_vec, -10.0, 10.0)
    return wpt_vec


def process_split(data_root: str, split_dir: str) -> int:
    base = Path(data_root)
    src = base / split_dir
    out = base / 'precomputed' / ('train' if 'train' in split_dir else 'valid') / 'wpt_vec'
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for fp in sorted(src.glob('*.npz')):
        stem = fp.stem
        out_fp = out / f'{stem}_wpt_vec.npy'
        if out_fp.exists():
            count += 1
            continue
        try:
            with np.load(fp, mmap_mode='r') as d:
                spec = d['spectrogram']
            if spec.shape != (6, 32, 32):
                continue
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
            wpt = compute_wpt_vec(spec)
            np.save(out_fp, wpt)
            count += 1
        except Exception:
            continue
    return count


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Build offline wpt_vec for train/valid splits')
    ap.add_argument('--data_root', type=str, default='optimized_patch_data')
    args = ap.parse_args()
    n1 = process_split(args.data_root, 'train_spectrograms')
    n2 = process_split(args.data_root, 'valid_spectrograms')
    print(f'✅ wpt_vec 生成完成: train={n1}, valid={n2}')


if __name__ == '__main__':
    main()




