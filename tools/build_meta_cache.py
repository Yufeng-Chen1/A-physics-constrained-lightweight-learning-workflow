#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成資料集統計緩存：optimized_patch_data/meta/dataset_stats.json

用途：避免每次訓練都重新掃描訓練集類別分布。

用法：
  python -u tools/build_meta_cache.py --data_root optimized_patch_data
"""

import argparse
import json
from pathlib import Path
import numpy as np


def build_dataset_stats(data_root: Path) -> dict:
    train_dir = data_root / 'train_spectrograms'
    if not train_dir.exists():
        raise SystemExit(f"找不到訓練資料夾: {train_dir}")

    counts = {}
    total = 0
    patterns = ['*_patch_6ch_*.npz', '*_patch_6ch_32x32_*_*.npz', '*patch_6ch*.npz', '*patch*.npz']
    # 先收集並去重，避免一個檔案被多個通配符重複匹配
    files = set()
    for pat in patterns:
        for fp in train_dir.glob(pat):
            files.add(fp)
    matched = 0
    for fp in sorted(files):
        try:
            with np.load(fp) as d:
                lb = int(d.get('label', -1))
                spec = d['spectrogram']
            # 僅統計 6×32×32 的子圖譜
            if 0 <= lb < 5 and isinstance(spec, np.ndarray) and spec.shape == (6, 32, 32):
                counts[lb] = counts.get(lb, 0) + 1
                total += 1
                matched += 1
        except Exception:
            continue

    if matched == 0:
        raise SystemExit("訓練目錄未匹配到任何 npz 樣本，請檢查路徑與檔名模式。")

    # 估計通道統計（僅 6×32×32 樣本）
    sum_c = np.zeros((6,), dtype=np.float64)
    sumsq_c = np.zeros((6,), dtype=np.float64)
    pixel_count = 0
    for fp in files:
        try:
            with np.load(fp) as d:
                spec = d['spectrogram']
            if isinstance(spec, np.ndarray) and spec.shape == (6, 32, 32):
                ch_flat = spec.reshape(6, -1)
                sum_c += ch_flat.sum(axis=1)
                sumsq_c += (ch_flat ** 2).sum(axis=1)
                pixel_count += ch_flat.shape[1]
        except Exception:
            continue
    channel_mean = None
    channel_std = None
    if pixel_count > 0:
        mean = sum_c / pixel_count
        var = np.maximum(sumsq_c / pixel_count - mean ** 2, 1e-8)
        std = np.sqrt(var)
        channel_mean = mean.astype(np.float32).tolist()
        channel_std = std.astype(np.float32).tolist()

    return {
        'train_class_counts': {int(k): int(v) for k, v in counts.items()},
        'train_total': int(total),
        'channel_mean': channel_mean,
        'channel_std': channel_std,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='optimized_patch_data', type=str)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    meta_dir = data_root / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    stats = build_dataset_stats(data_root)
    out_fp = meta_dir / 'dataset_stats.json'
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('✅ 已寫入:', out_fp)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


