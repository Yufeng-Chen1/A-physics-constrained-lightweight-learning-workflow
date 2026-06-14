#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成弱BSMOTE後備索引緩存：optimized_patch_data/meta/train_indices_bsmote.json

策略（與訓練一致的溫和上采樣）：
- 基於實際標籤計數設定目標上限 cap = min(max_count, int(median*1.1))
- 類3/4/1進行適度補樣，其他類保持原樣
- 固定隨機種子以可重現

用法：
  python -u tools/build_bsmote_indices.py --data_root optimized_patch_data --seed 42
"""

import argparse
import json
from pathlib import Path
import numpy as np
import random


def load_train_files(data_root: Path):
    train_dir = data_root / 'train_spectrograms'
    if not train_dir.exists():
        raise SystemExit(f"找不到訓練資料夾: {train_dir}")
    patterns = ['*_patch_6ch_*.npz', '*_patch_6ch_32x32_*_*.npz', '*patch_6ch*.npz', '*patch*.npz']
    files = set()
    for pat in patterns:
        for fp in train_dir.glob(pat):
            files.add(fp)
    files = sorted(files)
    labels = []
    keep = []
    for fp in files:
        try:
            with np.load(fp) as d:
                lb = int(d.get('label', -1))
                spec = d['spectrogram']
            # 僅保留 6×32×32
            if 0 <= lb < 5 and isinstance(spec, np.ndarray) and getattr(spec, 'shape', None) == (6, 32, 32):
                labels.append(lb)
                keep.append(fp)
        except Exception:
            continue
    return keep, labels


def build_indices(labels, seed: int = 42):
    rng = random.Random(seed)
    # 分類索引
    by_cls = {i: [] for i in range(5)}
    for idx, lb in enumerate(labels):
        by_cls[int(lb)].append(idx)
    # 目標cap
    counts = [len(by_cls[i]) for i in range(5)]
    if not any(counts):
        return list(range(len(labels)))
    med = int(np.median([c for c in counts if c > 0]))
    max_c = max(counts)
    cap = min(max_c, int(med * 1.1))
    new_indices = []
    for cid in range(5):
        cls_idx = by_cls[cid]
        cur = len(cls_idx)
        target = min(cur, cap)
        if cid in (3, 4, 1):
            target = max(target, int(0.9 * cap))
        # 補樣
        if target > cur and cur > 0:
            extra = target - cur
            for _ in range(extra):
                new_indices.append(rng.choice(cls_idx))
        new_indices.extend(cls_idx)
    rng.shuffle(new_indices)
    return new_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='optimized_patch_data')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    meta_dir = data_root / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    files, labels = load_train_files(data_root)
    indices = build_indices(labels, seed=args.seed)

    out_fp = meta_dir / 'train_indices_bsmote.json'
    with open(out_fp, 'w', encoding='utf-8') as f:
        json.dump({'indices': indices, 'num_files': len(files), 'num_labels': len(labels)}, f, ensure_ascii=False)

    print('✅ 已寫入:', out_fp)
    print(f' - 原始樣本數: {len(labels)}')
    print(f' - 索引長度: {len(indices)}')


if __name__ == '__main__':
    main()
