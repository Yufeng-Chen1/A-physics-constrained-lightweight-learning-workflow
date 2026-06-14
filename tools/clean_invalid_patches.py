#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掃描並刪除無效子圖譜：形狀非 (6,32,32) 或讀取失敗。
目標目錄：optimized_patch_data/train_spectrograms、valid_spectrograms、test_spectrograms

用法：
  python -u tools/clean_invalid_patches.py --data_root optimized_patch_data --dry_run 0
"""

import argparse
from pathlib import Path
import numpy as np


def scan_and_clean(dir_path: Path, dry_run: bool = False):
    if not dir_path.exists():
        return 0, 0
    total = 0
    removed = 0
    for fp in dir_path.glob('*.npz'):
        total += 1
        bad = False
        try:
            with np.load(fp) as d:
                spec = d['spectrogram']
            if not (hasattr(spec, 'shape') and tuple(spec.shape) == (6, 32, 32)):
                bad = True
        except Exception:
            bad = True
        if bad:
            removed += 1
            if not dry_run:
                try:
                    fp.unlink(missing_ok=True)
                except Exception:
                    pass
    return total, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default='optimized_patch_data')
    ap.add_argument('--dry_run', type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root)
    total = 0
    removed = 0
    for sub in ['train_spectrograms', 'valid_spectrograms', 'test_spectrograms']:
        t, r = scan_and_clean(root / sub, dry_run=bool(args.dry_run))
        total += t
        removed += r
        print(f'{sub}: 檢查 {t} 刪除 {r}')
    print(f'總計：檢查 {total} 刪除 {removed} (dry_run={bool(args.dry_run)})')


if __name__ == '__main__':
    main()




