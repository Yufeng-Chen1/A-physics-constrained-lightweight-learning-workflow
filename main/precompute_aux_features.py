#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
離線預計算輔助特徵：
- 對 optimized_patch_data/{train,valid}_spectrograms 內的 6通道 32×32 npz：
  1) 生成 aug_channels（低/高頻能量、可選互相關熱圖上採樣到32×32）
  2) 生成 aux_vec（頻帶能量比、能量佔比、跨道相關、敏感頻段極值、相位一致性）
  3) 生成 stats_vec（與訓練期一致的通道統計與多尺度塊特徵），供訓練/驗證直接使用
  結果保存為同名 sidecar：<name>_aux.npz

注意：參數（如頻帶row、極值閾值）只允許用訓練集擬合後再用於驗證/測試。
"""

import os
from pathlib import Path
import numpy as np
import argparse

def list_npz(dir_path: Path):
    pats = ["*_patch_6ch_*.npz", "*_patch_6ch_32x32_*_*.npz"]
    files = []
    for p in pats:
        files.extend(list(dir_path.glob(p)))
    # 排除 sidecar 檔
    files = [fp for fp in files if not str(fp).endswith('_aux.npz')]
    files = sorted(set(files), key=lambda p: str(p))
    return files

def compute_band_rows(H: int):
    # 固定行索引近似頻帶，後續可按需要改為根據訓練集適配
    return {'low': (4, 10), 'mid': (10, 20), 'high': (20, 28)}

def fit_thresholds_from_train(train_files, rows_cfg):
    # 擬合敏感頻段極值閾值（簡單取訓練集分位數作為穩定代理）
    lows = []
    highs = []
    for fp in train_files:
        try:
            with np.load(fp) as d:
                spec = d['spectrogram']  # (6,32,32)
            base = np.abs(spec[:6])
            sl_low = slice(*rows_cfg['low'])
            sl_high = slice(*rows_cfg['high'])
            lows.append(base[:, sl_low, :].ravel())
            highs.append(base[:, sl_high, :].ravel())
        except Exception:
            continue
    thr_low = 0.0
    thr_high = 0.0
    try:
        if lows:
            lows_cat = np.concatenate(lows)
            thr_low = float(np.quantile(lows_cat, 0.95))
        if highs:
            highs_cat = np.concatenate(highs)
            thr_high = float(np.quantile(highs_cat, 0.95))
    except Exception:
        thr_low, thr_high = 1.5, 1.5
    return {'low': thr_low, 'high': thr_high}

def build_sidecar(spec6: np.ndarray, rows_cfg, thr_cfg, heatmap_pairs=((0,2),(1,4)), enable_heatmaps: bool = True):
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(spec6).float()  # (6,32,32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.clamp(x, -10.0, 10.0)
    with torch.no_grad():
        x1 = x.unsqueeze(0)
        low = F.avg_pool2d(x1, kernel_size=3, stride=1, padding=1)
        high = x1 - low
        low_e = low.pow(2)
        high_e = high.pow(2)
        denom = x1.pow(2).mean(dim=(-2,-1), keepdim=True) + 1e-3
        low_e = (low_e / denom).squeeze(0)
        high_e = (high_e / denom).squeeze(0)
        aug_channels = torch.cat([low_e, high_e], dim=0)  # (12,32,32)

        # 熱圖（可選）：為提速可關閉
        heatmaps = torch.zeros(0,32,32)
        if enable_heatmaps:
            hms = []
            for (i,j) in heatmap_pairs:
                i = int(max(0, min(5, i))); j = int(max(0, min(5, j)))
                a = x[i]; b = x[j]
                hm = torch.zeros(8,8)
                for bi in range(8):
                    for bj in range(8):
                        r0,r1 = bi*4, bi*4+4
                        c0,c1 = bj*4, bj*4+4
                        aa = a[r0:r1, c0:c1].reshape(-1)
                        bb = b[r0:r1, c0:c1].reshape(-1)
                        aa = aa - aa.mean(); bb = bb - bb.mean()
                        denom = (aa.std()+1e-6)*(bb.std()+1e-6)
                        hm[bi,bj] = (aa*bb).mean()/denom
                hm = torch.nan_to_num(hm, nan=0.0)
                hm = torch.clamp(hm, -1.0, 1.0)
                hm_up = torch.nn.functional.interpolate(hm.unsqueeze(0).unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False).squeeze(0)
                hms.append(hm_up)
            if hms:
                heatmaps = torch.cat(hms, dim=0)

        # aux_vec
        base = x[:6]
        low_mean = low_e.mean(dim=(-2,-1))
        high_mean = high_e.mean(dim=(-2,-1))
        ratio_lh = low_mean / (high_mean + 1e-6)
        chan_energy = (base.pow(2).mean(dim=(-2,-1)))
        chan_energy_ratio = chan_energy / (chan_energy.sum()+1e-6)
        # 15個跨道相關
        flat = base.reshape(6,-1); flat = flat - flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True) + 1e-6
        flat_n = flat / std
        cc_vals = []
        for i in range(6):
            for j in range(i+1,6):
                cc_vals.append((flat_n[i]*flat_n[j]).mean())
        cc_vec = torch.stack(cc_vals) if len(cc_vals)>0 else torch.zeros(15)

        H,W = base.shape[-2], base.shape[-1]
        def _sl(rg):
            r0,r1 = rg; r0 = max(0,min(H,int(r0))); r1 = max(0,min(H,int(r1))); r1 = max(r1, r0+4)
            return slice(r0,r1)
        sl_low = _sl(rows_cfg['low']); sl_mid = _sl(rows_cfg['mid']); sl_high = _sl(rows_cfg['high'])
        abs_base = base.abs()
        ene_low = (abs_base[:, sl_low, :].pow(2)).sum(dim=(-2,-1))
        ene_mid = (abs_base[:, sl_mid, :].pow(2)).sum(dim=(-2,-1))
        ene_high = (abs_base[:, sl_high, :].pow(2)).sum(dim=(-2,-1))
        ratio_lh_band = (ene_low/(ene_high+1e-6)).mean()
        ratio_mh_band = (ene_mid/(ene_high+1e-6)).mean()
        def _extreme_sum(bslice, thr):
            band_abs = abs_base[:, bslice, :]
            mask = (band_abs > thr).float()
            return ( (band_abs * mask).sum(dim=(-2,-1)) ).mean()
        extreme_low = _extreme_sum(sl_low, float(thr_cfg['low']))
        extreme_high = _extreme_sum(sl_high, float(thr_cfg['high']))
        gx = torch.nn.functional.pad(base[:, :, 1:] - base[:, :, :-1], (1,0))
        gy = torch.nn.functional.pad(base[:, 1:, :] - base[:, :-1, :], (0,0,1,0))
        theta = torch.atan2(gy[:, sl_low, :], gx[:, sl_low, :] + 1e-6)
        c,s = torch.cos(theta), torch.sin(theta)
        R = torch.sqrt((c.mean(dim=(-2,-1))**2 + s.mean(dim=(-2,-1))**2))
        phase_consistency = R.mean()
        ext_feats = torch.stack([ratio_lh_band, ratio_mh_band, extreme_low, extreme_high, phase_consistency]).float()
        aux_vec = torch.cat([low_mean, high_mean, ratio_lh, chan_energy_ratio, cc_vec, ext_feats], dim=0).float()

        # stats_vec（與訓練/驗證期一致）
        x_reshaped = x.view(6, -1)
        mean = x_reshaped.mean(dim=-1)
        std = x_reshaped.std(dim=-1)
        energy = (x_reshaped ** 2).mean(dim=-1)
        x_pos = x_reshaped - x_reshaped.min(dim=-1, keepdim=True).values
        denom = x_pos.sum(dim=-1, keepdim=True) + 1e-6
        p = x_pos / denom
        entropy = -(p * (p + 1e-8).log()).sum(dim=-1)
        mu = x_reshaped.mean(dim=-1, keepdim=True)
        sigma = x_reshaped.std(dim=-1, keepdim=True) + 1e-6
        skewness = (((x_reshaped - mu) / sigma) ** 3).mean(dim=-1)
        kurtosis = (((x_reshaped - mu) / sigma) ** 4).mean(dim=-1)
        energy_map = x ** 2
        blocks_4 = torch.nn.functional.adaptive_avg_pool2d(energy_map, (4,4)).reshape(6, -1)
        blocks_8 = torch.nn.functional.adaptive_avg_pool2d(energy_map, (8,8)).reshape(6, -1)
        base_stats = torch.stack([energy, mean, std, entropy, skewness, kurtosis], dim=-1)  # (6,6)
        stats_vec = torch.cat([base_stats.reshape(-1), blocks_4.reshape(-1), blocks_8.reshape(-1)], dim=0).float()

        return (
            aug_channels.detach().cpu().numpy(),
            heatmaps.detach().cpu().numpy(),
            aux_vec.detach().cpu().numpy(),
            stats_vec.detach().cpu().numpy(),
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='optimized_patch_data')
    parser.add_argument('--split', type=str, default='train,valid')
    parser.add_argument('--pairs', type=str, default='0-2,1-4')
    parser.add_argument('--disable_heatmaps', action='store_true')
    args = parser.parse_args()

    root = Path(args.data_root)
    dir_train = root / 'train_spectrograms'
    dir_valid = root / 'valid_spectrograms'
    splits = []
    if 'train' in args.split:
        splits.append(('train', dir_train))
    if 'valid' in args.split or 'val' in args.split:
        splits.append(('valid', dir_valid))

    train_files = list_npz(dir_train)
    if len(train_files) == 0:
        print('未找到訓練集 npz，請確認路徑。')
        return
    rows_cfg = compute_band_rows(32)
    thr_cfg = fit_thresholds_from_train(train_files, rows_cfg)
    print(f'擬合閾值: low={thr_cfg["low"]:.3f}, high={thr_cfg["high"]:.3f}')

    # 互相關對
    pairs = []
    try:
        for token in args.pairs.split(','):
            a,b = token.split('-')
            pairs.append((int(a), int(b)))
    except Exception:
        pairs = [(0,2),(1,4)]

    for split_name, split_dir in splits:
        files = list_npz(split_dir)
        print(f'[{split_name}] 檔案數: {len(files)}')
        for fp in files:
            try:
                with np.load(fp) as d:
                    spec = d['spectrogram']
                if spec.shape != (6,32,32):
                    continue
                aug, hm, aux_vec, stats_vec = build_sidecar(spec, rows_cfg, thr_cfg, pairs, enable_heatmaps=(not args.disable_heatmaps))
                sidecar = Path(fp).with_name(Path(fp).stem + '_aux.npz')
                np.savez_compressed(sidecar, aug_channels=aug, heatmaps=hm, aux_vec=aux_vec, stats_vec=stats_vec)
                # 另外輸出到分類目錄：optimized_patch_data/precomputed/{split}/...
                root = Path(args.data_root)
                out_root = root / 'precomputed' / split_name
                (out_root / 'aug_channels').mkdir(parents=True, exist_ok=True)
                (out_root / 'heatmaps').mkdir(parents=True, exist_ok=True)
                (out_root / 'aux_vec').mkdir(parents=True, exist_ok=True)
                (out_root / 'stats_vec').mkdir(parents=True, exist_ok=True)
                stem = Path(fp).stem
                np.save(out_root / 'aug_channels' / f'{stem}_aug_channels.npy', aug)
                # 熱圖可能為空陣列，仍保存以保持結構一致
                np.save(out_root / 'heatmaps' / f'{stem}_heatmaps.npy', hm)
                np.save(out_root / 'aux_vec' / f'{stem}_aux_vec.npy', aux_vec)
                np.save(out_root / 'stats_vec' / f'{stem}_stats_vec.npy', stats_vec)
            except Exception as e:
                print(f'[{split_name}] 失敗: {fp}: {e}')

    print('✅ 預計算完成')

if __name__ == '__main__':
    main()


