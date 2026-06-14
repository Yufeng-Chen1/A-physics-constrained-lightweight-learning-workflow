#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6通道子圖譜數據加載器
專門處理6通道的32×32子圖譜數據
支持兩種來源：
1) 舊格式：optimized_patch_data/spectrograms/*.npz
2) 新格式：optimized_patch_data/train_spectrograms、valid_spectrograms、test_spectrograms
"""

import torch

import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import random
from typing import List

class SixChannelPatchDataset(Dataset):
    """6通道子圖譜數據集（可選擴充子頻帶能量通道 + 小波/2D-DWT向量 + 跨道互相關熱圖 + 擴展aux_vec）"""
    
    def __init__(self, file_paths, transform=None, augment=False, channel_mean=None, channel_std=None, add_subband_features: bool = False, add_wavelet_vector: bool = False, band_config: dict = None, enable_crosscorr_heatmap: bool = True, heatmap_pairs: list = None, prefer_precomputed_aux: bool = True, precomp_suffix: str = "_aux.npz"):
        self.file_paths = file_paths
        self.transform = transform
        self.augment = augment
        self.channel_mean = channel_mean
        self.channel_std = channel_std
        self.add_subband_features = bool(add_subband_features)
        self.add_wavelet_vector = bool(add_wavelet_vector)
        self.band_config = band_config or {}
        self.enable_crosscorr_heatmap = bool(enable_crosscorr_heatmap)
        # 指定更物理合理的配對：(AC,DEN)≈(3,5)、(RT,CNL)≈(2,4)、(GR,RT)≈(0,2)、(AC,CNL)≈(3,4)
        self.heatmap_pairs = heatmap_pairs if heatmap_pairs is not None else [(3, 5), (2, 4), (0, 2), (3, 4)]
        # 預計算配置
        self.prefer_precomputed_aux = bool(prefer_precomputed_aux)
        self.precomp_suffix = precomp_suffix
        # 僅首批打印一次預計算使用狀態，避免刷屏
        self._logged_precomp = False
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 加載數據
        data = np.load(file_path, mmap_mode='r', allow_pickle=False)
        spectrogram = data['spectrogram']  # (6, 32, 32)
        # 數值清理：處理 NaN/Inf，防止下游產生 NaN 損失
        try:
            spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        label = int(data['label'])
        
        # 數據增強
        if self.augment and random.random() < 0.3:
            spectrogram = self._augment_spectrogram(spectrogram)
        
        # 轉換為張量
        spectrogram_tensor = torch.as_tensor(spectrogram, dtype=torch.float32).clone()
        # 再次保險：張量級別清理與裁剪，避免極端值
        with torch.no_grad():
            spectrogram_tensor = torch.nan_to_num(spectrogram_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            spectrogram_tensor = torch.clamp(spectrogram_tensor, min=-10.0, max=10.0)
        # 數據集級標準化（per-channel） - 先對原始6通道做歸一化，再派生子頻帶能量
        if self.channel_mean is not None and self.channel_std is not None:
            try:
                cm = torch.as_tensor(self.channel_mean).view(-1, 1, 1).type_as(spectrogram_tensor)
                cs = torch.as_tensor(self.channel_std).view(-1, 1, 1).type_as(spectrogram_tensor)
                spectrogram_tensor = (spectrogram_tensor - cm) / (cs + 1e-6)
            except Exception:
                pass
        # 正規化後再做一次裁剪，防爆值
        with torch.no_grad():
            spectrogram_tensor = torch.clamp(spectrogram_tensor, min=-8.0, max=8.0)
        
        # 子頻帶能量擴充（每通道：低頻能量/高頻能量）+ 擴展aux_vec（頻帶比/敏感頻段極值/相位一致性）+ 跨道互相關熱圖
        if self.add_subband_features:
            # 1) 盡量優先讀取預計算副檔（與主npz同名+_aux.npz）
            aux_vec = None
            if self.prefer_precomputed_aux:
                try:
                    p = Path(str(file_path))
                    sidecar = p.with_name(p.stem + self.precomp_suffix)
                    # 新：嘗試從分類目錄讀取各項（若存在）
                    try:
                        split_dir = p.parent.name  # train_spectrograms 或 valid_spectrograms
                        pre_root = p.parent.parent / 'precomputed' / ('train' if 'train' in split_dir else 'valid')
                        stem = p.stem
                        aug_fp = pre_root / 'aug_channels' / f'{stem}_aug_channels.npy'
                        aux_fp = pre_root / 'aux_vec' / f'{stem}_aux_vec.npy'
                        stats_fp = pre_root / 'stats_vec' / f'{stem}_stats_vec.npy'
                        heat_fp = pre_root / 'heatmaps' / f'{stem}_heatmaps.npy'
                        wpt_fp = pre_root / 'wpt_vec' / f'{stem}_wpt_vec.npy'
                        if aug_fp.exists():
                            aug_np = np.load(aug_fp, mmap_mode='r')
                            aug_np = np.array(aug_np, copy=True)
                            aug = torch.from_numpy(aug_np).float()
                            aug = torch.nan_to_num(aug, nan=0.0, posinf=0.0, neginf=0.0)
                            aug = torch.clamp(aug, min=-6.0, max=6.0)
                            spectrogram_tensor = torch.cat([spectrogram_tensor, aug], dim=0)
                        if os.path.exists(str(heat_fp)):
                            hm_np = np.load(heat_fp, mmap_mode='r')
                            if hm_np.size > 0:
                                hm_np = np.array(hm_np, copy=True)
                                hm = torch.from_numpy(hm_np).float()
                                hm = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
                                hm = torch.clamp(hm, min=-1.0, max=1.0)
                                if hm.ndim == 3:
                                    spectrogram_tensor = torch.cat([spectrogram_tensor, hm], dim=0)
                        if aux_fp.exists():
                            aux_vec_np = np.load(aux_fp, mmap_mode='r').astype(np.float32)
                            aux_vec = torch.as_tensor(np.nan_to_num(aux_vec_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clone()
                        if stats_fp.exists():
                            stats_np = np.load(stats_fp, mmap_mode='r').astype(np.float32)
                            metadata_stats = torch.as_tensor(np.nan_to_num(stats_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clone()
                        if wpt_fp.exists():
                            wpt_np = np.load(wpt_fp, mmap_mode='r')
                            if wpt_np.size > 0:
                                wpt_np = np.array(wpt_np, copy=True)
                                _wpt = torch.from_numpy(wpt_np).float()
                                _wpt = torch.nan_to_num(_wpt, nan=0.0, posinf=0.0, neginf=0.0)
                                _wpt = torch.clamp(_wpt, min=-10.0, max=10.0)
                                metadata_wpt = _wpt.clone()
                    except Exception:
                        pass
                    if sidecar.exists():
                        sc = np.load(sidecar, allow_pickle=False, mmap_mode='r')
                        # aug_channels: (M,H,W) 例如低/高頻能量與熱圖等
                        if 'aug_channels' in sc:
                            _aug_np = np.array(sc['aug_channels'], copy=True)
                            aug = torch.from_numpy(_aug_np).float()
                            aug = torch.nan_to_num(aug, nan=0.0, posinf=0.0, neginf=0.0)
                            aug = torch.clamp(aug, min=-6.0, max=6.0)
                            spectrogram_tensor = torch.cat([spectrogram_tensor, aug], dim=0)
                        # 單獨熱圖
                        if 'heatmaps' in sc:
                            _hm_np = np.array(sc['heatmaps'], copy=True)
                            hm = torch.from_numpy(_hm_np).float()
                            hm = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
                            hm = torch.clamp(hm, min=-1.0, max=1.0)
                            if hm.ndim == 3:
                                spectrogram_tensor = torch.cat([spectrogram_tensor, hm], dim=0)
                        # 向量特徵
                        if 'aux_vec' in sc:
                            aux_vec_np = sc['aux_vec'].astype(np.float32)
                            aux_vec = torch.as_tensor(np.nan_to_num(aux_vec_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clone()
                        # 統計向量（離線計算，訓練/驗證時直接使用）
                        if 'stats_vec' in sc:
                            stats_np = sc['stats_vec'].astype(np.float32)
                            metadata_stats = torch.as_tensor(np.nan_to_num(stats_np, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32).clone()
                            # 保存到metadata，供訓練/驗證直接使用
                            # 先記錄，待metadata構造後再掛載
                        else:
                            metadata_stats = None
                        sc.close()
                        # 將離線 stats 暫存到本地變量，稍後寫入 metadata
                        if 'metadata_stats' in locals() and metadata_stats is not None:
                            pass
                except Exception:
                    aux_vec = None
            # 不輸出預計算通道詳情到訓練日誌
            try:
                # 僅在未使用離線預計算時，才在線生成子頻帶增廣與輔助向量
                if not self.prefer_precomputed_aux:
                    import torch.nn.functional as F
                    x = spectrogram_tensor.unsqueeze(0)  # (1,6,H,W)
                    # 盒式平滑近似低頻（k=3）
                    low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                    high = x - low
                    low_e = low.pow(2)
                    high_e = high.pow(2)
                    # 簡單尺度自適應：用均方能量做分母，數值更穩定
                    with torch.no_grad():
                        denom = x.pow(2).mean(dim=(-2, -1), keepdim=True) + 1e-3
                    low_e = low_e / denom
                    high_e = high_e / denom
                    aug = torch.cat([low_e, high_e], dim=1).squeeze(0)  # (12,H,W)
                    # 清理附加通道的極端與NaN
                    aug = torch.nan_to_num(aug, nan=0.0, posinf=0.0, neginf=0.0)
                    aug = torch.clamp(aug, min=-6.0, max=6.0)
                    # 若已經從預計算載入aug_channels，避免重複疊加
                    if spectrogram_tensor.shape[0] <= 6:
                        spectrogram_tensor = torch.cat([spectrogram_tensor, aug], dim=0)  # (>=18,H,W)
                # 生成輔助向量 aux_vec：頻帶比值與跨道關聯（僅在需要時）
                try:
                    # 每通道低/高頻能量的全局均值
                    if 'low_e' in locals() and 'high_e' in locals():
                        low_mean = low_e.mean(dim=(-2, -1)).squeeze(0)  # (6,)
                        high_mean = high_e.mean(dim=(-2, -1)).squeeze(0)  # (6,)
                    else:
                        low_mean = high_mean = None
                    ratio_lh = low_mean / (high_mean + 1e-6)
                    # 通道總能量佔比（原始6通道）
                    base = x.squeeze(0)[:6, :, :]
                    chan_energy = (base.pow(2).mean(dim=(-2, -1)))  # (6,)
                    chan_energy_ratio = chan_energy / (chan_energy.sum() + 1e-6)
                    # 跨道相關（6通道上三角 15 個相關係數）
                    flat = base.reshape(6, -1)
                    flat = flat - flat.mean(dim=1, keepdim=True)
                    std = flat.std(dim=1, keepdim=True) + 1e-6
                    flat_n = flat / std
                    cc_vals = []
                    for i in range(6):
                        for j in range(i + 1, 6):
                            cc = (flat_n[i] * flat_n[j]).mean()
                            cc_vals.append(cc)
                    cc_vec = torch.stack(cc_vals) if len(cc_vals) > 0 else torch.zeros(15, device=base.device)

                    # 擴展：敏感頻帶能量比與極值能量、相位一致性（代理）
                    H = base.shape[-2]
                    W = base.shape[-1]
                    # band rows（按行索引）
                    bands = self.band_config.get('rows', {'low': (4, 10), 'mid': (10, 20), 'high': (20, 28)})
                    # 能量（幅值平方）
                    abs_base = base.abs()
                    def _band_slice(bkey):
                        r0, r1 = bands.get(bkey, (0, H))
                        r0 = max(0, min(H, int(r0)))
                        r1 = max(0, min(H, int(r1)))
                        if r1 <= r0:
                            r1 = min(H, r0 + 4)
                        return slice(r0, r1)
                    sl_low = _band_slice('low')
                    sl_mid = _band_slice('mid')
                    sl_high = _band_slice('high')
                    ene_low = (abs_base[:, sl_low, :].pow(2)).sum(dim=(-2, -1))  # (6,)
                    ene_mid = (abs_base[:, sl_mid, :].pow(2)).sum(dim=(-2, -1))  # (6,)
                    ene_high = (abs_base[:, sl_high, :].pow(2)).sum(dim=(-2, -1))  # (6,)
                    # 頻帶能量比（聚合到標量：對6通道取平均）
                    ratio_lh_band = (ene_low / (ene_high + 1e-6)).mean()
                    ratio_mh_band = (ene_mid / (ene_high + 1e-6)).mean()
                    # 極值能量和（敏感頻段）：基於訓練擬合的閾值，這裡使用提供的閾值；若無則用分位數代理
                    thr_cfg = self.band_config.get('threshold', {})
                    thr_low = float(thr_cfg.get('low', 1.5))
                    thr_high = float(thr_cfg.get('high', 1.5))
                    # 對每通道在band內累加超閾值能量（簡化為對幅值做閾值）
                    def _extreme_sum(bslice, thr):
                        band_abs = abs_base[:, bslice, :]
                        mask = (band_abs > thr).float()
                        return ( (band_abs * mask).sum(dim=(-2, -1)) ).mean()  # 聚合到標量
                    extreme_low = _extreme_sum(sl_low, thr_low)
                    extreme_high = _extreme_sum(sl_high, thr_high)
                    # 相位一致性代理：使用梯度方向的圓方差（越小越一致），這裡取其倒數作為一致性
                    gx = torch.nn.functional.pad(base[:, :, 1:] - base[:, :, :-1], (1,0))
                    gy = torch.nn.functional.pad(base[:, 1:, :] - base[:, :-1, :], (0,0,1,0))
                    theta = torch.atan2(gy[:, sl_low, :], gx[:, sl_low, :] + 1e-6)  # 低頻帶
                    c, s = torch.cos(theta), torch.sin(theta)
                    R = torch.sqrt((c.mean(dim=(-2, -1))**2 + s.mean(dim=(-2, -1))**2))  # (6,)
                    phase_consistency = R.mean()  # 0~1，越大越一致

                    ext_feats = torch.stack([
                        ratio_lh_band, ratio_mh_band, extreme_low, extreme_high, phase_consistency
                    ]).detach().cpu().float()
                    aux_vec = torch.cat([
                        low_mean, high_mean, ratio_lh, chan_energy_ratio, cc_vec, ext_feats
                    ], dim=0).detach().cpu().float()
                except Exception:
                    aux_vec = None

                # 跨道互相關熱圖（8×8 blockwise 皮爾森相關）
                if self.enable_crosscorr_heatmap and len(self.heatmap_pairs) > 0 and (not self.prefer_precomputed_aux):
                    try:
                        maps = []
                        for (i, j) in self.heatmap_pairs:
                            i = int(max(0, min(5, i)))
                            j = int(max(0, min(5, j)))
                            ch_i = spectrogram_tensor[i]
                            ch_j = spectrogram_tensor[j]
                            # 切成 8×8 區塊，每塊 4×4，計算每塊皮爾森相關
                            hm = torch.zeros(8, 8, device=spectrogram_tensor.device, dtype=spectrogram_tensor.dtype)
                            for bi in range(8):
                                for bj in range(8):
                                    r0, r1 = bi*4, bi*4+4
                                    c0, c1 = bj*4, bj*4+4
                                    a = ch_i[r0:r1, c0:c1].reshape(-1)
                                    b = ch_j[r0:r1, c0:c1].reshape(-1)
                                    a = a - a.mean()
                                    b = b - b.mean()
                                    denom = (a.std()+1e-6)*(b.std()+1e-6)
                                    hm[bi, bj] = (a*b).mean()/denom
                            # 夾取穩定範圍
                            hm = torch.nan_to_num(hm, nan=0.0)
                            hm = torch.clamp(hm, min=-1.0, max=1.0)
                            maps.append(hm.unsqueeze(0))
                        if maps:
                            heatmaps = torch.cat(maps, dim=0)  # (num_pairs,8,8)
                            # 上採樣到 32×32 與主圖大小一致（或直接雙線性插值）
                            heatmaps_up = torch.nn.functional.interpolate(heatmaps.unsqueeze(0), size=(32,32), mode='bilinear', align_corners=False).squeeze(0)
                            spectrogram_tensor = torch.cat([spectrogram_tensor, heatmaps_up], dim=0)
                    except Exception:
                        pass
            except Exception:
                # 失敗則退回原6通道
                aux_vec = None
        label_tensor = torch.LongTensor([label])[0]
        
        # 元數據 - 避免字符串類型問題
        metadata = {
            'sample_idx': int(data.get('sample_idx', 0)),
        }
        # 小波/2D-DWT 向量（優先離線預計算；僅在未啟用或缺失時才在線計算）
        if self.add_wavelet_vector:
            try:
                if self.prefer_precomputed_aux and 'metadata_wpt' in locals() and metadata_wpt is not None:
                    metadata['wpt_vec'] = metadata_wpt
                else:
                    import pywt
                    ch = spectrogram_tensor[:6, :, :].detach().cpu().numpy()
                    feats = []
                    for c in range(ch.shape[0]):
                        arr = ch[c]
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
                    metadata['wpt_vec'] = torch.as_tensor(wpt_vec, dtype=torch.float32).clone()
            except Exception:
                pass
        # 附加：專屬可分性輔助向量（頻帶比值/跨道相關）
        try:
            if 'aux_vec' not in metadata and 'aux_vec' in locals() and aux_vec is not None:
                # 安全處理數值
                av = torch.nan_to_num(aux_vec, nan=0.0, posinf=0.0, neginf=0.0)
                av = torch.clamp(av, min=-10.0, max=10.0)
                metadata['aux_vec'] = av
            # 離線統計向量（若存在）
            if 'metadata_stats' in locals() and metadata_stats is not None:
                metadata['stats_vec'] = torch.nan_to_num(metadata_stats, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass
        
        return spectrogram_tensor, label_tensor, metadata
    
    def _augment_spectrogram(self, spectrogram):
        """數據增強"""
        # 隨機水平翻轉
        if random.random() < 0.5:
            spectrogram = np.flip(spectrogram, axis=-1).copy()
        
        # 添加輕微噪聲
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, spectrogram.shape)
            spectrogram = spectrogram + noise
        
        # 隨機縮放
        if random.random() < 0.2:
            scale = random.uniform(0.95, 1.05)
            spectrogram = spectrogram * scale
        
        return spectrogram


def create_six_channel_data_loaders(cache_dir="optimized_patch_data",
                                  batch_size=64,
                                  train_ratio=0.8,
                                  val_ratio=0.2,
                                  test_ratio=0.0,
                                  random_state=42,
                                  num_workers=0,
                                  augment_train=True,
                                  use_weighted_sampler: bool = False,
                                  upsample_class4_factor: int = 1,
                                  add_subband_features: bool = False,
                                  add_wavelet_vector: bool = False,
                                  ensure_minority_per_batch: bool = False,
                                  min_c3_per_batch: int = 2,
                                  min_c4_per_batch: int = 2,
                                  min_c1_per_batch: int = 0,
                                  prefer_precomputed_aux: bool = True,
                                  use_meta_index: bool = True):
    """創建6通道數據加載器"""
    print("創建6通道子圖譜數據加載器")
    print("=" * 60)
    
    base_dir = Path(cache_dir)
    dir_train = base_dir / "train_spectrograms"
    dir_val = base_dir / "valid_spectrograms"
    dir_test = base_dir / "test_spectrograms"
    spectrograms_dir = base_dir / "spectrograms"
    
    use_presplit = dir_train.exists() and dir_val.exists()
    
    def _glob_many(d: Path, patterns: List[str]):
        files: List[Path] = []
        for pat in patterns:
            files.extend(list(d.glob(pat)))
        # 去重：不同通配符可能匹配到同一文件，這裡做唯一化
        try:
            files = sorted(set(files), key=lambda p: str(p))
        except Exception:
            # 退化處理：用字符串路徑去重
            uniq = {}
            for p in files:
                uniq[str(p)] = p
            files = list(uniq.values())
        return files
    
    patterns = [
        "*_patch_6ch_*.npz",
        "*_patch_6ch_32x32_*_*.npz",
    ]
    
    if use_presplit:
        if use_meta_index:
            # 優先嘗試使用 meta 索引與通道統計
            try:
                import json as _json
                meta_dir = base_dir / 'meta'
                meta_fp = meta_dir / 'dataset_stats.json'
                # 通道統計
                channel_mean = None
                channel_std = None
                if meta_fp.exists():
                    with open(meta_fp, 'r', encoding='utf-8') as _f:
                        meta_obj = _json.load(_f)
                    cm = meta_obj.get('channel_mean')
                    cs = meta_obj.get('channel_std')
                    if isinstance(cm, list) and isinstance(cs, list) and len(cm) == 6 and len(cs) == 6:
                        channel_mean = np.asarray(cm, dtype=np.float32)
                        channel_std = np.asarray(cs, dtype=np.float32)
                # 現場仍需列舉檔名，但不再做深度掃描或重算
                train_files = _glob_many(dir_train, patterns)
                val_files = _glob_many(dir_val, patterns)
                test_files = _glob_many(dir_test, patterns) if dir_test.exists() else []
            except Exception:
                channel_mean = channel_std = None
                train_files = _glob_many(dir_train, patterns)
                val_files = _glob_many(dir_val, patterns)
                test_files = _glob_many(dir_test, patterns) if dir_test.exists() else []
        else:
            train_files = _glob_many(dir_train, patterns)
            val_files = _glob_many(dir_val, patterns)
            test_files = _glob_many(dir_test, patterns) if dir_test.exists() else []
        print(f"   採用預先劃分目錄: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    else:
        # 舊格式：統一從 spectrograms 搜索
        patch_files = _glob_many(spectrograms_dir, patterns)
        print(f"   從舊格式目錄讀取: {len(patch_files)} 個6通道子圖譜文件")
        if len(patch_files) == 0:
            raise ValueError("沒有找到6通道子圖譜文件！請先生成數據或檢查路徑")
    
    # 2. 收集標籤並校驗格式
    def _validate_and_collect(files: List[Path]):
        labels_local = []
        valid_local: List[Path] = []
        for file_path in files:
            try:
                with np.load(file_path) as data:
                    label = int(data.get('label', -1))
                    spec = data['spectrogram']
                if spec.shape == (6, 32, 32) and label >= 0:
                    labels_local.append(label)
                    valid_local.append(file_path)
            except Exception:
                continue
        return valid_local, labels_local
    
    if use_presplit:
        train_files, train_labels = _validate_and_collect(train_files)
        val_files, val_labels = _validate_and_collect(val_files)
        test_files, test_labels = _validate_and_collect(test_files) if dir_test.exists() else ([], [])
        
        # ❌ 禁用自动井级平衡过滤（已验证会丢失77%训练数据）
        # 【井级平衡】如果存在平衡后的文件列表，使用它过滤训练文件
        # balanced_list = base_dir / "balanced_train_files.txt"
        # if balanced_list.exists():
        #     with open(balanced_list, 'r', encoding='utf-8') as f:
        #         balanced_paths = set(Path(line.strip()) for line in f if line.strip())
        #     
        #     # 过滤出平衡后的文件
        #     filtered_train = [(f, l) for f, l in zip(train_files, train_labels) if f in balanced_paths]
        #     if filtered_train:
        #         train_files, train_labels = zip(*filtered_train)
        #         train_files, train_labels = list(train_files), list(train_labels)
        #         print(f"   使用井级平衡: {len(train_files)} 个训练样本 (过滤后)")
        #     else:
        #         print(f"   警告: 平衡列表为空或不匹配，使用原始训练集")
    else:
        valid_files, labels = _validate_and_collect(patch_files)
        print(f"   有效文件: {len(valid_files)} 個")
    
    # 3. 檢查類別分布
    label_dist = Counter(train_labels if use_presplit else labels)
    class_names = {0: "油層", 1: "水層", 2: "干層", 3: "差油層", 4: "油水層"}
    
    print(f"\n📊 類別分布:")
    total = len(train_labels if use_presplit else labels)
    for class_id in range(5):
        count = label_dist.get(class_id, 0)
        percentage = count / total * 100 if total > 0 else 0
        print(f"   {class_names[class_id]}: {count} ({percentage:.1f}%)")
    
    # 4. 分層抽樣分割數據（僅舊格式需要）
    print(f"\n📊 數據分割 (訓練:{train_ratio}, 驗證:{val_ratio}, 測試:{test_ratio})")
    
    if not use_presplit:
        # 第一次分割：分出測試集
        if test_ratio > 0:
            train_val_files, test_files, train_val_labels, test_labels = train_test_split(
                valid_files, labels,
                test_size=test_ratio,
                stratify=labels,
                random_state=random_state
            )
        else:
            train_val_files, train_val_labels = valid_files, labels
            test_files, test_labels = [], []
        
        # 第二次分割：分出訓練集和驗證集
        adjusted_train_ratio = train_ratio / (train_ratio + val_ratio)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files, train_val_labels,
            train_size=adjusted_train_ratio,
            stratify=train_val_labels,
            random_state=random_state
        )
    
    print(f"   訓練集: {len(train_files)} 個樣本")
    print(f"   驗證集: {len(val_files)} 個樣本")
    if (use_presplit and len(test_files) > 0) or (not use_presplit and test_ratio > 0):
        print(f"   測試集: {len(test_files)} 個樣本")
    
    # 5. 檢查訓練集類別分布
    train_label_dist = Counter(train_labels)
    print(f"\n📊 訓練集類別分布:")
    for class_id in range(5):
        count = train_label_dist.get(class_id, 0)
        percentage = count / len(train_labels) * 100 if train_labels else 0
        print(f"   {class_names[class_id]}: {count} ({percentage:.1f}%)")
    
    # 6. 可選：對少數類（油水層=4）做溫和上採樣（複製文件路徑）
    if upsample_class4_factor and upsample_class4_factor > 1:
        new_train_files: List[Path] = []
        new_train_labels: List[int] = []
        for fp, lb in zip(train_files, train_labels):
            new_train_files.append(fp)
            new_train_labels.append(lb)
            if lb == 4:
                # 輕度上採樣，不生成新樣本，僅提升抽樣概率
                for _ in range(upsample_class4_factor - 1):
                    new_train_files.append(fp)
                    new_train_labels.append(lb)
        print(f"\n   少數類上採樣: class=4 ×{upsample_class4_factor} -> {sum(1 for x in new_train_labels if x==4)} 個")
        train_files, train_labels = new_train_files, new_train_labels

    # 7. 創建數據集
    # 6.1 計算數據集級通道統計（mean/std）
    def _compute_channel_stats(files: List[Path]):
        if not files:
            return None, None
        import numpy as _np
        sum_c = _np.zeros((6,), dtype=_np.float64)
        sumsq_c = _np.zeros((6,), dtype=_np.float64)
        count = 0
        for fp in files:
            try:
                with _np.load(fp) as d:
                    spec = d['spectrogram']  # (6, H, W)
                if spec.shape[0] != 6:
                    continue
                ch_flat = spec.reshape(6, -1)
                sum_c += ch_flat.sum(axis=1)
                sumsq_c += (ch_flat ** 2).sum(axis=1)
                count += ch_flat.shape[1]
            except Exception:
                continue
        if count == 0:
            return None, None
        mean = sum_c / count
        var = _np.maximum(sumsq_c / count - mean ** 2, 1e-8)
        std = _np.sqrt(var)
        return mean.astype(_np.float32), std.astype(_np.float32)

    # 對原始6通道估計統計量；子頻帶由原通道歸一化後派生
    if use_presplit and use_meta_index and 'channel_mean' in locals() and channel_mean is not None:
        ch_mean, ch_std = channel_mean, channel_std
        try:
            import numpy as _np
            print(f"   使用meta緩存通道統計: mean={_np.round(ch_mean,4)}, std={_np.round(ch_std,4)}")
        except Exception:
            pass
    else:
        ch_mean, ch_std = _compute_channel_stats(train_files)
    if ch_mean is not None:
        import numpy as _np
        print(f"   數據集級標準化: mean={_np.round(ch_mean,4)}, std={_np.round(ch_std,4)}")

    band_cfg = {
        'rows': {'low': (4, 10), 'mid': (10, 20), 'high': (20, 28)},
        'threshold': {'low': 1.5, 'high': 1.5}
    }
    train_dataset = SixChannelPatchDataset(
        train_files,
        augment=augment_train,
        channel_mean=ch_mean,
        channel_std=ch_std,
        add_subband_features=add_subband_features,
        add_wavelet_vector=add_wavelet_vector,
        band_config=band_cfg,
        enable_crosscorr_heatmap=True,
        prefer_precomputed_aux=prefer_precomputed_aux,
    )
    val_dataset = SixChannelPatchDataset(
        val_files,
        augment=False,
        channel_mean=ch_mean,
        channel_std=ch_std,
        add_subband_features=add_subband_features,
        add_wavelet_vector=add_wavelet_vector,
        band_config=band_cfg,
        enable_crosscorr_heatmap=True,
        prefer_precomputed_aux=prefer_precomputed_aux,
    )
    
    # 7. 創建數據加載器（可選權重採樣/批內保證）
    sampler = None
    batch_sampler = None
    if use_weighted_sampler:
        try:
            from torch.utils.data import WeightedRandomSampler
            counts = Counter(train_labels)
            num_samples = len(train_files)
            if num_samples > 0 and len(counts) > 0:
                class_weights = {c: (num_samples / (len(counts) * cnt)) for c, cnt in counts.items() if cnt > 0}
                weights = [class_weights[int(lbl)] for lbl in train_labels]
                sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        except Exception:
            sampler = None

    # 自定義批採樣器：保證每個 batch 至少包含指定數量的類3/4 樣本
    if not use_weighted_sampler and ensure_minority_per_batch:
        try:
            from torch.utils.data import Sampler
            import math as _math
            indices_by_class = {i: [] for i in range(5)}
            for idx, lb in enumerate(train_labels):
                if 0 <= int(lb) < 5:
                    indices_by_class[int(lb)].append(idx)
            # 打散
            for k in indices_by_class:
                random.shuffle(indices_by_class[k])
            # 構建批次索引列表
            total = len(train_labels)
            num_batches = total // batch_size
            all_indices = list(range(len(train_labels)))
            random.shuffle(all_indices)
            cursor_all = 0
            cursor_c1 = 0
            cursor_c3 = 0
            cursor_c4 = 0
            c3_pool = indices_by_class.get(3, [])
            c4_pool = indices_by_class.get(4, [])
            c1_pool = indices_by_class.get(1, [])
            def _take(pool, cursor, k):
                taken = []
                if len(pool) == 0 or k <= 0:
                    return taken, cursor
                for _ in range(k):
                    if cursor >= len(pool):
                        cursor = 0
                        random.shuffle(pool)
                        if len(pool) == 0:
                            break
                    taken.append(pool[cursor])
                    cursor += 1
                return taken, cursor
            batches = []
            used = set()
            for _ in range(num_batches):
                current = []
                need1 = max(0, int(min_c1_per_batch))
                need3 = max(0, int(min_c3_per_batch))
                need4 = max(0, int(min_c4_per_batch))
                t1, cursor_c1 = _take(c1_pool, cursor_c1, need1)
                t3, cursor_c3 = _take(c3_pool, cursor_c3, need3)
                t4, cursor_c4 = _take(c4_pool, cursor_c4, need4)
                current.extend(t1)
                current.extend(t3)
                current.extend(t4)
                # 補齊
                while len(current) < batch_size and cursor_all < len(all_indices):
                    cand = all_indices[cursor_all]
                    cursor_all += 1
                    if cand in used:
                        continue
                    current.append(cand)
                # 若還不足，從整體隨機補
                while len(current) < batch_size:
                    cand = random.randrange(0, len(train_labels))
                    if cand in used:
                        continue
                    current.append(cand)
                for ii in current:
                    used.add(ii)
                batches.append(current[:batch_size])
            class _BalancedBatchSampler(Sampler[list]):
                def __init__(self, batches):
                    self._batches = batches
                def __iter__(self):
                    random.shuffle(self._batches)
                    for b in self._batches:
                        yield b
                def __len__(self):
                    return len(self._batches)
            batch_sampler = _BalancedBatchSampler(batches)
        except Exception:
            batch_sampler = None

    _dl_kwargs = {}
    if num_workers and num_workers > 0:
        _dl_kwargs.update({'persistent_workers': True})
        try:
            _dl_kwargs.update({'prefetch_factor': 2})
        except Exception:
            pass

    if batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            **_dl_kwargs
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **_dl_kwargs
    )
    
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if (use_presplit and len(test_files) > 0) or (not use_presplit and test_ratio > 0):
        test_dataset = SixChannelPatchDataset(test_files, augment=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            **_dl_kwargs
        )
        data_loaders['test'] = test_loader
    
    print(f"\n6通道數據加載器創建完成")
    
    # 8. 驗證數據格式
    print(f"\n驗證數據格式...")
    sample_batch = next(iter(train_loader))
    spectrograms, labels, metadata = sample_batch
    print(f"   批次形狀: {spectrograms.shape}")
    print(f"   標籤形狀: {labels.shape}")
    print(f"   通道數: {spectrograms.shape[1]}")
    print(f"   圖譜尺寸: {spectrograms.shape[2]}×{spectrograms.shape[3]}")
    # 允許 6 或擴展後的多通道（熱圖/aug加入後可能>18）
    if (spectrograms.shape[2:] == (32, 32)) and (spectrograms.shape[1] >= 6):
        print(f"   子圖譜格式正確")
    else:
        print(f"   格式不正確")
    
    return data_loaders
