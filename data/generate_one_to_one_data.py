#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的一对一数据生成方法
流程：
1. 先对整条井的测井曲线生成时频图谱序列（滑动窗口，与深度对齐）
2. 再根据流体标注点，一对一采样（深度索引对应时频图谱索引）
3. 8通道时频图谱（6曲线+2增强）+ 57维辅助特征
4. 多频带分解x3（低/中/高频）
5. 智能少数类增强（<5%增强4x, <10%增强2x, <15%增强1x）
6. 增强样本额外标注
"""

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy.signal import find_peaks
import json
import warnings
warnings.filterwarnings('ignore')

# ============ 配置参数 ============
WINDOW_SIZE = 128  # 滑动窗口大小
STRIDE = 1  # 滑动步长（1表示每个深度点都生成时频图谱）

# 测井曲线配置
CURVE_CONFIGS = {
    # 高频瞬态型：CWT + Morlet
    'ac': {'method': 'cwt', 'wavelet': 'morl', 'scales': np.arange(1, 33), 'size': (32, 32)},
    'resistivity': {'method': 'cwt', 'wavelet': 'morl', 'scales': np.arange(1, 33), 'size': (32, 32)},
    
    # 低频趋势型：WPT + Sym8
    'grd': {'method': 'wpt', 'wavelet': 'sym8', 'level': 4, 'size': (64, 64)},
    'sp': {'method': 'wpt', 'wavelet': 'sym8', 'level': 4, 'size': (64, 64)},
    'den': {'method': 'wpt', 'wavelet': 'sym8', 'level': 4, 'size': (64, 64)},
    'cnl': {'method': 'wpt', 'wavelet': 'sym8', 'level': 4, 'size': (64, 64)}
}

UNIFIED_SIZE = (64, 64)

# 曲线别名映射
CURVE_ALIASES = {
    'ac': ['ac', 'AC', 'DT', 'dt', '声波时差', '声波'],
    'cnl': ['cnl', 'CNL', 'NPHI', 'nphi', '中子孔隙度', '中子'],
    'den': ['den', 'DEN', 'RHOB', 'rhob', '密度'],
    'grd': ['grd', 'GRD', 'GR', 'gr', '自然伽马', '伽马'],
    'resistivity': ['resistivity', 'RT', 'rt', 'RD', 'rd', '电阻率', '深电阻率'],
    'sp': ['sp', 'SP', '自然电位']
}

# 流体映射
FLUID_MAP = {
    '油层': 0, '致密油层': 0,
    '水层': 1,
    '干层': 2,
    '差油层': 3,
    '油水层': 4, '含油水层': 4, '油水同层': 4
}

FLUID_NAMES = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}

# 井划分
TEST_WELLS = ['姬117', '姬121', '耿86']

# 输出目录
OUTPUT_DIR = Path('one_to_one_data')

# ============ 工具函数 ============

def find_curve_column(df, curve_type):
    """查找曲线列名"""
    aliases = CURVE_ALIASES.get(curve_type, [])
    for alias in aliases:
        for col in df.columns:
            if alias.lower() in col.lower():
                return col
    return None

def read_well_data(file_path):
    """读取井数据"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for encoding in encodings:
        try:
            for sep in ['\t', None, r'\s+']:
                try:
                    if sep is None:
                        df = pd.read_csv(file_path, encoding=encoding)
                    else:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                    
                    if '深度' in df.columns and '解释结论' in df.columns:
                        return df
                except:
                    continue
        except:
            continue
    return None

def generate_cwt_spectrogram(signal, scales, wavelet='morl', target_size=(32, 32)):
    """生成CWT连续小波时频图谱"""
    try:
        coefficients, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1.0)
        spec = np.abs(coefficients).astype(np.float32)
        
        if spec.max() > 0:
            spec = (spec - spec.min()) / (spec.max() - spec.min())
        
        if spec.shape != target_size:
            spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            spec = F.interpolate(spec_tensor, size=target_size, mode='bilinear', align_corners=False)
            spec = spec.squeeze().numpy()
        
        return spec
    except Exception as e:
        return np.zeros(target_size, dtype=np.float32)

def generate_wpt_spectrogram(signal, wavelet, level, target_size):
    """生成WPT小波包时频图谱"""
    try:
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'natural')]
        
        coeffs_list = []
        for node in nodes:
            coeffs = wp[node].data
            if len(coeffs) > 0:
                coeffs_list.append(coeffs)
        
        if not coeffs_list:
            return np.zeros(target_size, dtype=np.float32)
        
        max_len = max(len(c) for c in coeffs_list)
        spec = np.zeros((len(coeffs_list), max_len), dtype=np.float32)
        for i, coeffs in enumerate(coeffs_list):
            spec[i, :len(coeffs)] = np.abs(coeffs)
        
        if spec.max() > 0:
            spec = (spec - spec.min()) / (spec.max() - spec.min())
        
        if spec.shape != target_size:
            spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            spec = F.interpolate(spec_tensor, size=target_size, mode='bilinear', align_corners=False)
            spec = spec.squeeze().numpy()
        
        return spec.astype(np.float32)
    except Exception as e:
        return np.zeros(target_size, dtype=np.float32)

def extract_6channel_spectrograms(window_df):
    """提取6通道时频图谱"""
    spectrograms = []
    
    for curve_type, config in CURVE_CONFIGS.items():
        col_name = find_curve_column(window_df, curve_type)
        
        if col_name is None or col_name not in window_df.columns:
            spectrograms.append(np.zeros(UNIFIED_SIZE, dtype=np.float32))
            continue
        
        signal = window_df[col_name].values
        
        if np.isnan(signal).any():
            signal = pd.Series(signal).interpolate(method='linear', limit_direction='both').fillna(0).values
        
        method = config['method']
        
        if method == 'cwt':
            spec = generate_cwt_spectrogram(signal, config['scales'], config['wavelet'], config['size'])
        elif method == 'wpt':
            spec = generate_wpt_spectrogram(signal, config['wavelet'], config['level'], config['size'])
        else:
            spec = np.zeros(config['size'], dtype=np.float32)
        
        if spec.shape != UNIFIED_SIZE:
            spec_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            spec = F.interpolate(spec_tensor, size=UNIFIED_SIZE, mode='bilinear', align_corners=False)
            spec = spec.squeeze().numpy()
        
        spectrograms.append(spec)
    
    return np.array(spectrograms, dtype=np.float32)

def generate_enhanced_channels(spec_6ch):
    """生成2个增强通道"""
    mixed_spec = np.mean(spec_6ch, axis=0)
    ch1 = np.log1p(mixed_spec)
    ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-8)
    
    low_freq = spec_6ch[:, :32, :]
    low_freq_avg = np.mean(low_freq, axis=0)
    grad_freq = np.gradient(low_freq_avg, axis=0)
    grad_tensor = torch.from_numpy(grad_freq).unsqueeze(0).unsqueeze(0).float()
    ch2 = F.interpolate(grad_tensor, size=UNIFIED_SIZE, mode='bilinear', align_corners=False)
    ch2 = ch2.squeeze().numpy()
    ch2 = (ch2 - ch2.min()) / (ch2.max() - ch2.min() + 1e-8)
    
    return np.stack([ch1, ch2], axis=0).astype(np.float32)

def extract_auxiliary_features(window_df):
    """提取57维辅助特征"""
    features = []
    curve_types = ['ac', 'resistivity', 'grd', 'sp', 'den', 'cnl']
    
    for curve_type in curve_types:
        col_name = find_curve_column(window_df, curve_type)
        
        if col_name is None or col_name not in window_df.columns:
            features.extend([0.0] * 8)
            continue
        
        signal = window_df[col_name].values
        
        if np.isnan(signal).any():
            signal = pd.Series(signal).interpolate(method='linear', limit_direction='both').fillna(0).values
        
        curve_features = [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.min(signal)),
            float(np.max(signal)),
            float(np.median(signal)),
            float(np.percentile(signal, 25)),
            float(np.percentile(signal, 75)),
            float(signal[-1] - signal[0])
        ]
        features.extend(curve_features)
    
    all_signals = []
    for curve_type in curve_types:
        col_name = find_curve_column(window_df, curve_type)
        if col_name and col_name in window_df.columns:
            signal = window_df[col_name].values
            if not np.isnan(signal).any():
                all_signals.append(signal)
    
    if len(all_signals) > 0:
        all_signals = np.array(all_signals)
        corr_matrix = np.corrcoef(all_signals)
        avg_corr = float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        
        fft_result = np.fft.fft(all_signals, axis=1)
        power_spectrum = np.abs(fft_result) ** 2
        high_freq_energy = np.sum(power_spectrum[:, WINDOW_SIZE//2:], axis=1)
        low_freq_energy = np.sum(power_spectrum[:, :WINDOW_SIZE//2], axis=1)
        energy_ratio = float(np.mean(high_freq_energy / (low_freq_energy + 1e-8)))
        
        overall_cv = float(np.mean(np.std(all_signals, axis=1) / (np.mean(all_signals, axis=1) + 1e-8)))
        
        peak_counts = []
        for sig in all_signals:
            peaks, _ = find_peaks(sig, height=np.mean(sig))
            peak_counts.append(len(peaks))
        avg_peak_count = float(np.mean(peak_counts) / WINDOW_SIZE)
        
        def sample_entropy(signal, m=2, r=0.2):
            N = len(signal)
            r = r * np.std(signal)
            
            def _maxdist(x_i, x_j):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
                C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
                return sum(np.log(C)) / (N - m + 1.0)
            
            return abs(_phi(m + 1) - _phi(m))
        
        try:
            complexity = float(np.mean([sample_entropy(sig) for sig in all_signals]))
        except:
            complexity = 0.0
        
        global_mean = float(np.mean(all_signals))
        global_std = float(np.std(all_signals))
        signal_range = float(np.max(all_signals) - np.min(all_signals))
        skewness = float(np.mean([pd.Series(sig).skew() for sig in all_signals]))
        
        global_features = [
            avg_corr, energy_ratio, overall_cv, avg_peak_count,
            complexity, global_mean, global_std, signal_range, skewness
        ]
    else:
        global_features = [0.0] * 9
    
    features.extend(global_features)
    return np.array(features, dtype=np.float32)

def augment_sample(spectrogram_8ch, aug_type='noise'):
    """数据增强"""
    aug_spec = spectrogram_8ch.copy()
    
    if aug_type == 'noise':
        noise = np.random.normal(0, 0.02, aug_spec.shape).astype(np.float32)
        aug_spec = np.clip(aug_spec + noise, 0, 1)
    elif aug_type == 'scale':
        scale = np.random.uniform(0.95, 1.05)
        aug_spec = np.clip(aug_spec * scale, 0, 1)
    
    return aug_spec

# ============ 主流程 ============

def main():
    print("=" * 80)
    print("[START] 一对一数据生成：先生成全井时频图谱序列，再一对一采样")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Step 1: 加载所有井数据
    print("\n[Step 1] 加载井数据...")
    
    all_wells = {}
    for root_name in ['train_wells', 'val_wells', 'test_wells']:
        root = Path(root_name)
        if not root.exists():
            continue
        
        for file_path in root.glob('*.txt'):
            well_name = file_path.stem
            df = read_well_data(file_path)
            
            if df is not None:
                all_wells[well_name] = df
                print(f"  [OK] {well_name}: {len(df)}行")
    
    print(f"\n  总计加载: {len(all_wells)}口井")
    
    # Step 2: 为每口井生成全井时频图谱序列
    print("\n[Step 2] 生成全井时频图谱序列（滑动窗口，深度对齐）...")
    
    well_spectrogram_sequences = {}
    well_aux_feature_sequences = {}
    
    for well_name, well_df in all_wells.items():
        if len(well_df) < WINDOW_SIZE:
            print(f"  [SKIP] {well_name}: 数据不足{WINDOW_SIZE}行")
            continue
        
        spec_sequence = []
        aux_sequence = []
        
        # 滑动窗口生成时频图谱序列
        for i in range(0, len(well_df) - WINDOW_SIZE + 1, STRIDE):
            window_df = well_df.iloc[i:i+WINDOW_SIZE]
            
            # 生成8通道时频图谱
            spec_6ch = extract_6channel_spectrograms(window_df)
            enhanced_2ch = generate_enhanced_channels(spec_6ch)
            spec_8ch = np.concatenate([spec_6ch, enhanced_2ch], axis=0)
            
            # 提取辅助特征
            aux_feat = extract_auxiliary_features(window_df)
            
            spec_sequence.append(spec_8ch)
            aux_sequence.append(aux_feat)
        
        well_spectrogram_sequences[well_name] = np.array(spec_sequence, dtype=np.float32)
        well_aux_feature_sequences[well_name] = np.array(aux_sequence, dtype=np.float32)
        
        print(f"  [OK] {well_name}: 生成{len(spec_sequence)}个时频图谱（深度0~{len(spec_sequence)-1}）")
    
    print(f"\n  成功生成: {len(well_spectrogram_sequences)}口井的时频图谱序列")
    
    # Step 3: 一对一采样（标注点→时频图谱）
    print("\n[Step 3] 一对一采样（流体标注点 → 对应时频图谱）...")
    
    base_samples = []
    
    for well_name in well_spectrogram_sequences.keys():
        well_df = all_wells[well_name]
        spec_seq = well_spectrogram_sequences[well_name]
        aux_seq = well_aux_feature_sequences[well_name]
        
        well_samples = 0
        
        # 遍历每个深度点
        for idx in range(len(well_df)):
            # 检查是否有流体标注
            fluid_label_text = well_df.iloc[idx]['解释结论']
            
            if pd.isna(fluid_label_text) or fluid_label_text not in FLUID_MAP:
                continue
            
            fluid_label = FLUID_MAP[fluid_label_text]
            
            # 计算对应的时频图谱索引（考虑窗口中心对齐）
            spec_idx = idx - WINDOW_SIZE // 2
            
            # 边界检查
            if spec_idx < 0 or spec_idx >= len(spec_seq):
                continue
            
            # 一对一采样
            sample = {
                'spectrogram': spec_seq[spec_idx],
                'aux_features': aux_seq[spec_idx],
                'label': fluid_label,
                'well': well_name,
                'depth_idx': idx,
                'spec_idx': spec_idx,
                'sample_id': f"{well_name}_{idx}",
                'is_augmented': False,
                'aug_type': 'none',
                'source_idx': well_samples
            }
            base_samples.append(sample)
            well_samples += 1
        
        print(f"  [OK] {well_name}: {well_samples}个样本")
    
    print(f"\n  基础样本总数: {len(base_samples)}")
    
    # 统计流体分布
    fluid_counter = Counter(s['label'] for s in base_samples)
    print(f"\n  流体分布:")
    for label_id in sorted(fluid_counter.keys()):
        count = fluid_counter[label_id]
        pct = count / len(base_samples) * 100
        print(f"    {FLUID_NAMES[label_id]}: {count} ({pct:.1f}%)")
    
    # Step 4: 划分训练集和验证集（随机划分 + 分组约束）
    print("\n[Step 4] 划分train/val（随机划分 + 分组约束，80:20）...")
    print("  策略：保持跨井学习能力，但避免相邻样本泄露")
    
    test_samples = [s for s in base_samples if s['well'] in TEST_WELLS]
    trainval_samples = [s for s in base_samples if s['well'] not in TEST_WELLS]
    
    print(f"  训练验证样本: {len(trainval_samples)}")
    print(f"  测试样本: {len(test_samples)}")
    
    # 关键：为每口井的样本分组（避免相邻样本跨集合）
    # 分组规则：每WINDOW_SIZE个连续深度索引为一组（避免99%重叠）
    print(f"\n  分组策略：每{WINDOW_SIZE}个连续样本为一组（避免时频图谱重叠）")
    
    groups = []
    labels_for_stratify = []
    
    for sample in trainval_samples:
        # 分组ID = 井名 + 深度组号
        depth_group = sample['depth_idx'] // WINDOW_SIZE
        group_id = f"{sample['well']}_g{depth_group}"
        groups.append(group_id)
        labels_for_stratify.append(sample['label'])
    
    groups = np.array(groups)
    labels_for_stratify = np.array(labels_for_stratify)
    
    unique_groups = np.unique(groups)
    print(f"  总分组数: {len(unique_groups)}")
    
    # 计算每个组的主要流体类型（用于分层）
    group_labels = {}
    for group in unique_groups:
        group_mask = groups == group
        group_label_counts = Counter(labels_for_stratify[group_mask])
        group_labels[group] = group_label_counts.most_common(1)[0][0]
    
    # GroupKFold：确保同组样本不跨集合
    # 同时使用分层策略保持流体分布平衡
    group_label_array = np.array([group_labels[g] for g in unique_groups])
    
    # 先对组进行分层抽样
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_group_idx, val_group_idx = next(skf.split(unique_groups, group_label_array))
    
    train_groups = set(unique_groups[train_group_idx])
    val_groups = set(unique_groups[val_group_idx])
    
    # 根据组划分样本
    train_samples = [s for i, s in enumerate(trainval_samples) if groups[i] in train_groups]
    val_samples = [s for i, s in enumerate(trainval_samples) if groups[i] in val_groups]
    
    # 统计井分布
    train_wells = set(s['well'] for s in train_samples)
    val_wells = set(s['well'] for s in val_samples)
    overlap_wells = train_wells & val_wells
    
    print(f"\n  分组结果:")
    print(f"    训练组: {len(train_groups)}, 验证组: {len(val_groups)}")
    print(f"    训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")
    print(f"    训练井: {len(train_wells)}, 验证井: {len(val_wells)}")
    print(f"    重叠井（跨井学习）: {len(overlap_wells)} 口 {sorted(overlap_wells)}")
    print(f"  [OK] 同组样本不跨集合，但保留跨井学习能力！")
    
    # 检查流体分布
    train_dist = Counter(s['label'] for s in train_samples)
    val_dist = Counter(s['label'] for s in val_samples)
    test_dist = Counter(s['label'] for s in test_samples)
    
    print(f"\n  流体分布检查（差异<10%即可）:")
    for label_id in sorted(set(train_dist.keys()) | set(val_dist.keys())):
        train_pct = train_dist.get(label_id, 0) / len(train_samples) * 100 if len(train_samples) > 0 else 0
        val_pct = val_dist.get(label_id, 0) / len(val_samples) * 100 if len(val_samples) > 0 else 0
        diff = abs(train_pct - val_pct)
        status = "[OK]" if diff < 10 else "[WARN]"
        print(f"    {FLUID_NAMES[label_id]}: train {train_pct:.1f}% vs val {val_pct:.1f}% -> 差异{diff:.1f}% {status}")
    
    # Step 5: 轻微少数类增强（根据实际分布）
    print("\n[Step 5] 轻微少数类增强（根据实际分布，避免过度增强）...")
    
    total_train = len(train_samples)
    class_counts = Counter(s['label'] for s in train_samples)
    
    print(f"  训练集分布:")
    for label_id in sorted(class_counts.keys()):
        count = class_counts[label_id]
        pct = count / total_train * 100
        print(f"    {FLUID_NAMES[label_id]}: {count} ({pct:.1f}%)")
    
    # 轻微增强策略（参考优化方案）
    AUGMENT_MULTIPLIER = {
        0: 1,  # 油层：不增强（样本充足）
        1: 2,  # 水层：×2（轻微增强）
        2: 1,  # 干层：不增强（样本充足）
        3: 2,  # 差油层：×2（轻微增强）
        4: 3   # 油水层：×3（最稀缺）
    }
    
    augmented_train = []
    aug_counter = 0
    
    for sample in train_samples:
        augmented_train.append(sample)
        
        label_id = sample['label']
        aug_times = AUGMENT_MULTIPLIER.get(label_id, 1) - 1  # 减1因为原样本已添加
        
        for aug_round in range(aug_times):
            aug_type = np.random.choice(['noise', 'scale'])
            aug_spec = augment_sample(sample['spectrogram'], aug_type=aug_type)
            aug_aux = sample['aux_features'] + np.random.normal(0, 0.01, sample['aux_features'].shape).astype(np.float32)
            aug_aux = np.clip(aug_aux, 0, None)
            
            aug_sample = {
                'spectrogram': aug_spec,
                'aux_features': aug_aux,
                'label': sample['label'],
                'well': sample['well'],
                'depth_idx': sample['depth_idx'],
                'spec_idx': sample['spec_idx'],
                'sample_id': f"{sample['sample_id']}_aug{aug_counter}",
                'is_augmented': True,
                'aug_type': aug_type,
                'source_idx': sample['source_idx'],
                'aug_params': {
                    'round': aug_round,
                    'noise_std': 0.02 if aug_type == 'noise' else 0.0,
                    'scale_factor': np.random.uniform(0.95, 1.05) if aug_type == 'scale' else 1.0
                }
            }
            augmented_train.append(aug_sample)
            aug_counter += 1
    
    print(f"\n  增强策略:")
    for label_id in sorted(class_counts.keys()):
        orig_count = class_counts[label_id]
        aug_count = len([s for s in augmented_train if s['label'] == label_id])
        pct_before = orig_count / total_train * 100
        pct_after = aug_count / len(augmented_train) * 100
        print(f"    {FLUID_NAMES[label_id]}: {orig_count}({pct_before:.1f}%) -> {aug_count}({pct_after:.1f}%)")
    
    train_samples = augmented_train
    print(f"\n  增强后训练集: {len(train_samples)} 样本")
    
    # Step 6: 保存数据
    print("\n[Step 6] 保存数据...")
    
    def save_samples(samples, split_name):
        specs = np.array([s['spectrogram'] for s in samples], dtype=np.float32)
        aux_feats = np.array([s['aux_features'] for s in samples], dtype=np.float32)
        labels = np.array([s['label'] for s in samples], dtype=np.int32)
        wells = np.array([s['well'] for s in samples], dtype='<U50')
        depth_indices = np.array([s['depth_idx'] for s in samples], dtype=np.int32)
        sample_ids = np.array([s['sample_id'] for s in samples], dtype='<U150')
        is_augmented = np.array([s.get('is_augmented', False) for s in samples], dtype=bool)
        aug_types = np.array([s.get('aug_type', 'none') for s in samples], dtype='<U20')
        source_indices = np.array([s.get('source_idx', -1) for s in samples], dtype=np.int32)
        
        output_file = OUTPUT_DIR / f'{split_name}_samples.npz'
        np.savez_compressed(
            output_file,
            spectrograms=specs,
            aux_features=aux_feats,
            labels=labels,
            wells=wells,
            depth_indices=depth_indices,
            sample_ids=sample_ids,
            is_augmented=is_augmented,
            aug_types=aug_types,
            source_indices=source_indices
        )
        
        n_aug = is_augmented.sum()
        n_orig = len(samples) - n_aug
        
        print(f"  [OK] {split_name}: {len(samples)}样本 -> {output_file}")
        print(f"      Shape: {specs.shape} (8ch×64×64), Aux: {aux_feats.shape} (57-dim)")
        print(f"      原始: {n_orig}, 增强: {n_aug}")
    
    save_samples(train_samples, 'train')
    save_samples(val_samples, 'val')
    save_samples(test_samples, 'test')
    
    # 保存统计报告
    report = {
        'method': 'one_to_one_with_group_constraint',
        'description': '先生成全井时频图谱序列（滑动窗口），再根据标注点一对一采样，使用分组约束避免相邻样本泄露',
        'split_strategy': '随机划分 + 分组约束（每128个连续样本为一组，同组不跨集合，保留跨井学习能力）',
        'augmentation_strategy': '轻微增强（水层x2，差油层x2，油水层x3，油层/干层x1）',
        'test_wells': TEST_WELLS,
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'group_size': WINDOW_SIZE,
        'spectrogram_size': list(UNIFIED_SIZE),
        'channels': 8,
        'channel_description': '6通道时频图谱(AC/RT高频CWT, GR/SP/DEN低频WPT, CNL混合) + 2通道增强(log1p混合谱, 低频梯度)',
        'aux_feature_dim': 57,
        'data_counts': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
            'total': len(train_samples) + len(val_samples) + len(test_samples)
        },
        'well_distribution': {
            'train_wells': sorted(list(train_wells)),
            'val_wells': sorted(list(val_wells)),
            'overlap_wells': sorted(list(overlap_wells))
        },
        'fluid_distribution': {
            'train': {FLUID_NAMES[k]: int(v) for k, v in Counter(s['label'] for s in train_samples).items()},
            'val': {FLUID_NAMES[k]: int(v) for k, v in Counter(s['label'] for s in val_samples).items()},
            'test': {FLUID_NAMES[k]: int(v) for k, v in Counter(s['label'] for s in test_samples).items()}
        }
    }
    
    report_file = OUTPUT_DIR / 'generation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n  [OK] 统计报告: {report_file}")
    
    # 最终总结
    print("\n" + "=" * 80)
    print("[SUCCESS] 一对一数据生成完成！")
    print("=" * 80)
    print(f"\n最终数据量:")
    print(f"  训练集: {len(train_samples)} 样本")
    print(f"  验证集: {len(val_samples)} 样本")
    print(f"  测试集: {len(test_samples)} 样本")
    print(f"  总计: {len(train_samples) + len(val_samples) + len(test_samples)} 样本")
    
    train_aug_count = sum(1 for s in train_samples if s.get('is_augmented', False))
    val_aug_count = sum(1 for s in val_samples if s.get('is_augmented', False))
    test_aug_count = sum(1 for s in test_samples if s.get('is_augmented', False))
    
    print(f"\n增强样本统计:")
    print(f"  训练集增强: {train_aug_count}/{len(train_samples)} ({train_aug_count/len(train_samples)*100:.1f}%)")
    val_aug_pct = val_aug_count/len(val_samples)*100 if len(val_samples) > 0 else 0
    test_aug_pct = test_aug_count/len(test_samples)*100 if len(test_samples) > 0 else 0
    print(f"  验证集增强: {val_aug_count}/{len(val_samples)} ({val_aug_pct:.1f}%)")
    print(f"  测试集增强: {test_aug_count}/{len(test_samples)} ({test_aug_pct:.1f}%)")
    
    print(f"\n数据保存位置: {OUTPUT_DIR.absolute()}")
    print(f"\n关键特性:")
    print(f"  [OK] 先生成全井时频图谱序列（滑动窗口STRIDE={STRIDE}）")
    print(f"  [OK] 再根据流体标注点一对一采样（深度索引→时频图谱索引）")
    print(f"  [OK] 8通道时频图谱（64x64）:")
    print(f"      - 前6通道: AC/RT(CWT+Morlet), GR/SP/DEN(WPT+Sym8), CNL(混合)")
    print(f"      - 增强通道1: log1p混合谱")
    print(f"      - 增强通道2: 低频梯度")
    print(f"  [OK] 57维辅助特征向量")
    print(f"  [OK] 方案1（随机划分 + 分组约束）:")
    print(f"      - 每{WINDOW_SIZE}个连续样本为一组（避免99%重叠泄露）")
    print(f"      - 同组样本不跨train/val集合")
    print(f"      - 但保留跨井学习能力（不同井的组可在不同集合）")
    print(f"      - 分层抽样保持流体分布平衡（差不多即可）")
    print(f"  [OK] 轻微少数类增强（参考优化方案）:")
    print(f"      - 水层x2, 差油层x2, 油水层x3")
    print(f"      - 油层x1, 干层x1（不增强）")
    print(f"      - 增强方法：轻微噪声(std=0.02)、缩放(±5%)")
    print(f"  [OK] 增强样本额外标注（is_augmented, aug_type, source_idx, aug_params）")

if __name__ == '__main__':
    main()

