#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
井级平衡数据加载器
解决井内类别分布不均匀的问题
不需要重新生成optimized_patch_data，直接在加载时平衡采样
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import random

from six_channel_loader import (
    SixChannelPatchDataset,
    create_six_channel_data_loaders
)


def analyze_well_class_distribution(files: List[Path]) -> Dict:
    """分析井和类别的分布"""
    well_class_counts = defaultdict(lambda: defaultdict(int))
    well_files = defaultdict(lambda: defaultdict(list))
    
    print("\n" + "="*80)
    print("📊 分析井级类别分布...")
    print("="*80)
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            well_name = str(data['well_name'])
            label = int(data['label'])
            well_class_counts[well_name][label] += 1
            well_files[well_name][label].append(f)
        except Exception as e:
            print(f"  ⚠️ 跳过文件 {f.name}: {e}")
            continue
    
    # 打印分布
    fluid_names = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}
    
    for well_name in sorted(well_class_counts.keys()):
        class_counts = well_class_counts[well_name]
        total = sum(class_counts.values())
        
        print(f"\n📍 {well_name} (总样本: {total})")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            pct = 100 * count / total
            print(f"   {fluid_names.get(cls, f'类{cls}')}: {count:5d} ({pct:5.1f}%)")
        
        # 检查平衡性
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 5:
                print(f"   ⚠️ 严重失衡 (比例: {imbalance_ratio:.1f}:1)")
            elif imbalance_ratio > 2:
                print(f"   ⚠️ 中度失衡 (比例: {imbalance_ratio:.1f}:1)")
            else:
                print(f"   ✅ 相对平衡 (比例: {imbalance_ratio:.1f}:1)")
    
    print("="*80)
    
    return {
        'well_class_counts': dict(well_class_counts),
        'well_files': dict(well_files)
    }


def balance_samples_within_wells(
    files: List[Path], 
    strategy: str = 'downsample',
    target_samples_per_class: Optional[int] = None,
    min_samples_per_class: int = 50,
    random_seed: int = 42
) -> List[Path]:
    """
    在井级别进行类别平衡采样
    
    Args:
        files: 所有npz文件路径
        strategy: 平衡策略
            - 'downsample': 下采样到少数类（推荐，避免过拟合）
            - 'target': 采样到目标数量
        target_samples_per_class: 每口井每个类别的目标样本数（仅strategy='target'时有效）
        min_samples_per_class: 最小样本数（少于此数的类别将被全部保留）
        random_seed: 随机种子
    
    Returns:
        平衡后的文件列表
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 分析分布
    analysis = analyze_well_class_distribution(files)
    well_files = analysis['well_files']
    well_class_counts = analysis['well_class_counts']
    
    print(f"\n🔄 应用井级平衡策略: {strategy}")
    print("="*80)
    
    balanced_files = []
    stats = []
    
    for well_name in sorted(well_files.keys()):
        class_files_dict = well_files[well_name]
        class_counts = well_class_counts[well_name]
        
        if not class_counts:
            continue
        
        # 确定采样目标
        if strategy == 'downsample':
            # 下采样到少数类的数量
            target = max(min(class_counts.values()), min_samples_per_class)
        elif strategy == 'target':
            # 采样到指定目标
            target = target_samples_per_class or min(class_counts.values())
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        well_balanced = []
        well_stats = {'well': well_name, 'original': {}, 'balanced': {}}
        
        for cls, files_list in class_files_dict.items():
            original_count = len(files_list)
            well_stats['original'][cls] = original_count
            
            if original_count <= target:
                # 少数类：全部保留
                sampled = files_list
            else:
                # 多数类：随机下采样
                sampled = random.sample(files_list, target)
            
            well_balanced.extend(sampled)
            well_stats['balanced'][cls] = len(sampled)
        
        balanced_files.extend(well_balanced)
        stats.append(well_stats)
        
        # 打印该井的平衡结果
        print(f"\n📍 {well_name}:")
        print(f"   目标样本数/类: {target}")
        fluid_names = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}
        for cls in sorted(well_stats['original'].keys()):
            orig = well_stats['original'][cls]
            bal = well_stats['balanced'][cls]
            print(f"   {fluid_names.get(cls, f'类{cls}')}: {orig} → {bal}")
    
    print("="*80)
    print(f"✅ 平衡完成:")
    print(f"   原始样本总数: {len(files)}")
    print(f"   平衡后样本数: {len(balanced_files)}")
    print(f"   样本保留率: {100*len(balanced_files)/len(files):.1f}%")
    print("="*80)
    
    return balanced_files


def create_balanced_data_loaders(
    use_presplit: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
    # 平衡参数
    enable_well_balance: bool = True,
    balance_strategy: str = 'downsample',
    target_samples_per_class: Optional[int] = None,
    min_samples_per_class: int = 50,
    # 数据增强参数
    add_subband_features: bool = True,
    add_wavelet_vector: bool = True,
    enable_crosscorr_heatmap: bool = True,
    prefer_precomputed_aux: bool = True,
    # 其他参数
    random_seed: int = 42,
    **kwargs
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    创建井级平衡的数据加载器
    
    Args:
        enable_well_balance: 是否启用井级平衡
        balance_strategy: 平衡策略 ('downsample' 或 'target')
        target_samples_per_class: 每口井每类的目标样本数（strategy='target'时有效）
        min_samples_per_class: 最小样本数阈值
        其他参数: 与原 create_six_channel_data_loaders 相同
    
    Returns:
        (train_loader, val_loader, metadata)
    """
    
    print("\n" + "="*80)
    print("🚀 创建井级平衡数据加载器")
    print("="*80)
    
    if use_presplit:
        train_dir = Path('optimized_patch_data/train_spectrograms')
        val_dir = Path('optimized_patch_data/valid_spectrograms')
        
        train_files = list(train_dir.glob('*.npz'))
        val_files = list(val_dir.glob('*.npz'))
    else:
        # 使用旧格式（未分割）
        data_dir = Path('optimized_patch_data/spectrograms')
        all_files = list(data_dir.glob('*.npz'))
        # 这里需要按井划分，暂不实现
        raise NotImplementedError("请使用 use_presplit=True")
    
    print(f"\n📁 原始数据统计:")
    print(f"   训练样本: {len(train_files)}")
    print(f"   验证样本: {len(val_files)}")
    
    # 应用井级平衡
    if enable_well_balance:
        train_files = balance_samples_within_wells(
            train_files,
            strategy=balance_strategy,
            target_samples_per_class=target_samples_per_class,
            min_samples_per_class=min_samples_per_class,
            random_seed=random_seed
        )
    else:
        print("\n⚠️ 未启用井级平衡，使用原始数据分布")
    
    # 调用原始加载器创建数据集
    print("\n📦 创建数据集和加载器...")
    
    # 计算通道统计（用于标准化）
    print("   计算通道统计...")
    # 检测实际通道数
    sample_file = train_files[0] if train_files else None
    if sample_file:
        test_data = np.load(sample_file, allow_pickle=False)
        actual_channels = test_data['spectrogram'].shape[0]
        print(f"   检测到通道数: {actual_channels}")
    else:
        actual_channels = 6
    
    # 收集样本并计算统计
    channel_means = []
    channel_stds = []
    
    for f in train_files[:min(1000, len(train_files))]:
        try:
            data = np.load(f, allow_pickle=False)
            spec = data['spectrogram']
            # 按通道计算均值和标准差
            channel_means.append(np.mean(spec, axis=(1, 2)))  # (C,)
            channel_stds.append(np.std(spec, axis=(1, 2)))    # (C,)
        except Exception:
            continue
    
    if channel_means:
        channel_mean = np.mean(channel_means, axis=0)  # (C,)
        channel_std = np.mean(channel_stds, axis=0) + 1e-6  # (C,)
    else:
        channel_mean = np.zeros(actual_channels)
        channel_std = np.ones(actual_channels)
    
    # 创建数据集
    train_dataset = SixChannelPatchDataset(
        file_paths=train_files,
        transform=None,
        augment=True,
        channel_mean=channel_mean,
        channel_std=channel_std,
        add_subband_features=add_subband_features,
        add_wavelet_vector=add_wavelet_vector,
        enable_crosscorr_heatmap=enable_crosscorr_heatmap,
        prefer_precomputed_aux=prefer_precomputed_aux
    )
    
    val_dataset = SixChannelPatchDataset(
        file_paths=val_files,
        transform=None,
        augment=False,
        channel_mean=channel_mean,
        channel_std=channel_std,
        add_subband_features=add_subband_features,
        add_wavelet_vector=add_wavelet_vector,
        enable_crosscorr_heatmap=enable_crosscorr_heatmap,
        prefer_precomputed_aux=prefer_precomputed_aux
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 元数据
    metadata = {
        'train_samples': len(train_files),
        'val_samples': len(val_files),
        'channel_mean': channel_mean,
        'channel_std': channel_std,
        'well_balanced': enable_well_balance,
        'balance_strategy': balance_strategy if enable_well_balance else None
    }
    
    print("\n✅ 数据加载器创建完成!")
    print(f"   训练批次数: {len(train_loader)}")
    print(f"   验证批次数: {len(val_loader)}")
    print("="*80 + "\n")
    
    return train_loader, val_loader, metadata


def test_balanced_loader():
    """测试井级平衡数据加载器"""
    print("\n🧪 测试井级平衡数据加载器\n")
    
    # 创建平衡加载器
    train_loader, val_loader, meta = create_balanced_data_loaders(
        use_presplit=True,
        batch_size=32,
        num_workers=0,
        enable_well_balance=True,
        balance_strategy='downsample',
        min_samples_per_class=100,
        add_subband_features=True
    )
    
    print(f"\n📊 元数据:")
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            print(f"   {k}: {v.shape}")
        else:
            print(f"   {k}: {v}")
    
    # 测试一个batch
    print(f"\n🔍 测试第一个batch:")
    for batch_idx, (spectrograms, metadata_dict, labels) in enumerate(train_loader):
        print(f"   Batch {batch_idx}:")
        print(f"     Spectrograms shape: {spectrograms.shape}")
        print(f"     Labels shape: {labels.shape}")
        print(f"     Labels: {labels.numpy()}")
        print(f"     Label distribution: {Counter(labels.numpy().tolist())}")
        
        if batch_idx >= 2:  # 只测试前3个batch
            break
    
    print("\n✅ 测试完成!")


if __name__ == '__main__':
    test_balanced_loader()

