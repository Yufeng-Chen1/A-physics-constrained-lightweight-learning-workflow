#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B：数据质量深度诊断
重点检查：相对深度对齐问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json

print("=" * 80)
print("🔍 方案B：数据质量深度诊断")
print("=" * 80)
print()

# 加载数据
from pathlib import Path
import torch
import numpy as np

print("📊 加载训练和验证数据...")

# 查找数据目录
if Path('awpd_final_correct_data').exists():
    data_dir = Path('awpd_final_correct_data')
elif Path('../awpd_final_correct_data').exists():
    data_dir = Path('../awpd_final_correct_data')
else:
    raise FileNotFoundError("找不到awpd_final_correct_data目录")

print(f"数据目录: {data_dir}")

# 加载训练集
train_npz = data_dir / 'train_samples.npz'
train_data = np.load(train_npz, allow_pickle=True)
train_labels = train_data['labels']
train_well_names = train_data['well_names']
print(f"训练集: {len(train_labels)} 样本")

# 加载验证集
val_npz = data_dir / 'val_samples.npz'
val_data = np.load(val_npz, allow_pickle=True)
val_labels = val_data['labels']
val_well_names = val_data['well_names']
print(f"验证集: {len(val_labels)} 样本")

# 提取频谱数据
train_spectrograms = train_data['spectrogram_low']  # 使用低频谱作为代表
val_spectrograms = val_data['spectrogram_low']

# 尝试加载辅助特征
precomputed_train_dir = data_dir / 'precomputed' / 'train' / 'aux_vec'
precomputed_val_dir = data_dir / 'precomputed' / 'val' / 'aux_vec'

if precomputed_train_dir.exists():
    print(f"找到预计算辅助特征目录")
    # 加载第一个样本查看特征维度
    sample_files = list(precomputed_train_dir.glob('*.npy'))
    if len(sample_files) > 0:
        first_aux = np.load(sample_files[0])
        print(f"辅助特征维度: {first_aux.shape}")
        
        # 加载所有辅助特征
        train_aux_list = []
        for idx in range(len(train_labels)):
            well_name = str(train_well_names[idx])
            aux_file = precomputed_train_dir / f"{well_name}_{idx}.npy"
            if aux_file.exists():
                train_aux_list.append(np.load(aux_file))
            else:
                print(f"警告：缺少辅助特征文件: {aux_file}")
                train_aux_list.append(np.zeros_like(first_aux))
        train_aux = np.array(train_aux_list)
        
        val_aux_list = []
        for idx in range(len(val_labels)):
            well_name = str(val_well_names[idx])
            aux_file = precomputed_val_dir / f"{well_name}_{idx}.npy"
            if aux_file.exists():
                val_aux_list.append(np.load(aux_file))
            else:
                val_aux_list.append(np.zeros_like(first_aux))
        val_aux = np.array(val_aux_list)
    else:
        print("警告：辅助特征目录为空")
        train_aux = np.zeros((len(train_labels), 57))
        val_aux = np.zeros((len(val_labels), 57))
else:
    print("警告：未找到预计算辅助特征目录")
    train_aux = np.zeros((len(train_labels), 57))
    val_aux = np.zeros((len(val_labels), 57))

print()

class_names = ['油层', '水层', '干层', '差油层', '油水层']

print("=" * 80)
print("📋 检查1: 类别分布对比")
print("=" * 80)

train_counts = np.bincount(train_labels)
val_counts = np.bincount(val_labels)

print(f"\n{'类别':<10} {'训练集数量':<12} {'训练集比例':<12} {'验证集数量':<12} {'验证集比例':<12} {'差异':<10}")
print("-" * 80)
for i, name in enumerate(class_names):
    train_ratio = train_counts[i] / len(train_labels) * 100
    val_ratio = val_counts[i] / len(val_labels) * 100
    diff = abs(train_ratio - val_ratio)
    status = "✅" if diff < 5 else ("⚠️" if diff < 10 else "❌")
    print(f"{name:<10} {train_counts[i]:<12} {train_ratio:>5.1f}%{'':6} {val_counts[i]:<12} {val_ratio:>5.1f}%{'':6} {diff:>5.1f}% {status}")

print()

print("=" * 80)
print("📋 检查2: 频谱特征质量")
print("=" * 80)

print(f"\n训练集频谱统计:")
print(f"  形状: {train_spectrograms.shape}")
print(f"  均值: {train_spectrograms.mean():.4f}")
print(f"  标准差: {train_spectrograms.std():.4f}")
print(f"  最小值: {train_spectrograms.min():.4f}")
print(f"  最大值: {train_spectrograms.max():.4f}")
print(f"  NaN数量: {np.isnan(train_spectrograms).sum()}")
print(f"  Inf数量: {np.isinf(train_spectrograms).sum()}")

print(f"\n验证集频谱统计:")
print(f"  形状: {val_spectrograms.shape}")
print(f"  均值: {val_spectrograms.mean():.4f}")
print(f"  标准差: {val_spectrograms.std():.4f}")
print(f"  最小值: {val_spectrograms.min():.4f}")
print(f"  最大值: {val_spectrograms.max():.4f}")
print(f"  NaN数量: {np.isnan(val_spectrograms).sum()}")
print(f"  Inf数量: {np.isinf(val_spectrograms).sum()}")

# 检查各通道的均值和标准差
print(f"\n各通道统计 (训练集):")
for ch in range(train_spectrograms.shape[1]):
    ch_data = train_spectrograms[:, ch, :, :]
    print(f"  通道{ch}: 均值={ch_data.mean():.4f}, 标准差={ch_data.std():.4f}, 范围=[{ch_data.min():.4f}, {ch_data.max():.4f}]")

print()

print("=" * 80)
print("📋 检查3: 辅助特征质量")
print("=" * 80)

print(f"\n训练集辅助特征统计:")
print(f"  形状: {train_aux.shape}")
print(f"  均值: {train_aux.mean():.4f}")
print(f"  标准差: {train_aux.std():.4f}")
print(f"  最小值: {train_aux.min():.4f}")
print(f"  最大值: {train_aux.max():.4f}")
print(f"  NaN数量: {np.isnan(train_aux).sum()}")
print(f"  Inf数量: {np.isinf(train_aux).sum()}")

print(f"\n验证集辅助特征统计:")
print(f"  形状: {val_aux.shape}")
print(f"  均值: {val_aux.mean():.4f}")
print(f"  标准差: {val_aux.std():.4f}")
print(f"  最小值: {val_aux.min():.4f}")
print(f"  最大值: {val_aux.max():.4f}")
print(f"  NaN数量: {np.isnan(val_aux).sum()}")
print(f"  Inf数量: {np.isinf(val_aux).sum()}")

# 检查是否有常数特征
print(f"\n常数特征检查 (std < 1e-6):")
train_stds = train_aux.std(axis=0)
const_features = np.where(train_stds < 1e-6)[0]
if len(const_features) > 0:
    print(f"  ❌ 发现 {len(const_features)} 个常数特征: {const_features[:10]}...")
else:
    print(f"  ✅ 无常数特征")

# 检查特征分布差异
print(f"\n训练集vs验证集特征分布差异:")
train_means = train_aux.mean(axis=0)
val_means = val_aux.mean(axis=0)
mean_diff = np.abs(train_means - val_means).mean()
print(f"  均值差异: {mean_diff:.4f}")

train_stds = train_aux.std(axis=0)
val_stds = val_aux.std(axis=0)
std_diff = np.abs(train_stds - val_stds).mean()
print(f"  标准差差异: {std_diff:.4f}")

print()

print("=" * 80)
print("📋 检查4: ⚠️ 相对深度对齐问题（关键！）")
print("=" * 80)

# 尝试从原始数据文件中加载深度信息
try:
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'awpd_final_correct_data')
    
    # 检查是否有深度信息文件
    depth_file = os.path.join(data_dir, 'depth_info.npy')
    metadata_file = os.path.join(data_dir, 'metadata.json')
    
    has_depth_info = False
    
    if os.path.exists(depth_file):
        print(f"✅ 找到深度信息文件: {depth_file}")
        depth_info = np.load(depth_file, allow_pickle=True)
        print(f"   深度信息形状: {depth_info.shape if hasattr(depth_info, 'shape') else 'dict'}")
        has_depth_info = True
    else:
        print(f"❌ 未找到深度信息文件: {depth_file}")
    
    if os.path.exists(metadata_file):
        print(f"✅ 找到元数据文件: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"   元数据键: {list(metadata.keys())}")
        has_depth_info = True
    else:
        print(f"❌ 未找到元数据文件: {metadata_file}")
    
    if not has_depth_info:
        print()
        print("⚠️ 警告：没有找到深度信息！")
        print("   这可能是导致性能瓶颈的根本原因！")
        print()
        print("🔍 深度对齐问题分析:")
        print("   1. 如果训练数据中没有深度信息，模型无法学习深度相关的特征")
        print("   2. 同一储层类型在不同深度可能有不同的物理特性")
        print("   3. 需要相对深度归一化才能让同类样本对齐")
        print()
        print("💡 建议解决方案:")
        print("   1. 检查原始数据是否包含深度信息")
        print("   2. 对每个样本添加相对深度特征（层顶/层底归一化）")
        print("   3. 确保训练时使用深度信息")
    
except Exception as e:
    print(f"❌ 读取深度信息时出错: {e}")

print()

print("=" * 80)
print("📋 检查5: 类内相似度 vs 类间相似度")
print("=" * 80)

print("\n计算频谱特征的余弦相似度...")

def compute_similarity_matrix(data, labels, max_samples=200):
    """计算类内和类间相似度"""
    # 展平频谱特征
    features = data.reshape(len(data), -1)
    
    # 随机采样以加速计算
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # 归一化
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # 计算相似度矩阵
    similarity = np.dot(features, features.T)
    
    # 计算类内和类间相似度
    intra_class_sim = []
    inter_class_sim = []
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            sim = similarity[i, j]
            if labels[i] == labels[j]:
                intra_class_sim.append(sim)
            else:
                inter_class_sim.append(sim)
    
    return np.array(intra_class_sim), np.array(inter_class_sim)

train_intra, train_inter = compute_similarity_matrix(train_spectrograms, train_labels)

print(f"\n训练集相似度统计:")
print(f"  类内相似度: 均值={train_intra.mean():.4f}, 标准差={train_intra.std():.4f}")
print(f"  类间相似度: 均值={train_inter.mean():.4f}, 标准差={train_inter.std():.4f}")
print(f"  分离度: {train_intra.mean() - train_inter.mean():.4f}")

if train_intra.mean() - train_inter.mean() < 0.1:
    print(f"  ❌ 警告：类内相似度与类间相似度差异很小！")
    print(f"     说明同类样本之间差异大，不同类样本之间差异小")
    print(f"     这可能是深度未对齐导致的！")
else:
    print(f"  ✅ 类内外相似度差异合理")

print()

print("=" * 80)
print("📋 检查6: 每个类别的特征统计")
print("=" * 80)

for class_idx, class_name in enumerate(class_names):
    mask = train_labels == class_idx
    if mask.sum() == 0:
        continue
    
    class_spectrograms = train_spectrograms[mask]
    class_aux = train_aux[mask]
    
    print(f"\n{class_name} (样本数: {mask.sum()}):")
    print(f"  频谱均值: {class_spectrograms.mean():.4f} ± {class_spectrograms.std():.4f}")
    print(f"  辅助特征均值: {class_aux.mean():.4f} ± {class_aux.std():.4f}")
    
    # 检查类内方差
    class_var = class_spectrograms.var(axis=0).mean()
    print(f"  类内方差: {class_var:.4f}")

print()

print("=" * 80)
print("📊 诊断总结")
print("=" * 80)

issues = []
warnings = []

# 检查类别分布
for i in range(len(class_names)):
    train_ratio = train_counts[i] / len(train_labels) * 100
    val_ratio = val_counts[i] / len(val_labels) * 100
    diff = abs(train_ratio - val_ratio)
    if diff >= 10:
        issues.append(f"❌ {class_names[i]}：训练/验证集分布差异过大 ({diff:.1f}%)")
    elif diff >= 5:
        warnings.append(f"⚠️ {class_names[i]}：训练/验证集分布略有差异 ({diff:.1f}%)")

# 检查数据质量
if np.isnan(train_spectrograms).sum() > 0 or np.isnan(train_aux).sum() > 0:
    issues.append("❌ 训练数据包含NaN值")

if np.isinf(train_spectrograms).sum() > 0 or np.isinf(train_aux).sum() > 0:
    issues.append("❌ 训练数据包含Inf值")

if len(const_features) > 0:
    warnings.append(f"⚠️ 发现{len(const_features)}个常数特征（可能无用）")

if train_intra.mean() - train_inter.mean() < 0.1:
    issues.append("❌ 类内相似度与类间相似度差异过小（<0.1）")

if not has_depth_info:
    issues.append("❌ 缺少深度信息！这可能是导致性能瓶颈的根本原因！")

print()
if len(issues) > 0:
    print("🚨 发现的严重问题:")
    for issue in issues:
        print(f"   {issue}")
    print()

if len(warnings) > 0:
    print("⚠️ 发现的警告:")
    for warning in warnings:
        print(f"   {warning}")
    print()

if len(issues) == 0 and len(warnings) == 0:
    print("✅ 未发现明显的数据质量问题")
    print("   性能瓶颈可能来自其他方面（模型架构、任务难度等）")
else:
    print("💡 建议的解决方案:")
    if not has_depth_info:
        print("   1. 🔥 最优先：添加深度信息并进行相对深度对齐")
        print("      - 对每个样本记录其在储层中的相对位置")
        print("      - 将深度归一化到[0,1]（0=层顶，1=层底）")
        print("      - 作为额外特征输入模型")
    if len(const_features) > 0:
        print("   2. 移除常数特征或重新提取特征")
    if train_intra.mean() - train_inter.mean() < 0.1:
        print("   3. 考虑使用对比学习或度量学习提高类别可分性")

print()
print("=" * 80)
print("✅ 诊断完成！")
print("=" * 80)

