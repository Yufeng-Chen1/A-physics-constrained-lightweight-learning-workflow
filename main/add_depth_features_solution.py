#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解决方案：添加相对深度特征
这是突破30-36%性能瓶颈的关键！
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

print("=" * 80)
print("🎯 方案1：添加相对深度特征")
print("=" * 80)
print()

print("📋 步骤概述:")
print("1. 检查原始TXT数据是否包含深度列")
print("2. 提取每个样本的深度信息")
print("3. 计算每个储层的层顶/层底深度")
print("4. 计算相对深度 = (depth - layer_top) / (layer_bottom - layer_top)")
print("5. 将相对深度保存到NPZ文件")
print("6. 修改训练脚本，将相对深度作为特征输入")
print()

print("=" * 80)
print("步骤1: 检查原始数据")
print("=" * 80)

# 查找原始数据目录
possible_dirs = [
    'welldata',
    '../welldata',
    'data/welldata',
    'original_data',
    '../original_data'
]

original_data_dir = None
for dir_path in possible_dirs:
    if Path(dir_path).exists():
        original_data_dir = Path(dir_path)
        print(f"✅ 找到原始数据目录: {original_data_dir}")
        break

if original_data_dir is None:
    print("❌ 未找到原始数据目录")
    print()
    print("⚠️ 可能的原因:")
    print("1. 原始TXT数据已被删除")
    print("2. 数据在其他位置")
    print()
    print("💡 请提供原始测井数据的位置，或确认以下信息:")
    print()
    print("【需要的信息】")
    print("1. 原始TXT文件在哪里？")
    print("2. TXT文件是否包含深度列（DEPTH或深度）？")
    print("3. 每个储层段的深度范围是否已知？")
    print()
    print("【如果无法获取原始数据】")
    print("可以尝试方案2（井内归一化）或方案3（井ID编码）")
    print("但效果会明显不如方案1")
    sys.exit(1)

# 检查数据结构
print()
print("检查数据结构...")
wells = list(original_data_dir.iterdir())
if len(wells) > 0:
    print(f"找到 {len(wells)} 个井目录")
    
    # 检查第一口井
    first_well = wells[0]
    print(f"\n示例井: {first_well.name}")
    
    # 查找TXT文件
    txt_files = list(first_well.glob('*.txt'))
    if len(txt_files) > 0:
        print(f"找到 {len(txt_files)} 个TXT文件")
        
        # 读取第一个TXT文件查看列名
        first_txt = txt_files[0]
        print(f"\n读取示例文件: {first_txt.name}")
        
        try:
            # 尝试不同的编码
            for encoding in ['utf-8', 'gbk', 'gb2312']:
                try:
                    df = pd.read_csv(first_txt, sep='\t', encoding=encoding, nrows=5)
                    print(f"✅ 成功读取（编码: {encoding}）")
                    print(f"\n列名: {list(df.columns)}")
                    print(f"\n前5行:")
                    print(df.head())
                    
                    # 检查是否有深度列
                    depth_columns = [col for col in df.columns if 'depth' in col.lower() or '深度' in col or 'dep' in col.lower()]
                    
                    if len(depth_columns) > 0:
                        print(f"\n✅ 找到深度列: {depth_columns}")
                        print()
                        print("=" * 80)
                        print("🎉 太好了！数据中包含深度信息")
                        print("=" * 80)
                        print()
                        print("接下来的步骤:")
                        print()
                        print("1. ✅ 原始数据包含深度信息")
                        print("2. 🔄 需要修改数据生成脚本，提取并保存深度")
                        print("3. 🔄 计算每个储层样本的相对深度")
                        print("4. 🔄 将相对深度添加到训练数据")
                        print("5. 🔄 修改模型输入，使用深度特征")
                        print()
                        print("💡 建议:")
                        print("   查找并修改数据生成脚本（可能是generate_awpd_*.py）")
                        print("   在生成NPZ文件时，添加depth和relative_depth字段")
                        print()
                        print("📝 需要修改的关键代码:")
                        print("   ```python")
                        print("   # 在数据生成时保存深度信息")
                        print("   np.savez(output_file,")
                        print("            spectrogram_high=...,")
                        print("            spectrogram_low=...,")
                        print("            ...,")
                        print("            depths=depths,  # 添加这行")
                        print("            relative_depths=rel_depths)  # 添加这行")
                        print("   ```")
                    else:
                        print(f"\n❌ 未找到深度列")
                        print(f"   列名中不包含'depth'、'深度'等关键词")
                        print()
                        print("⚠️ 请检查:")
                        print("   1. 深度列是否有其他名称？")
                        print("   2. 深度信息是否在其他文件中？")
                    
                    break
                except Exception as e:
                    continue
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
    else:
        print("❌ 未找到TXT文件")
else:
    print("❌ 未找到井目录")

print()
print("=" * 80)
print("📊 当前状况总结")
print("=" * 80)
print()

print("✅ 已确认的问题:")
print("   - 训练数据(awpd_final_correct_data)中没有深度信息")
print("   - 这导致模型无法对齐同类样本")
print("   - 导致验证准确率卡在30-36%")
print()

print("💡 解决方案:")
if original_data_dir and len(wells) > 0:
    print("   ✅ 原始数据存在，可以提取深度信息")
    print("   📋 下一步：修改数据生成脚本")
else:
    print("   ⚠️ 需要先找到原始数据")
    print("   📋 下一步：确认原始数据位置")

print()
print("=" * 80)


