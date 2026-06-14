#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
創建平衡的井劃分方案
確保訓練/驗證/測試集都包含全部5類
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import argparse

def load_well_distributions():
    """加載井分布分析結果"""
    json_file = Path('well_distribution_analysis.json')
    if not json_file.exists():
        raise FileNotFoundError("請先運行 analyze_all_well_distributions.py")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def score_well_diversity(well_info):
    """評分井的多樣性"""
    score = 0
    
    # 類別覆蓋得分（0-50分）
    score += well_info['unique_classes'] * 10
    
    # 平衡性得分（0-30分）
    # 主導類越少，平衡性越好
    balance_score = 30 * (1 - well_info['dominant_pct'] / 100)
    score += balance_score
    
    # 樣本量得分（0-20分）
    sample_score = min(20, well_info['total'] / 100)
    score += sample_score
    
    return score

def create_balanced_split(well_distributions, train_wells=10, val_wells=5, test_wells=3):
    """創建平衡的井劃分"""
    
    print("="*80)
    print("🔬 創建平衡井劃分方案")
    print("="*80)
    
    # 分離不同split的井
    all_wells = {}
    for key, info in well_distributions.items():
        well_name = info['well']
        if well_name not in all_wells:
            all_wells[well_name] = info
    
    # 計算每口井的得分
    well_scores = {}
    for well_name, info in all_wells.items():
        score = score_well_diversity(info)
        well_scores[well_name] = score
        print(f"\n{well_name}:")
        print(f"   得分: {score:.1f}")
        print(f"   類別數: {info['unique_classes']}/5")
        print(f"   樣本數: {info['total']}")
        print(f"   主導類: {info['dominant_pct']:.1f}%")
    
    # 按得分排序
    sorted_wells = sorted(well_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*80}")
    print("📊 井排名（按多樣性得分）")
    print(f"{'='*80}")
    for rank, (well, score) in enumerate(sorted_wells, 1):
        diversity = all_wells[well]['unique_classes']
        print(f"{rank:2d}. {well:10s} - 得分:{score:5.1f}, 類別:{diversity}/5")
    
    # 智能分配
    print(f"\n{'='*80}")
    print("🎯 智能分配井")
    print(f"{'='*80}")
    
    # 優先選擇高得分井作為驗證集（確保驗證更可靠）
    val_candidates = [w for w, s in sorted_wells[:12]]  # 前12名候選
    test_candidates = [w for w, s in sorted_wells if w not in val_candidates]
    
    # 貪心算法：選擇能最大化類別覆蓋的組合
    def find_best_combination(candidates, n, must_have_classes=5):
        """找到最佳的n口井組合，確保覆蓋must_have_classes類"""
        from itertools import combinations
        
        best_combo = None
        best_score = -1
        best_coverage = 0
        
        for combo in combinations(candidates, n):
            # 計算覆蓋的類別
            covered_classes = set()
            total_score = 0
            for well in combo:
                info = all_wells[well]
                for class_id, class_info in info['distribution'].items():
                    if class_info['count'] > 0:
                        covered_classes.add(int(class_id))
                total_score += well_scores[well]
            
            coverage = len(covered_classes)
            
            # 優先選擇覆蓋5類的組合
            if coverage > best_coverage or (coverage == best_coverage and total_score > best_score):
                best_coverage = coverage
                best_score = total_score
                best_combo = combo
        
        return best_combo, best_coverage
    
    # 先選驗證集
    print(f"\n選擇驗證井（目標{val_wells}口）...")
    val_selected, val_coverage = find_best_combination(val_candidates, val_wells)
    val_selected = list(val_selected)
    print(f"   選中: {val_selected}")
    print(f"   類別覆蓋: {val_coverage}/5")
    
    # 再選測試集
    remaining = [w for w in test_candidates if w not in val_selected]
    print(f"\n選擇測試井（目標{test_wells}口）...")
    test_selected, test_coverage = find_best_combination(remaining, test_wells)
    test_selected = list(test_selected)
    print(f"   選中: {test_selected}")
    print(f"   類別覆蓋: {test_coverage}/5")
    
    # 剩餘作為訓練集
    train_selected = [w for w in all_wells.keys() if w not in val_selected and w not in test_selected]
    print(f"\n訓練井（共{len(train_selected)}口）:")
    print(f"   {train_selected}")
    
    # 檢查訓練集覆蓋
    train_coverage = set()
    for well in train_selected:
        info = all_wells[well]
        for class_id, class_info in info['distribution'].items():
            if class_info['count'] > 0:
                train_coverage.add(int(class_id))
    print(f"   類別覆蓋: {len(train_coverage)}/5")
    
    # 生成劃分方案
    split_plan = {
        'train': sorted(train_selected),
        'val': sorted(val_selected),
        'test': sorted(test_selected),
        'statistics': {
            'train_wells': len(train_selected),
            'val_wells': len(val_selected),
            'test_wells': len(test_selected),
            'train_coverage': len(train_coverage),
            'val_coverage': val_coverage,
            'test_coverage': test_coverage
        }
    }
    
    # 保存方案
    output_file = Path('balanced_well_split.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(split_plan, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"💾 劃分方案已保存: {output_file}")
    print(f"{'='*80}")
    
    # 同時生成文本列表文件（用於generate_split_spectrograms.py）
    for split in ['train', 'val', 'test']:
        txt_file = Path(f'optimized_patch_data/{split}_wells.txt')
        txt_file.parent.mkdir(exist_ok=True)
        with open(txt_file, 'w', encoding='utf-8') as f:
            for well in split_plan[split]:
                f.write(f"{well}\n")
        print(f"   生成: {txt_file}")
    
    # 總結
    print(f"\n{'='*80}")
    print("📋 劃分方案總結")
    print(f"{'='*80}")
    print(f"\n訓練集: {len(train_selected)}口井, 覆蓋{len(train_coverage)}/5類")
    print(f"驗證集: {len(val_selected)}口井, 覆蓋{val_coverage}/5類 {'✅' if val_coverage == 5 else '⚠️'}")
    print(f"測試集: {len(test_selected)}口井, 覆蓋{test_coverage}/5類 {'✅' if test_coverage == 5 else '⚠️'}")
    
    if val_coverage == 5 and test_coverage == 5:
        print(f"\n✅ 劃分成功！驗證集和測試集都包含全部5類")
    else:
        print(f"\n⚠️ 建議調整井數量，確保驗證集和測試集都包含全部5類")
    
    return split_plan

def main():
    parser = argparse.ArgumentParser(description='創建平衡的井劃分方案')
    parser.add_argument('--train-wells', type=int, default=10, help='訓練井數量')
    parser.add_argument('--val-wells', type=int, default=5, help='驗證井數量')
    parser.add_argument('--test-wells', type=int, default=3, help='測試井數量')
    
    args = parser.parse_args()
    
    # 加載井分布
    well_distributions = load_well_distributions()
    
    # 創建劃分
    split_plan = create_balanced_split(
        well_distributions,
        train_wells=args.train_wells,
        val_wells=args.val_wells,
        test_wells=args.test_wells
    )
    
    print(f"\n🎯 下一步:")
    print(f"   1. 檢查劃分方案: balanced_well_split.json")
    print(f"   2. 重新生成數據: python main/generate_split_spectrograms.py")
    print(f"   3. 重新訓練: python main/run_final_optimized_training.py")

if __name__ == '__main__':
    main()


