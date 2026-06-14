#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMOTE數據增強器 - 解決缺失類別問題
為缺失的水層(1)和油水層(4)類別生成合成數據
"""

import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import random
from sklearn.neighbors import NearestNeighbors

class SMOTEDataAugmenter:
    """SMOTE數據增強器"""
    
    def __init__(self, cache_dir: str = "optimized_patch_data"):
        self.cache_dir = Path(cache_dir)
        self.spectrograms_dir = self.cache_dir / "spectrograms"
        
        # 5類標籤映射
        self.class_names = {
            0: "油層", 1: "水層", 2: "干層", 
            3: "差油層", 4: "油水層"
        }
        
        # 目標類別分佈
        self.target_distribution = {
            0: 0.25,  # 油層: 25%
            1: 0.15,  # 水層: 15%
            2: 0.35,  # 干層: 35%
            3: 0.15,  # 差油層: 15%
            4: 0.10   # 油水層: 10%
        }
    
    def analyze_current_distribution(self) -> Dict[int, int]:
        """分析當前類別分佈"""
        print("📊 分析當前數據分佈...")
        
        # 掃描所有子圖譜文件（確保只統計子圖譜）
        patch_files = list(self.spectrograms_dir.glob("*_patch_32x32_*.npz"))
        labels = []
        
        print(f"   找到 {len(patch_files)} 個子圖譜文件")
        
        for file_path in patch_files:
            try:
                data = np.load(file_path)
                label = int(data['label'])
                labels.append(label)
            except Exception:
                continue
        
        distribution = Counter(labels)
        total = len(labels)
        
        print(f"📋 當前分佈 (共 {total} 個樣本):")
        for class_id in range(5):
            count = distribution.get(class_id, 0)
            percentage = count / total * 100 if total > 0 else 0
            print(f"   {self.class_names[class_id]}: {count} ({percentage:.1f}%)")
        
        return distribution
    
    def generate_synthetic_samples(self, 
                                 source_class: int, 
                                 target_class: int, 
                                 n_samples: int,
                                 k_neighbors: int = 5) -> List[np.ndarray]:
        """使用SMOTE生成合成樣本"""
        print(f"🔬 從{self.class_names[source_class]}生成{n_samples}個{self.class_names[target_class]}樣本...")
        
        # 收集源類別的所有樣本
        source_samples = []
        patch_files = list(self.spectrograms_dir.glob("*_patch_32x32_*.npz"))
        
        for file_path in patch_files:
            try:
                data = np.load(file_path)
                label = int(data['label'])
                if label == source_class:
                    spectrogram = data['spectrogram']
                    source_samples.append(spectrogram.flatten())
            except Exception:
                continue
        
        if len(source_samples) < k_neighbors:
            print(f"   ⚠️ 源樣本數量不足: {len(source_samples)} < {k_neighbors}")
            k_neighbors = max(1, len(source_samples))
        
        source_samples = np.array(source_samples)
        
        # 使用KNN找到最近鄰
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(source_samples)
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # 隨機選擇一個源樣本
            idx = random.randint(0, len(source_samples) - 1)
            sample = source_samples[idx]
            
            # 找到最近鄰
            _, neighbors_idx = nn.kneighbors([sample])
            neighbor_idx = random.choice(neighbors_idx[0][1:])  # 排除自己
            neighbor = source_samples[neighbor_idx]
            
            # SMOTE插值
            alpha = random.random()
            synthetic = sample + alpha * (neighbor - sample)
            
            # 添加輕微擾動以增加多樣性
            noise = np.random.normal(0, 0.01, synthetic.shape)
            synthetic += noise
            
            # 重新整形為時頻圖譜格式
            synthetic_spectrogram = synthetic.reshape(1, 32, 32)
            synthetic_samples.append(synthetic_spectrogram)
        
        return synthetic_samples
    
    def save_synthetic_samples(self, 
                             samples: List[np.ndarray], 
                             target_class: int, 
                             base_well_name: str = "synthetic") -> int:
        """保存合成樣本"""
        saved_count = 0
        
        for i, spectrogram in enumerate(samples):
            try:
                # 生成文件名
                filename = f"sample_synthetic_{target_class}_{i:06d}_patch_32x32_0_0.npz"
                file_path = self.spectrograms_dir / filename
                
                # 保存數據
                np.savez_compressed(
                    file_path,
                    spectrogram=spectrogram,
                    label=target_class,
                    well_name=base_well_name,
                    sample_idx=999900 + i,  # 使用特殊索引避免衝突
                    window_start=0,
                    window_end=32,
                    channel_idx=0,
                    patch_idx=0,
                    position=[0, 0],
                    energy=np.mean(spectrogram),
                    variance=np.var(spectrogram),
                    gradient=np.std(np.gradient(spectrogram)),
                    diversity_score=1.0,
                    type='synthetic_patch'
                )
                
                saved_count += 1
                
            except Exception as e:
                print(f"   ⚠️ 保存失敗: {e}")
                continue
        
        return saved_count
    
    def augment_missing_classes(self) -> Dict[int, int]:
        """增強缺失類別"""
        print("🚀 開始增強缺失類別...")
        
        # 分析當前分佈
        current_distribution = self.analyze_current_distribution()
        total_current = sum(current_distribution.values())
        
        augment_results = {}
        
        # 為缺失的水層生成數據
        if current_distribution.get(1, 0) == 0:
            print("\n🌊 生成水層數據...")
            target_count = int(total_current * self.target_distribution[1])
            synthetic_samples = self.generate_synthetic_samples(
                source_class=0,  # 從油層生成
                target_class=1,  # 水層
                n_samples=target_count
            )
            saved_count = self.save_synthetic_samples(synthetic_samples, 1, "synthetic_water")
            augment_results[1] = saved_count
            print(f"   ✅ 成功生成 {saved_count} 個水層樣本")
        
        # 為缺失的油水層生成數據
        if current_distribution.get(4, 0) == 0:
            print("\n🌊 生成油水層數據...")
            target_count = int(total_current * self.target_distribution[4])
            synthetic_samples = self.generate_synthetic_samples(
                source_class=0,  # 從油層生成
                target_class=4,  # 油水層
                n_samples=target_count
            )
            saved_count = self.save_synthetic_samples(synthetic_samples, 4, "synthetic_oil_water")
            augment_results[4] = saved_count
            print(f"   ✅ 成功生成 {saved_count} 個油水層樣本")
        
        return augment_results
    
    def balance_all_classes(self) -> Dict[int, int]:
        """平衡所有類別"""
        print("⚖️ 平衡所有類別分佈...")
        
        # 分析當前分佈
        current_distribution = self.analyze_current_distribution()
        total_current = sum(current_distribution.values())
        
        # 計算目標數量
        target_total = max(total_current, 10000)  # 至少10k樣本
        balance_results = {}
        
        for class_id in range(5):
            current_count = current_distribution.get(class_id, 0)
            target_count = int(target_total * self.target_distribution[class_id])
            
            if current_count < target_count:
                need_count = target_count - current_count
                print(f"\n📈 增強{self.class_names[class_id]}：{current_count} -> {target_count}")
                
                if class_id in [1, 4] and current_count == 0:
                    # 缺失類別從相似類別生成
                    source_class = 0  # 從油層生成
                else:
                    source_class = class_id  # 從自身生成
                
                synthetic_samples = self.generate_synthetic_samples(
                    source_class=source_class,
                    target_class=class_id,
                    n_samples=need_count
                )
                
                saved_count = self.save_synthetic_samples(
                    synthetic_samples, class_id, f"balanced_{self.class_names[class_id]}"
                )
                balance_results[class_id] = saved_count
                print(f"   ✅ 成功增強 {saved_count} 個樣本")
        
        return balance_results

def main():
    """主函數"""
    print("🎯 SMOTE數據增強器")
    print("解決缺失類別和類別不平衡問題")
    print("=" * 50)
    
    augmenter = SMOTEDataAugmenter()
    
    # 第一步：增強缺失類別
    missing_results = augmenter.augment_missing_classes()
    
    # 第二步：平衡所有類別
    balance_results = augmenter.balance_all_classes()
    
    # 最終分析
    print("\n📊 最終數據分佈:")
    final_distribution = augmenter.analyze_current_distribution()
    
    print("\n🎉 數據增強完成！")
    print(f"   缺失類別增強: {missing_results}")
    print(f"   類別平衡結果: {balance_results}")

if __name__ == "__main__":
    main()
