#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一优化数据加载器
支持智能子图谱加载和PCA降维特征
专门处理256×256完整时频图谱和32×32优化子图谱
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split

class UnifiedOptimizedDataset(Dataset):
    """统一优化时频图谱数据集
    支持完整图谱和智能子图谱的混合训练
    """
    
    def __init__(self, cache_dir: str = "optimized_patch_data", 
                 indices: Optional[List[int]] = None,
                 mode: str = "mixed",  # "full", "patch", "mixed"
                 patch_probability: float = 0.3,
                 augment: bool = True,
                 use_pca_features: bool = False):
        """
        初始化统一优化数据集
        
        Args:
            cache_dir: 缓存目录路径
            indices: 要加载的样本索引列表
            mode: 加载模式
            patch_probability: 混合模式下子图谱的概率
            augment: 是否启用数据增强
            use_pca_features: 是否使用PCA特征
        """
        self.cache_dir = Path(cache_dir)
        self.spectrograms_dir = self.cache_dir / "spectrograms"
        self.metadata_dir = self.cache_dir / "metadata"
        self.pca_dir = self.cache_dir / "pca_models"
        self.mode = mode
        self.patch_probability = patch_probability
        self.augment = augment
        self.use_pca_features = use_pca_features
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 加载PCA模型（如果需要）
        self.pca_models = {}
        self.scalers = {}
        if use_pca_features:
            self._load_pca_models()
        
        # 分类样本文件
        self.full_samples, self.patch_samples = self._categorize_samples(indices)
        
        # 根据模式设置样本列表
        if mode == "full":
            self.sample_files = self.full_samples
        elif mode == "patch":
            self.sample_files = self.patch_samples
        else:  # mixed
            # 混合模式：为每个完整图谱配对少量子图谱，避免数据不平衡
            matched_patches = []
            for full_file in self.full_samples:
                # 从完整图谱文件名提取基础索引
                parts = full_file.stem.split('_')
                if len(parts) >= 2:
                    base_idx = parts[1]
                    # 查找对应的子图谱（每个完整图谱配对3个子图谱）
                    matching_patches = [p for p in self.patch_samples if f"sample_{base_idx}_patch" in p.stem]
                    matched_patches.extend(matching_patches[:3])
            
            self.sample_files = self.full_samples + matched_patches
        
        print(f"📊 统一优化数据集加载完成:")
        print(f"   模式: {mode}")
        print(f"   完整图谱(256×256): {len(self.full_samples)}")
        if mode == "mixed":
            patch_count = len(self.sample_files) - len(self.full_samples)
            print(f"   优化子图谱(32×32): {patch_count} (匹配的)")
        else:
            print(f"   优化子图谱(32×32): {len(self.patch_samples)}")
        print(f"   总样本: {len(self.sample_files)}")
        print(f"   PCA特征: {'启用' if use_pca_features else '禁用'}")
        
        # 验证数据
        self._verify_samples()
    
    def _load_metadata(self) -> Dict:
        """加载缓存元数据"""
        metadata_file = self.metadata_dir / "cache_metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"缓存元数据文件不存在: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        return metadata
    
    def _load_pca_models(self):
        """加载PCA模型"""
        try:
            pca_files = list(self.pca_dir.glob("*_pca.pkl"))
            for pca_file in pca_files:
                channel_key = pca_file.stem.replace('_pca', '')
                scaler_file = self.pca_dir / f"{channel_key}_scaler.pkl"
                
                if scaler_file.exists():
                    with open(pca_file, 'rb') as f:
                        self.pca_models[channel_key] = pickle.load(f)
                    
                    with open(scaler_file, 'rb') as f:
                        self.scalers[channel_key] = pickle.load(f)
            
            print(f"   加载了 {len(self.pca_models)} 个PCA模型")
            
        except Exception as e:
            print(f"   ⚠️ PCA模型加载失败: {e}")
            self.use_pca_features = False
    
    def _categorize_samples(self, indices: Optional[List[int]]) -> Tuple[List[Path], List[Path]]:
        """分类样本文件 - 簡化邏輯，patch模式下直接使用所有子圖譜"""
        all_sample_files = sorted(list(self.spectrograms_dir.glob("sample_*.npz")))
        
        full_samples = []
        patch_samples = []
        
        for file_path in all_sample_files:
            try:
                if "_full_256x256" in file_path.stem:
                    if indices is None:
                        full_samples.append(file_path)
                    else:
                        parts = file_path.stem.split('_')
                        if len(parts) >= 2:
                            sample_idx = int(parts[1])
                            if sample_idx in indices:
                                full_samples.append(file_path)
                
                elif "_patch_32x32" in file_path.stem:
                    if indices is None:
                        patch_samples.append(file_path)
                    else:
                        # 改進策略：基於完整圖譜的sample_idx來匹配對應的子圖譜
                        # 只在訓練時進行精確匹配，驗證時採用寬鬆策略避免空集
                        try:
                            data = np.load(file_path)
                            sample_idx = int(data['sample_idx'])
                            data.close()
                            
                            # 檢查該sample_idx是否有對應的完整圖譜在indices中
                            # 由於完整圖譜和子圖譜的sample_idx可能不完全對應，採用範圍匹配
                            if any(abs(sample_idx - idx) <= 5 for idx in indices):  # 允許一定容差
                                patch_samples.append(file_path)
                        except Exception:
                            # 如果讀取失敗，採用文件名策略
                            parts = file_path.stem.split('_')
                            if len(parts) >= 2:
                                try:
                                    file_idx = int(parts[1])
                                    if file_idx < max(indices) * 15:  # 更保守的估計
                                        patch_samples.append(file_path)
                                except Exception:
                                    continue
                    
            except Exception as e:
                print(f"   ⚠️ 跳过损坏的文件: {file_path}, 错误: {e}")
        
        return full_samples, patch_samples
    
    def _verify_samples(self):
        """验证样本数据"""
        if len(self.sample_files) == 0:
            raise ValueError("没有找到有效的样本文件")
        
        # 随机检查几个样本
        check_count = min(5, len(self.sample_files))
        check_files = random.sample(self.sample_files, check_count)
        
        for file_path in check_files:
            try:
                data = np.load(file_path)
                spectrogram = data['spectrogram']
                label = data['label']
                
                # 基本验证
                if isinstance(label, np.ndarray):
                    label = label.item()  # 从numpy数组中提取标量
                if not isinstance(label, (int, np.integer, float)):
                    raise ValueError(f"标签类型错误: {type(label)}")
                
                if spectrogram.ndim not in [3, 4]:
                    raise ValueError(f"时频图谱维度错误: {spectrogram.ndim}")
                
            except Exception as e:
                raise RuntimeError(f"样本文件验证失败 {file_path}: {e}")
        
        print(f"✅ 样本验证通过 (检查了 {check_count} 个样本)")
    
    def _augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """数据增强"""
        if not self.augment:
            return spectrogram
        
        try:
            # 随机翻转
            if random.random() < 0.4:
                spectrogram = np.flip(spectrogram, axis=-1).copy()  # 水平翻转並複製以避免負步長
            
            # 随机噪声（轻微）
            if random.random() < 0.2:
                noise_factor = 0.005
                noise = np.random.normal(0, noise_factor, spectrogram.shape)
                spectrogram = spectrogram + noise
            
            # 随机缩放（轻微）
            if random.random() < 0.2:
                scale_factor = random.uniform(0.98, 1.02)
                spectrogram = spectrogram * scale_factor
            
            return spectrogram
            
        except Exception:
            return spectrogram  # 增强失败则返回原始数据
    
    def _apply_pca_transform(self, spectrogram: np.ndarray, channel_idx: int) -> np.ndarray:
        """应用PCA变换"""
        if not self.use_pca_features:
            return spectrogram
        
        try:
            channel_key = f"channel_{channel_idx}"
            
            if channel_key in self.pca_models and channel_key in self.scalers:
                # 展平时频图谱：(1, 32, 32) -> (1024,)
                original_shape = spectrogram.shape
                flat_spectrogram = spectrogram.flatten().reshape(1, -1)
                
                # 应用标准化和PCA
                scaler = self.scalers[channel_key]
                pca = self.pca_models[channel_key]
                
                # 標準化
                scaled_features = scaler.transform(flat_spectrogram)
                # PCA降維
                pca_features = pca.transform(scaled_features)
                
                # 固定PCA特徵圖譜尺寸為8x8，確保一致性
                n_components = pca_features.shape[1]
                fixed_size = 8  # 固定8x8尺寸
                
                # 創建固定尺寸的矩陣
                pca_matrix = np.zeros((fixed_size, fixed_size))
                pca_flat = pca_features.flatten()
                
                # 填充或截斷PCA特徵
                max_fill = min(len(pca_flat), fixed_size * fixed_size)
                pca_matrix.flat[:max_fill] = pca_flat[:max_fill]
                
                # 返回固定尺寸的單通道PCA特徵圖譜
                return pca_matrix[np.newaxis, :, :]  # (1, 8, 8)
            else:
                return spectrogram
                
        except Exception:
            return spectrogram
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """获取单个样本"""
        if idx >= len(self.sample_files):
            raise IndexError(f"索引超出范围: {idx} >= {len(self.sample_files)}")
        
        # 在混合模式下，根据概率选择样本类型
        if self.mode == "mixed":
            if random.random() < self.patch_probability and len(self.patch_samples) > 0:
                # 选择子图谱样本
                file_path = random.choice(self.patch_samples)
            elif len(self.full_samples) > 0:
                # 选择完整图谱样本
                file_path = random.choice(self.full_samples)
            else:
                # 回退到原始索引
                file_path = self.sample_files[idx]
        else:
            file_path = self.sample_files[idx]
        
        # 加载样本数据
        data = np.load(file_path)
        
        spectrogram = data['spectrogram']
        label = data['label']
        sample_type = str(data.get('type', 'unknown'))
        well_name = str(data['well_name'])
        
        # 数据增强
        spectrogram = self._augment_spectrogram(spectrogram)
        
        # 应用PCA变换（如果启用）
        if self.use_pca_features and sample_type == 'patch':
            channel_idx = data.get('channel_idx', 0)
            spectrogram = self._apply_pca_transform(spectrogram, channel_idx)
        
        # 处理标签格式
        if isinstance(label, np.ndarray):
            label = label.item()  # 从numpy数组中提取标量
        
        # 确保数组连续性（修复负步长问题）
        if not spectrogram.flags['C_CONTIGUOUS']:
            spectrogram = np.ascontiguousarray(spectrogram)
        
        # 转换为张量
        spectrogram_tensor = torch.FloatTensor(spectrogram)
        label_tensor = torch.LongTensor([int(label)])
        
        # 元数据
        metadata = {
            'well_name': well_name,
            'sample_type': sample_type,
            'file_path': str(file_path)
        }
        
        # 添加子图谱特有信息
        if sample_type == 'patch':
            metadata.update({
                'channel_idx': data.get('channel_idx', 0),
                'patch_idx': data.get('patch_idx', 0),
                'position': data.get('position', (0, 0)),
                'energy': data.get('energy', 0.0),
                'diversity_score': data.get('diversity_score', 0.0)
            })
        
        return spectrogram_tensor, label_tensor.squeeze(), metadata
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        class_counts = {}
        
        for file_path in self.sample_files:
            try:
                data = np.load(file_path)
                label = int(data['label'])
                class_counts[label] = class_counts.get(label, 0) + 1
            except Exception:
                continue
        
        return class_counts
    
    def get_sample_type_distribution(self) -> Dict[str, int]:
        """获取样本类型分布"""
        type_counts = {}
        
        for file_path in self.sample_files:
            try:
                data = np.load(file_path)
                sample_type = str(data.get('type', 'unknown'))
                type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
            except Exception:
                continue
        
        return type_counts

def create_unified_data_loaders(
    cache_dir: str = "optimized_patch_data",
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    test_ratio: float = 0.0,
    mode: str = "mixed",
    patch_probability: float = 0.3,
    use_pca_features: bool = False,
    use_stratified_split: bool = True,
    random_state: int = 42,
    num_workers: int = 0,
    augment_train: bool = True
) -> Dict[str, DataLoader]:
    """
    创建统一优化数据加载器
    """
    print("🚀 创建统一优化数据加载器")
    print("=" * 60)
    
    # 验证缓存存在
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"缓存目录不存在: {cache_path}")
    
    # 获取所有样本索引和标签（包括完整图谱和子图谱）
    spectrograms_dir = cache_path / "spectrograms"
    full_files = sorted(list(spectrograms_dir.glob("sample_*_full_256x256.npz")))
    patch_files = sorted(list(spectrograms_dir.glob("sample_*_patch_32x32_*.npz")))
    
    print(f"   发现完整图谱: {len(full_files)} 个")
    print(f"   发现子图谱: {len(patch_files)} 个")
    
    all_indices = []
    all_labels = []
    
    print(f"📊 扫描缓存样本...")
    
    # 使用完整图谱作为基础样本索引（因为子图谱是基于完整图谱生成的）
    for file_path in full_files:
        try:
            data = np.load(file_path)
            
            # 从文件名提取索引
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                idx = int(parts[1])
                label = int(data['label'])
                
                all_indices.append(idx)
                all_labels.append(label)
                
        except Exception as e:
            print(f"   ⚠️ 跳过损坏的文件: {file_path}, 错误: {e}")
    
    total_samples = len(all_indices)
    print(f"   基础样本索引: {total_samples} 个")
    print(f"   总可用样本: {len(full_files) + len(patch_files)} 个")
    
    if total_samples == 0:
        raise ValueError("没有找到有效的缓存样本")
    
    # 数据分割
    print(f"📊 数据分割 (训练:{train_ratio:.1f}, 验证:{val_ratio:.1f}, 测试:{test_ratio:.1f})")
    
    # 设置随机种子
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)
    
    if use_stratified_split and len(set(all_labels)) > 1:
        print("   使用分层抽样分割")
        try:
            if test_ratio > 0:
                train_indices, temp_indices, train_labels, temp_labels = train_test_split(
                    all_indices, all_labels, 
                    test_size=(val_ratio + test_ratio),
                    stratify=all_labels,
                    random_state=random_state
                )
                
                val_indices, test_indices, _, _ = train_test_split(
                    temp_indices, temp_labels,
                    test_size=test_ratio/(val_ratio + test_ratio),
                    stratify=temp_labels,
                    random_state=random_state
                )
            else:
                train_indices, val_indices, _, _ = train_test_split(
                    all_indices, all_labels,
                    test_size=val_ratio,
                    stratify=all_labels,
                    random_state=random_state
                )
                test_indices = []
                
        except ValueError as e:
            print(f"   ⚠️ 分层抽样失败，使用随机分割: {e}")
            use_stratified_split = False
    
    if not use_stratified_split:
        print("   使用随机分割")
        indices = np.array(all_indices)
        np.random.shuffle(indices)
        
        train_size = int(len(indices) * train_ratio)
        val_size = int(len(indices) * val_ratio)
        
        train_indices = indices[:train_size].tolist()
        val_indices = indices[train_size:train_size + val_size].tolist()
        test_indices = indices[train_size + val_size:].tolist() if test_ratio > 0 else []
    
    # 创建数据集
    print(f"📦 创建统一优化数据集...")
    train_dataset = UnifiedOptimizedDataset(
        cache_dir, train_indices, mode=mode, 
        patch_probability=patch_probability, augment=augment_train,
        use_pca_features=use_pca_features
    )
    val_dataset = UnifiedOptimizedDataset(
        cache_dir, val_indices, mode=mode, 
        patch_probability=patch_probability, augment=False,
        use_pca_features=use_pca_features
    )
    
    data_loaders = {}
    
    # 训练数据加载器
    data_loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    # 验证数据加载器
    data_loaders['val'] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    # 测试数据加载器（如果需要）
    if test_ratio > 0 and len(test_indices) > 0:
        test_dataset = UnifiedOptimizedDataset(
            cache_dir, test_indices, mode=mode, 
            patch_probability=patch_probability, augment=False,
            use_pca_features=use_pca_features
        )
        data_loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        print(f"   测试集: {len(test_dataset)} 样本")
    
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本")
    
    # 显示类别分布
    print("\n📊 训练集类别分布:")
    train_class_dist = train_dataset.get_class_distribution()
    for class_id, count in sorted(train_class_dist.items()):
        from data.fluid_types import FluidTypes
        class_name = FluidTypes.FLUID_TYPES.get(class_id, f"Class_{class_id}")
        percentage = count / len(train_dataset) * 100
        print(f"   {class_name}: {count} 样本 ({percentage:.1f}%)")
    
    # 显示样本类型分布
    print("\n📊 训练集样本类型分布:")
    type_dist = train_dataset.get_sample_type_distribution()
    for sample_type, count in type_dist.items():
        percentage = count / len(train_dataset) * 100
        print(f"   {sample_type}: {count} 样本 ({percentage:.1f}%)")
    
    print("✅ 统一优化数据加载器创建完成")
    return data_loaders

if __name__ == "__main__":
    # 测试统一优化数据加载器
    try:
        # 创建数据加载器
        data_loaders = create_unified_data_loaders(
            batch_size=8,
            mode="mixed",
            patch_probability=0.3,
            use_pca_features=False,
            train_ratio=0.7,
            val_ratio=0.3,
            use_stratified_split=True
        )
        
        # 测试加载
        print("\n🧪 测试数据加载...")
        train_loader = data_loaders['train']
        
        for batch_idx, (spectrograms, labels, metadata) in enumerate(train_loader):
            print(f"   批次 {batch_idx}: 时频图谱 {spectrograms.shape}, 标签 {labels.shape}")
            print(f"   样本类型: {[m['sample_type'] for m in metadata]}")
            if batch_idx >= 2:
                break
        
        print("✅ 统一优化数据加载器测试通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
