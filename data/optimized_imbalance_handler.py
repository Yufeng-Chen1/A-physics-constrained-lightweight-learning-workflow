#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的类别不平衡处理方案
针对极不平衡分布 (5:1) 的BSMOTE + 类别权重优化策略
"""

import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.decomposition import PCA

try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")


class BalancedTripletDataset(torch.utils.data.Dataset):
    """顶层定义的数据集类，避免Windows多进程pickle错误"""
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = True):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            # 轻量增强：水平翻转/微噪声/强度缩放
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            if torch.rand(1).item() < 0.2:
                scale = 0.95 + 0.10 * torch.rand(1).item()
                x = x * scale
        return x, self.y[idx], {}

class OptimizedImbalanceHandler:
    """优化的类别不平衡处理器"""
    
    def __init__(self, 
                 method: str = 'bsmote',
                 target_ratio: float = 0.7,
                 k_neighbors: int = 3,
                 random_state: int = 42,
                 enable_pca: bool = True,
                 pca_components: int = 1000,
                 pca_variance_ratio: float = 0.95,
                 max_weight: float = 5.0,
                 min_weight: float = 0.5,
                 directed_enhance: bool = False,
                 target_percent_per_class: Optional[Dict[int, float]] = None,
                 class_neighbor_overrides: Optional[Dict[int, int]] = None,
                 cap_minor_vs_major: float = 0.5,
                 kind: str = 'borderline-1'):
        """
        初始化优化的不平衡处理器
        
        Args:
            method: 采样方法 ('bsmote', 'smote', 'adasyn', 'smoteenn')
            target_ratio: 目标平衡比例
            k_neighbors: 近邻数量
            random_state: 随机种子
            enable_pca: 是否启用PCA降维
            pca_components: PCA组件数
            pca_variance_ratio: PCA保留方差比例
            max_weight: 最大类别权重
            min_weight: 最小类别权重
        """
        self.method = method
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.enable_pca = enable_pca
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.max_weight = max_weight
        self.min_weight = min_weight
        # 定向增强配置
        self.directed_enhance = directed_enhance
        self.target_percent_per_class = target_percent_per_class or {4: 0.20}  # 仅增强油水层≈20%
        self.class_neighbor_overrides = class_neighbor_overrides or {4: 10}
        self.cap_minor_vs_major = cap_minor_vs_major
        self.kind = kind
        
        # 流体类型名称（项目统一为5类）
        # 说明：致密油层并入“油层”；含油水层/含水油层/油水同层统一为“油水层”
        self.fluid_names = ['油层', '水层', '干层', '差油层', '油水层']
        
        # 存储状态
        self.pca_ = None
        self.class_weights_ = None
        self.class_distribution_ = None
        
    def analyze_class_distribution(self, labels: np.ndarray) -> Dict:
        """分析类别分布"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        distribution = {}
        for class_id in range(len(self.fluid_names)):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_samples) * 100
            distribution[class_id] = {
                'count': count,
                'percentage': percentage,
                'name': self.fluid_names[class_id]
            }
        
        self.class_distribution_ = distribution
        return distribution
    
    def calculate_optimized_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """计算优化的类别权重"""
        distribution = self.analyze_class_distribution(labels)
        
        # 使用平滑平方根权重策略
        max_count = max(info['count'] for info in distribution.values())
        class_weights = {}
        
        for class_id, info in distribution.items():
            if info['count'] > 0:
                # 平滑平方根权重
                weight = np.sqrt(max_count / info['count'])
                # 限制权重范围
                weight = np.clip(weight, self.min_weight, self.max_weight)
                class_weights[class_id] = weight
            else:
                class_weights[class_id] = 1.0
        
        self.class_weights_ = class_weights
        
        print(f"⚖️  优化类别权重计算:")
        for class_id, weight in class_weights.items():
            info = distribution[class_id]
            print(f"   类别{class_id} ({info['name']}): {weight:.3f} (样本数: {info['count']})")
        
        return class_weights
    
    def apply_pca_if_needed(self, X: np.ndarray) -> Tuple[np.ndarray, bool]:
        """根据需要应用PCA降维"""
        if not self.enable_pca:
            return X, False
        
        # 检查是否需要PCA
        if X.shape[1] <= self.pca_components:
            print(f"🔧 特征维度 {X.shape[1]} <= {self.pca_components}，跳过PCA")
            return X, False
        
        print(f"🔧 应用PCA降维: {X.shape[1]} -> {self.pca_components}")
        
        try:
            # 计算保留指定方差比例所需的组件数
            pca_temp = PCA(random_state=self.random_state)
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components_variance = np.argmax(cumsum >= self.pca_variance_ratio) + 1
            
            # 选择较小的组件数
            n_components = min(self.pca_components, n_components_variance, X.shape[0], X.shape[1])
            
            if n_components >= X.shape[1]:
                print(f"🔧 PCA组件数 {n_components} >= 原始特征数 {X.shape[1]}，跳过PCA")
                return X, False
            
            # 应用PCA
            self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
            X_pca = self.pca_.fit_transform(X)
            
            explained_variance = np.sum(self.pca_.explained_variance_ratio_)
            print(f"✅ PCA完成: {X.shape[1]} -> {X_pca.shape[1]} (解释方差: {explained_variance:.3f})")
            
            return X_pca, True
            
        except Exception as e:
            print(f"⚠️  PCA失败，使用原始特征: {e}")
            return X, False
    
    def inverse_pca_transform(self, X_pca: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """PCA逆变换"""
        if self.pca_ is None:
            return X_pca.reshape(original_shape)
        
        try:
            X_original = self.pca_.inverse_transform(X_pca)
            # 确保形状正确
            if X_original.shape[1] == np.prod(original_shape):
                return X_original.reshape(-1, *original_shape)
            else:
                print(f"⚠️  PCA逆变换形状不匹配: {X_original.shape} vs {(-1, *original_shape)}")
                return X_pca.reshape(-1, *original_shape)
        except Exception as e:
            print(f"⚠️  PCA逆变换失败: {e}")
            return X_pca.reshape(-1, *original_shape)
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用优化的重采样策略"""
        print(f"🔄 应用优化重采样策略: {self.method}")
        print(f"   原始数据形状: {X.shape}")
        print(f"   类别分布: {Counter(y)}")
        
        # 分析类别分布
        distribution = self.analyze_class_distribution(y)
        
        # 计算类别权重
        class_weights = self.calculate_optimized_class_weights(y)
        
        # 检查是否需要重采样
        max_count = max(info['count'] for info in distribution.values())
        min_count = min(info['count'] for info in distribution.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"   不平衡比例: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio < 2.0:
            print(f"✅ 数据相对平衡，跳过重采样")
            return X, y
        
        # 特征展开
        X_flat = X.reshape(X.shape[0], -1)
        # 定向增强时，为保留少数类细粒度纹理，跳过PCA
        if self.method == 'bsmote' and self.directed_enhance:
            X_pca, pca_applied = X_flat, False
        else:
            X_pca, pca_applied = self.apply_pca_if_needed(X_flat)
        
        try:
            if self.method == 'bsmote' and self.directed_enhance and IMBLEARN_AVAILABLE:
                # 仅定向增强指定少数类
                current_counts = Counter(y)
                total_before = len(y)
                majority_count = max(current_counts.values()) if current_counts else 0
                # 计算固定类外的样本总数
                fixed_total = total_before - sum(current_counts.get(cid, 0) for cid in self.target_percent_per_class.keys())
                s_sum = sum(self.target_percent_per_class.values())
                # 期望目标数（先基于未增强类的总量估计）
                target_final_counts: Dict[int, int] = {}
                for cid, frac in self.target_percent_per_class.items():
                    base_target = int(round((frac / max(1e-8, (1.0 - s_sum))) * fixed_total))
                    # 不能低于当前数量
                    base_target = max(base_target, current_counts.get(cid, 0))
                    # 不超过多数类的上限比例
                    cap = int(self.cap_minor_vs_major * majority_count) if majority_count > 0 else base_target
                    base_target = min(base_target, cap)
                    target_final_counts[cid] = base_target

                # 逐类执行BorderlineSMOTE，允许类特定m_neighbors
                X_working, y_working = X_pca, y
                for cid, tgt in target_final_counts.items():
                    cur = Counter(y_working).get(cid, 0)
                    if tgt <= cur:
                        continue
                    sampling_strategy = {cid: tgt}
                    m_neighbors_val = int(self.class_neighbor_overrides.get(cid, 10))
                    sampler = BorderlineSMOTE(
                        sampling_strategy=sampling_strategy,
                        k_neighbors=self.k_neighbors,
                        m_neighbors=m_neighbors_val,
                        random_state=self.random_state,
                        kind=self.kind
                    )
                    X_working, y_working = sampler.fit_resample(X_working, y_working)

                X_resampled, y_resampled = X_working, y_working
                print(f"✅ 定向BSMOTE完成: {X_pca.shape[0]} -> {X_resampled.shape[0]}")

            else:
                # 回退为常规采样器策略
                if self.method == 'bsmote' and IMBLEARN_AVAILABLE:
                    sampler = BorderlineSMOTE(
                        k_neighbors=self.k_neighbors,
                        random_state=self.random_state,
                        kind=self.kind
                    )
                elif self.method == 'smote' and IMBLEARN_AVAILABLE:
                    sampler = SMOTE(
                        k_neighbors=self.k_neighbors,
                        random_state=self.random_state
                    )
                elif self.method == 'adasyn' and IMBLEARN_AVAILABLE:
                    sampler = ADASYN(
                        n_neighbors=self.k_neighbors,
                        random_state=self.random_state
                    )
                elif self.method == 'smoteenn' and IMBLEARN_AVAILABLE:
                    sampler = SMOTEENN(
                        smote=SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state),
                        random_state=self.random_state
                    )
                else:
                    raise ValueError(f"不支持的采样方法: {self.method}")

                X_resampled, y_resampled = sampler.fit_resample(X_pca, y)
                print(f"✅ 重采样完成: {X_pca.shape[0]} -> {X_resampled.shape[0]}")

            # 逆PCA或恢复形状
            if pca_applied:
                X_resampled = self.inverse_pca_transform(X_resampled, X.shape[1:])
            else:
                X_resampled = X_resampled.reshape(-1, *X.shape[1:])

            # 分析重采样后的分布
            resampled_distribution = self.analyze_class_distribution(y_resampled)
            print(f"📊 重采样后分布:")
            for class_id, info in resampled_distribution.items():
                print(f"   类别{class_id} ({info['name']}): {info['count']} 样本 ({info['percentage']:.1f}%)")

            return X_resampled, y_resampled

        except Exception as e:
            print(f"⚠️  重采样失败，返回原始数据: {e}")
            return X, y

def create_optimized_balanced_data_loader(data_loaders: Dict, 
                                        method: str = 'bsmote',
                                        target_ratio: float = 0.7,
                                        k_neighbors: int = 3,
                                        random_state: int = 42,
                                        enable_pca: bool = True,
                                        pca_components: int = 1000,
                                        pca_variance_ratio: float = 0.95,
                                        max_weight: float = 5.0,
                                        min_weight: float = 0.5,
                                        directed_enhance: bool = True,
                                        target_percent_per_class: Optional[Dict[int, float]] = None,
                                        class_neighbor_overrides: Optional[Dict[int, int]] = None,
                                        cap_minor_vs_major: float = 0.5,
                                        augment_balanced: bool = True) -> Dict:
    """
    创建优化的平衡数据加载器
    
    Args:
        data_loaders: 原始数据加载器字典
        method: 采样方法
        target_ratio: 目标平衡比例
        k_neighbors: 近邻数量
        random_state: 随机种子
        enable_pca: 是否启用PCA
        pca_components: PCA组件数
        max_weight: 最大类别权重
        min_weight: 最小类别权重
    
    Returns:
        包含平衡数据加载器和类别权重的字典
    """
    print(f"🚀 创建优化的平衡数据加载器")
    print(f"   方法: {method}")
    print(f"   目标比例: {target_ratio}")
    print(f"   PCA降维: {'启用' if enable_pca else '禁用'}")
    print("=" * 60)
    
    try:
        # 创建优化处理器
        handler = OptimizedImbalanceHandler(
            method=method,
            target_ratio=target_ratio,
            k_neighbors=k_neighbors,
            random_state=random_state,
            enable_pca=enable_pca,
            pca_components=pca_components,
            pca_variance_ratio=pca_variance_ratio,
            max_weight=max_weight,
            min_weight=min_weight,
            directed_enhance=directed_enhance,
            target_percent_per_class=target_percent_per_class or {4: 0.20},
            class_neighbor_overrides=class_neighbor_overrides or {4: 10},
            cap_minor_vs_major=cap_minor_vs_major,
            kind='borderline-1'
        )
        
        # 收集训练数据
        print(f"📥 收集训练数据...")
        X_train_list: List[np.ndarray] = []
        y_train_list: List[np.ndarray] = []
        for batch_data in data_loaders['train']:
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 3:
                    features, labels, _ = batch_data
                elif len(batch_data) == 2:
                    features, labels = batch_data
                else:
                    continue
                X_train_list.append(features.cpu().numpy())
                y_train_list.append(labels.cpu().numpy())
        if len(X_train_list) == 0:
            raise RuntimeError("未从训练加载器收集到数据，无法进行BSMOTE")
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        
        print(f"   训练数据形状: {X_train.shape}")
        print(f"   训练标签形状: {y_train.shape}")
        
        # 应用优化的重采样
        X_balanced, y_balanced = handler.fit_resample(X_train, y_train)
        
        # 创建平衡的数据集（返回三元组，第三项为占位metadata）
        balanced_dataset = BalancedTripletDataset(X_balanced, y_balanced, augment=augment_balanced)
        
        # 创建平衡的数据加载器
        balanced_loader = torch.utils.data.DataLoader(
            balanced_dataset,
            batch_size=data_loaders['train'].batch_size,
            shuffle=True,
            num_workers=data_loaders['train'].num_workers,
            pin_memory=data_loaders['train'].pin_memory,
            drop_last=True
        )
        
        # 计算类别权重
        class_weights = handler.calculate_optimized_class_weights(y_balanced)
        
        # 创建权重张量
        weight_tensor = torch.FloatTensor([class_weights[i] for i in range(len(class_weights))])
        
        result = {
            'train': balanced_loader,
            'val': data_loaders['val'],
            'class_weights': class_weights,
            'weight_tensor': weight_tensor,
            'handler': handler
        }
        
        # 如果存在测试集，也包含进去
        if 'test' in data_loaders:
            result['test'] = data_loaders['test']
        
        print(f"✅ 优化平衡数据加载器创建成功")
        print(f"   训练集: {len(balanced_dataset)} 样本")
        print(f"   验证集: {len(data_loaders['val'].dataset)} 样本")
        if 'test' in result:
            print(f"   测试集: {len(data_loaders['test'].dataset)} 样本")
        
        return result
        
    except Exception as e:
        print(f"❌ 创建优化平衡数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回原始数据加载器 + 基础类别权重
        print(f"🔄 回退到基础类别权重方案")
        
        # 计算基础类别权重
        train_labels = []
        for batch_data in data_loaders['train']:
            if len(batch_data) == 2:
                _, labels = batch_data
                train_labels.extend(labels.cpu().numpy())
        
        train_labels = np.array(train_labels)
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        class_weights = {}
        num_classes = 5  # 5个流体类型
        for class_id in range(num_classes):
            count = class_counts.get(class_id, 1)
            weight = total_samples / (num_classes * count)
            class_weights[class_id] = weight
        
        weight_tensor = torch.FloatTensor([class_weights[i] for i in range(num_classes)])
        
        return {
            'train': data_loaders['train'],
            'val': data_loaders['val'],
            'class_weights': class_weights,
            'weight_tensor': weight_tensor,
            'handler': None
        }
