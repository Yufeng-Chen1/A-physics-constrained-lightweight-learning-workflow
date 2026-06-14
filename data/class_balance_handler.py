#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别平衡处理器
使用BSMOTE和其他方法处理储层流体类型样本不平衡问题
"""

import numpy as np
import torch
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

from data.fluid_types import get_fluid_name, get_num_fluid_classes

class ClassBalanceHandler:
    """类别平衡处理器"""
    
    def __init__(self, method: str = 'bsmote', random_state: int = 42):
        """
        初始化类别平衡处理器
        
        Args:
            method: 平衡方法 ('bsmote', 'smote', 'adasyn', 'smote_tomek', 'class_weight')
            random_state: 随机种子
        """
        self.method = method
        self.random_state = random_state
        self.sampler = None
        self.class_weights = None
        self.fluid_names = [get_fluid_name(i) for i in range(get_num_fluid_classes())]
        
        if not IMBLEARN_AVAILABLE and method != 'class_weight':
            raise ImportError("imbalanced-learn is required for sampling methods. Install with: pip install imbalanced-learn")
    
    def analyze_class_distribution(self, labels: np.ndarray) -> Dict:
        """分析类别分布"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        distribution = {}
        for label_id, count in label_counts.items():
            fluid_name = get_fluid_name(label_id)
            percentage = count / total_samples * 100
            distribution[fluid_name] = {
                'count': count,
                'percentage': percentage,
                'label_id': label_id
            }
        
        # 计算不平衡比例
        counts = list(label_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return {
            'distribution': distribution,
            'imbalance_ratio': imbalance_ratio,
            'total_samples': total_samples,
            'num_classes': len(label_counts)
        }
    
    def create_sampling_strategy(self, labels: np.ndarray, target_ratio: float = 0.5) -> Dict:
        """
        创建采样策略
        
        Args:
            labels: 标签数组
            target_ratio: 目标平衡比例 (0.5表示1:2的比例)
        """
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        sampling_strategy = {}
        for label_id, count in label_counts.items():
            # 计算目标样本数
            target_count = int(max_count * target_ratio)
            if count < target_count:
                sampling_strategy[label_id] = target_count
        
        return sampling_strategy
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray, 
                    sampling_strategy: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用采样方法
        
        Args:
            X: 特征数据 (n_samples, n_features)
            y: 标签数据 (n_samples,)
            sampling_strategy: 采样策略
        """
        if self.method == 'class_weight':
            # 只计算类别权重，不进行采样
            self._calculate_class_weights(y)
            return X, y
        
        # 创建采样器
        if self.method == 'bsmote':
            self.sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=5,
                random_state=self.random_state
            )
        elif self.method == 'smote':
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=5,
                random_state=self.random_state
            )
        elif self.method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=5,
                random_state=self.random_state
            )
        elif self.method == 'smote_tomek':
            self.sampler = SMOTETomek(
                sampling_strategy=sampling_strategy,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # 应用采样
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        
        # 计算类别权重（用于损失函数）
        self._calculate_class_weights(y_resampled)
        
        return X_resampled, y_resampled
    
    def _calculate_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """计算类别权重（长度固定为全体流体类别数）"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        # 使用全体类别数（例如6类），即使本批次只出现部分类别
        try:
            from data.fluid_types import get_num_fluid_classes
            total_classes = get_num_fluid_classes()
        except Exception:
            total_classes = int(max(label_counts.keys()) + 1) if label_counts else 1
        
        # 计算权重 (使用逆频率)，对缺失类别使用1.0权重
        weights = []
        effective_classes = max(len(label_counts), 1)
        for i in range(total_classes):
            if i in label_counts and label_counts[i] > 0:
                weight = total_samples / (effective_classes * label_counts[i])
                weights.append(float(weight))
            else:
                weights.append(1.0)
        
        self.class_weights = torch.FloatTensor(weights)
        return self.class_weights
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """获取类别权重"""
        return self.class_weights
    
    def print_balance_report(self, labels_before: np.ndarray, labels_after: np.ndarray = None):
        """打印平衡报告"""
        print(f"📊 类别平衡处理报告")
        print("=" * 50)
        
        # 分析原始分布
        before_analysis = self.analyze_class_distribution(labels_before)
        print(f"🔍 原始数据分布:")
        print(f"   总样本数: {before_analysis['total_samples']}")
        print(f"   类别数: {before_analysis['num_classes']}")
        print(f"   不平衡比例: {before_analysis['imbalance_ratio']:.1f}:1")
        
        for fluid_name, info in before_analysis['distribution'].items():
            print(f"   {fluid_name}: {info['count']} 样本 ({info['percentage']:.1f}%)")
        
        if labels_after is not None:
            # 分析处理后分布
            after_analysis = self.analyze_class_distribution(labels_after)
            print(f"\n✅ 处理后数据分布:")
            print(f"   总样本数: {after_analysis['total_samples']}")
            print(f"   类别数: {after_analysis['num_classes']}")
            print(f"   不平衡比例: {after_analysis['imbalance_ratio']:.1f}:1")
            
            for fluid_name, info in after_analysis['distribution'].items():
                print(f"   {fluid_name}: {info['count']} 样本 ({info['percentage']:.1f}%)")
            
            # 计算改善情况
            improvement = before_analysis['imbalance_ratio'] / after_analysis['imbalance_ratio']
            print(f"\n📈 改善情况:")
            print(f"   不平衡比例改善: {improvement:.1f}x")
            print(f"   样本增加: {after_analysis['total_samples'] - before_analysis['total_samples']} 个")
        
        if self.class_weights is not None:
            print(f"\n⚖️  类别权重:")
            for i, weight in enumerate(self.class_weights):
                fluid_name = get_fluid_name(i)
                print(f"   {fluid_name}: {weight:.3f}")

def create_balanced_data_loader(data_loaders: Dict, method: str = 'bsmote', 
                               target_ratio: float = 0.5, config: Dict = None) -> Dict:
    """
    创建平衡的数据加载器
    
    Args:
        data_loaders: 原始数据加载器字典
        method: 平衡方法
        target_ratio: 目标平衡比例
    
    Returns:
        平衡后的数据加载器字典
    """
    print(f"🔄 应用类别平衡方法: {method.upper()}")
    print("=" * 50)
    
    # 创建平衡处理器
    balance_handler = ClassBalanceHandler(method=method)
    
    # 处理训练集
    train_loader = data_loaders['train']
    
    # 收集训练数据（使用更智能的内存管理）
    X_train = []
    y_train = []
    
    print("📥 收集训练数据...")
    try:
        batch_count = 0
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                # 处理可能返回2个或3个值的情况
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 3:
                        data, labels, _ = batch_data
                    elif len(batch_data) == 2:
                        data, labels = batch_data
                    else:
                        print(f"⚠️  意外的批次数据格式: {len(batch_data)} 个元素")
                        continue
                else:
                    print(f"⚠️  批次数据不是列表或元组: {type(batch_data)}")
                    continue
                
                # 确保数据在CPU上并转换为numpy
                if data.is_cuda:
                    data = data.cpu()
                if labels.is_cuda:
                    labels = labels.cpu()
                
                X_train.append(data.numpy())
                y_train.append(labels.numpy())
                batch_count += 1
                
                # 每50个批次显示进度，减少输出频率
                if (batch_idx + 1) % 50 == 0:
                    print(f"   已处理 {batch_idx + 1} 个批次...")
                
                # 内存管理：每1000个批次清理一次
                if (batch_idx + 1) % 1000 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                print(f"⚠️  处理批次 {batch_idx} 时出错: {e}")
                continue
        
        print(f"   收集到训练数据: {batch_count} 批次")
        
    except KeyboardInterrupt:
        print("⚠️  数据收集被中断")
        if len(X_train) == 0:
            print("❌ 没有收集到任何数据，无法继续")
            return data_loaders
        else:
            print(f"   使用已收集的 {len(X_train)} 批次数据继续处理")
    except Exception as e:
        print(f"❌ 数据收集失败: {e}")
        return data_loaders
    
    # 合并数据
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    print(f"   收集到训练数据: {len(X_train)} 样本")
    
    # 重塑数据以适应采样器 (n_samples, n_features)
    original_shape = X_train.shape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    print(f"   原始训练数据形状: {original_shape}")
    print(f"   扁平化后形状: {X_train_flat.shape}")
    
    # 智能PCA降维策略
    pca = None
    if config is not None:
        enable_pca = config.get('enable_pca', True)
        pca_components = config.get('pca_components', 1000)
        pca_variance_ratio = config.get('pca_variance_ratio', 0.90)
    else:
        enable_pca = True  # 默认启用PCA降维
        pca_components = 1000  # PCA组件数
        pca_variance_ratio = 0.90  # 保留方差比例
    
    print(f"🔧 原始特征形状: {X_train_flat.shape}")
    
    if enable_pca and X_train_flat.shape[1] > pca_components:
        try:
            from sklearn.decomposition import PCA
            print("🔄 应用智能特征选择以减少计算复杂度...")
            
            # 智能PCA降维策略
            print("🔄 计算最优PCA组件数...")
            
            # 使用更保守的PCA策略，避免内存问题
            max_components = min(pca_components, X_train_flat.shape[0] - 1, X_train_flat.shape[1] - 1)
            
            if max_components < 100:  # 如果组件数太少，跳过PCA
                print(f"   组件数太少 ({max_components})，跳过PCA降维")
            else:
                # 计算保留指定方差比例所需的主成分数
                pca_temp = PCA(random_state=42)
                pca_temp.fit(X_train_flat)
                cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components_variance = np.argmax(cumsum >= pca_variance_ratio) + 1
                
                # 选择最优的组件数：取配置值、方差比例值和数据限制的最小值
                n_components = min(
                    max_components, 
                    n_components_variance
                )
                
                if n_components < X_train_flat.shape[1]:
                    print(f"   保留{pca_variance_ratio*100}%方差需要: {n_components_variance} 个主成分")
                    print(f"   选择组件数: {n_components}")
                    
                    # 应用PCA降维
                    pca = PCA(n_components=n_components, random_state=42)
                    X_train_flat = pca.fit_transform(X_train_flat)
                    
                    explained_variance = np.sum(pca.explained_variance_ratio_)
                    print(f"✅ PCA降维完成: {X_train_flat.shape}")
                    print(f"   解释方差比: {explained_variance:.3f}")
                    print(f"   降维比例: {X_train_flat.shape[1]/X_train_flat.shape[1]:.3f}")
                else:
                    print("   特征数量适中，跳过PCA降维")
        except Exception as e:
            print(f"⚠️  PCA降维失败: {e}")
            print("   使用原始特征继续处理")
    else:
        print("🔧 特征维度适中，跳过PCA降维")
        print(f"   特征数: {X_train_flat.shape[1]} <= 阈值: {pca_components}")
    
    # 创建采样策略（仅当需要重采样时）
    sampling_strategy = None
    if method != 'class_weight':
        # 对于时频图谱数据，使用更保守的采样策略
        print("🔄 创建保守的采样策略...")
        
        # 计算类别分布
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        max_count = np.max(class_counts)
        min_count = np.min(class_counts)
        
        # 使用更保守的目标比例，避免过度采样
        conservative_target_ratio = min(target_ratio, 0.7)  # 最多平衡到70%
        
        # 计算目标样本数
        target_samples = int(max_count * conservative_target_ratio)
        
        # 只为少数类创建采样策略
        sampling_strategy = {}
        for class_label, count in zip(unique_classes, class_counts):
            if count < target_samples:
                sampling_strategy[class_label] = target_samples
            else:
                # 多数类不进行过采样
                sampling_strategy[class_label] = count
        
        print(f"   保守采样策略: {sampling_strategy}")
        print(f"   目标平衡比例: {conservative_target_ratio}")
    
    # 应用平衡方法
    if method == 'class_weight':
        # 只计算类别权重，不改动原始数据加载器，保持验证/测试分布不变
        balance_handler._calculate_class_weights(y_train)
        balance_handler.print_balance_report(y_train)
        result = {
            'train': data_loaders['train'],
            'val': data_loaders['val']
        }
        if 'test' in data_loaders:
            result['test'] = data_loaders['test']
        result['class_weights'] = balance_handler.get_class_weights()
        return result
    else:
        # 应用采样，添加错误处理
        try:
            print("🔄 应用BSMOTE采样...")
            X_balanced_flat, y_balanced = balance_handler.fit_resample(
                X_train_flat, y_train, sampling_strategy
            )
            print("✅ BSMOTE采样完成")
            
            # 如果使用了PCA降维，需要重建原始形状
            if pca is not None:
                print("🔄 重建原始特征维度...")
                # 使用PCA逆变换重建原始特征
                X_balanced_flat = pca.inverse_transform(X_balanced_flat)
                print(f"   重建后形状: {X_balanced_flat.shape}")
            
            # 重塑回原始形状
            X_balanced = X_balanced_flat.reshape(-1, *original_shape[1:])
            
        except Exception as e:
            print(f"⚠️  BSMOTE采样失败: {e}")
            print("🔄 回退到类别权重方法...")
            
            # 回退到类别权重方法
            balance_handler._calculate_class_weights(y_train)
            balance_handler.print_balance_report(y_train)
            
            # 返回原始数据加载器，但添加类别权重
            result = {
                'train': data_loaders['train'],
                'val': data_loaders['val']
            }
            if 'test' in data_loaders:
                result['test'] = data_loaders['test']
            if 'dataset' in data_loaders:
                result['dataset'] = data_loaders['dataset']
            result['class_weights'] = balance_handler.get_class_weights()
            return result
    
    # 打印平衡报告
    balance_handler.print_balance_report(y_train, y_balanced)
    
    # 创建新的数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_balanced)
    y_tensor = torch.LongTensor(y_balanced)
    
    # 创建数据集
    balanced_dataset = TensorDataset(X_tensor, y_tensor)
    
    # 创建数据加载器
    balanced_loader = DataLoader(
        balanced_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )
    
    # 返回结果
    result = {
        'train': balanced_loader,
        'val': data_loaders['val']
    }
    
    if 'test' in data_loaders:
        result['test'] = data_loaders['test']
    
    if 'dataset' in data_loaders:
        result['dataset'] = data_loaders['dataset']
    
    # 添加类别权重
    if balance_handler.get_class_weights() is not None:
        result['class_weights'] = balance_handler.get_class_weights()
        # 添加weight_tensor用于损失函数
        class_weights = balance_handler.get_class_weights()
        if class_weights is not None:
            # 确保权重按类别顺序排列
            weight_list = []
            for i in range(len(class_weights)):
                weight_list.append(class_weights[i])
            result['weight_tensor'] = torch.FloatTensor(weight_list).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return result

def test_class_balance_handler():
    """测试类别平衡处理器"""
    print("🧪 测试类别平衡处理器")
    print("=" * 50)
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟不平衡数据
    X = np.random.randn(1000, 64, 64)  # 模拟时频图谱
    y = np.array([0] * 400 + [1] * 300 + [2] * 200 + [3] * 100)  # 不平衡标签
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始标签分布: {Counter(y)}")
    
    # 测试不同方法
    methods = ['bsmote', 'smote', 'class_weight']
    
    for method in methods:
        print(f"\n🔧 测试方法: {method}")
        try:
            handler = ClassBalanceHandler(method=method)
            
            if method == 'class_weight':
                X_balanced, y_balanced = handler.fit_resample(X.reshape(X.shape[0], -1), y)
                X_balanced = X_balanced.reshape(X.shape)
            else:
                X_balanced, y_balanced = handler.fit_resample(X.reshape(X.shape[0], -1), y)
                X_balanced = X_balanced.reshape(-1, *X.shape[1:])
            
            handler.print_balance_report(y, y_balanced)
            
        except Exception as e:
            print(f"❌ 方法 {method} 测试失败: {e}")

if __name__ == "__main__":
    test_class_balance_handler()
