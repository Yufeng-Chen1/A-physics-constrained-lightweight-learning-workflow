#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于行的井数据加载器
将每一行数据作为一个样本，大幅增加数据集大小
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pywt

# 延迟导入sklearn以避免多进程问题
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    # 如果在多进程环境中导入失败，使用替代方案
    def train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
        """简单的train_test_split替代实现"""
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        
        return X_train, X_test, y_train, y_test
    
    class LabelEncoder:
        """简单的LabelEncoder替代实现"""
        def __init__(self):
            self.classes_ = None
            self.class_to_index = {}
        
        def fit_transform(self, y):
            unique_labels = list(set(y))
            self.classes_ = np.array(unique_labels)
            self.class_to_index = {label: i for i, label in enumerate(unique_labels)}
            return np.array([self.class_to_index[label] for label in y])
        
        def transform(self, y):
            return np.array([self.class_to_index.get(label, 0) for label in y])
        
        def inverse_transform(self, y):
            return np.array([self.classes_[i] for i in y])


class RowBasedWellLogDataset(Dataset):
    """
    基于行的井数据加载器
    将每一行数据作为一个样本，大幅增加数据集大小
    """
    
    def __init__(self, 
                 welldata_dir: str = "welldata",
                 curve_names: List[str] = None,
                 enable_wavelet: bool = True,
                 image_size: Tuple[int, int] = (16, 16),
                 enable_cleaning: bool = True,
                 test_mode: bool = False):
        """
        初始化基于行的井数据加载器
        
        Args:
            welldata_dir: 井数据目录路径
            curve_names: 测井曲线名称列表
            enable_wavelet: 是否启用小波变换
            image_size: 图像尺寸 (H, W)
            enable_cleaning: 是否启用数据清洗
            test_mode: 测试模式，只加载少量数据
        """
        self.welldata_dir = welldata_dir
        self.curve_names = curve_names or ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
        self.enable_wavelet = enable_wavelet
        self.image_size = image_size
        self.enable_cleaning = enable_cleaning
        self.test_mode = test_mode
        
        # 标签编码器
        self.label_encoder = LabelEncoder()
        
        # 加载所有井数据
        self.data, self.labels = self._load_all_well_data()
        
        # 编码标签
        if len(self.labels) > 0:
            self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        else:
            self.labels_encoded = np.array([])
        
        print(f"📊 基于行的数据加载完成:")
        print(f"   总数据量: {len(self.data)}")
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
            print(f"   标签类别: {list(self.label_encoder.classes_)}")
        else:
            print(f"   标签类别: {list(set(self.labels))}")
        print(f"   图像尺寸: {self.image_size}")
    
    def _load_all_well_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """加载所有井的数据，每一行作为一个样本"""
        data_files = glob.glob(os.path.join(self.welldata_dir, "*.txt"))
        
        if not data_files:
            print(f"❌ 在 {self.welldata_dir} 目录中未找到数据文件")
            return [], []
        
        print(f"🔍 发现 {len(data_files)} 个井数据文件")
        
        all_data = []
        all_labels = []
        
        for file_path in data_files:
            well_name = os.path.basename(file_path).replace('.txt', '')
            print(f"   正在加载: {well_name}")
            
            try:
                # 加载井数据
                well_data = self._load_single_well_data(file_path)
                
                if well_data is not None and len(well_data) > 0:
                    # 数据清洗
                    if self.enable_cleaning:
                        well_data = self._clean_well_data(well_data, well_name)
                    
                    # 将每一行作为一个样本
                    row_samples = self._generate_row_samples(well_data, well_name)
                    
                    if len(row_samples) > 0:
                        samples, sample_labels = zip(*row_samples)
                        all_data.extend(samples)
                        all_labels.extend(sample_labels)
                        
                        if self.test_mode and len(all_data) >= 1000:  # 测试模式限制数据量
                            break
                    
            except Exception as e:
                print(f"   ❌ 加载 {well_name} 失败: {e}")
                continue
        
        print(f"✅ 成功加载 {len(all_data)} 个数据样本")
        return all_data, all_labels
    
    def _load_single_well_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单个井的数据"""
        try:
            # 尝试不同的编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding, sep='\t')
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                print(f"   ❌ 无法读取文件: {file_path}")
                return None
            
            # 检查必要的列
            available_curves = [col for col in data.columns if col in self.curve_names]
            
            if len(available_curves) < 3:  # 至少需要3个测井曲线
                print(f"   ⚠️  {file_path} 缺少必要的测井曲线")
                return None
            
            # 保留可用的测井曲线与解释结论列（若存在）
            label_col = '解释结论'
            keep_cols = available_curves + ([label_col] if label_col in data.columns else [])
            data = data[keep_cols].copy()
            
            # 处理缺失值
            data = data.ffill().bfill()
            
            # 移除无效行
            data = data.dropna()
            
            if len(data) < 10:  # 至少需要10行数据
                print(f"   ⚠️  {file_path} 数据量不足 ({len(data)} < 10)")
                return None
            
            return data
            
        except Exception as e:
            print(f"   ❌ 读取 {file_path} 失败: {e}")
            return None
    
    def _clean_well_data(self, data: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """清洗井数据"""
        try:
            # 过滤有效数据
            valid_data = self._filter_valid_data(data, well_name)
            
            if len(valid_data) == 0:
                print(f"   ⚠️  {well_name} 过滤后没有有效数据")
                return valid_data
            
            # 数据清洗和归一化
            cleaned_data = self._normalize_data(valid_data)
            
            return cleaned_data
        except Exception as e:
            print(f"   ⚠️  清洗 {well_name} 数据失败: {e}")
            return data
    
    def _filter_valid_data(self, data: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """过滤出有效数据 - 放宽过滤条件"""
        try:
            # 大幅放宽各测井曲线的合理值范围
            valid_ranges = {
                'GR': (-100, 500),    # 自然伽马 (API) - 极宽松
                'SP': (-200, 200),    # 自然电位 (mV) - 极宽松
                'AC': (0, 1000),      # 声波时差 (μs/ft) - 极宽松
                'DEN': (0.5, 5.0),    # 密度 (g/cm³) - 极宽松
                'CNL': (-20, 80),     # 中子孔隙度 (%) - 极宽松
                'RT': (0.01, 1000)    # 电阻率 (Ω·m) - 极宽松
            }
            
            # 过滤有效数据 - 只对数值列进行过滤
            valid_mask = pd.Series([True] * len(data), index=data.index)
            
            for curve_name in data.columns:
                if curve_name in valid_ranges:
                    # 只对数值列进行范围过滤，跳过标签列
                    if curve_name in ['解释结论', '结论', '岩性', '含油性']:
                        continue
                    
                    try:
                        min_val, max_val = valid_ranges[curve_name]
                        curve_mask = (data[curve_name] >= min_val) & (data[curve_name] <= max_val)
                        valid_mask = valid_mask & curve_mask
                    except (TypeError, ValueError) as e:
                        # 如果列包含非数值数据，跳过该列的过滤
                        print(f"     ⚠️  跳过非数值列 {curve_name} 的过滤: {e}")
                        continue
            
            # 应用过滤条件
            filtered_data = data[valid_mask].copy()
            
            print(f"     {well_name} 数据过滤: {len(data)} -> {len(filtered_data)} 行 ({len(filtered_data)/len(data)*100:.1f}%)")
            
            # 如果过滤后数据太少，返回原始数据
            if len(filtered_data) < len(data) * 0.05:  # 如果少于5%的数据
                print(f"     ⚠️  {well_name} 过滤后数据极少，返回原始数据")
                return data
            
            return filtered_data
            
        except Exception as e:
            print(f"     ⚠️  {well_name} 数据过滤失败: {e}")
            return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """归一化数据"""
        try:
            cleaned_data = data.copy()
            
            for curve in self.curve_names:
                if curve in data.columns:
                    # 替换无穷大值
                    cleaned_data[curve] = cleaned_data[curve].replace([np.inf, -np.inf], np.nan)
                    
                    # 使用中位数填充NaN值
                    if cleaned_data[curve].isna().any():
                        median_val = cleaned_data[curve].median()
                        cleaned_data[curve] = cleaned_data[curve].fillna(median_val)
                    
                    # 标准化归一化
                    mean_val = cleaned_data[curve].mean()
                    std_val = cleaned_data[curve].std()
                    if std_val > 1e-8:  # 防止除零
                        cleaned_data[curve] = (cleaned_data[curve] - mean_val) / std_val
                    else:
                        cleaned_data[curve] = cleaned_data[curve] - mean_val
            
            return cleaned_data
        except Exception as e:
            print(f"     ⚠️  归一化失败: {e}")
            return data
    
    def _clean_label_string(self, label: str) -> str:
        """清理标签字符串"""
        if not label or label.strip() == '':
            return 'unknown'
            
        label = str(label).strip()
        
        # 标准化标签为5个主要流体类型：油层、水层、干层、差油层、油水层
        # 非这5类的归为None，将被过滤
        label_mapping = {
            # 主要流体类型
            '油层': '油层', '水层': '水层', '干层': '干层', '乾层': '干层',
            '差油层': '差油层', '差油層': '差油层', 
            # 油水层：包含油水同层和含油水层
            '油水层': '油水层', '油水層': '油水层',
            '油水同层': '油水层', '油水同層': '油水层',
            '含油水层': '油水层', '含油水層': '油水层',
            '油水共存': '油水层', '油水互层': '油水层',
            # 气层和致密油层归为油层
            '气层': '油层', '氣层': '油层', '油气层': '油层',
            '致密油层': '油层', '致密油': '油层', '致密': '油层',
            # 英文标签
            'oil': '油层', 'water': '水层', 'dry': '干层', 
            'gas': '油层', 'poor': '差油层', 'oil-water': '油水层',
            # 细分类型
            '良好油层': '油层', '高产油层': '油层', '可采油层': '油层',
            '低产油层': '差油层', '薄油层': '差油层', '边底水油层': '差油层',
            '底水层': '水层', '边水层': '水层', '高含水层': '水层'
        }
        
        return label_mapping.get(label, None)  # 非主要类型返回None，将被过滤
    
    def _generate_row_samples(self, data: pd.DataFrame, well_name: Optional[str] = None) -> List[Tuple[np.ndarray, str]]:
        """将每一行作为一个样本"""
        samples_with_labels = []
        
        # 确定可用的解释结论列
        label_column_candidates = ['解释结论', '结论', '岩性', '含油性']
        label_col = None
        for col in label_column_candidates:
            if col in data.columns:
                label_col = col
                break
        
        # 仅保留数值曲线列用于特征
        numeric_cols = [c for c in data.columns if c in self.curve_names]
        if len(numeric_cols) == 0:
            return []
        
        # 遍历每一行
        for idx, row in data.iterrows():
            # 获取数值特征
            features = row[numeric_cols].values
            
            # 数据验证 - 修复数据类型问题
            try:
                # 确保features是数值类型
                features = features.astype(np.float64)
                if np.isnan(features).any() or np.isinf(features).any():
                    continue
            except (TypeError, ValueError) as e:
                print(f"     ⚠️  数据类型转换失败: {e}")
                continue
            
            # 确定标签
            label = '干层'  # 默认标签
            if label_col is not None and label_col in row:
                raw_label = str(row[label_col]).strip()
                if raw_label and raw_label.lower() != 'nan':
                    label = self._clean_label_string(raw_label)
            
            # 如果标签仍然是unknown，使用井名
            if label == 'unknown':
                label = well_name if well_name is not None else '干层'
            
            samples_with_labels.append((features, label))
        
        return samples_with_labels
    
    def _generate_wavelet_features(self, row_data: np.ndarray) -> np.ndarray:
        """为单行数据生成小波特征"""
        try:
            # 将单行数据扩展为序列
            # 使用重复和插值来创建序列
            sequence_length = self.image_size[0] * self.image_size[1]
            
            if len(row_data) >= sequence_length:
                # 如果数据足够，直接截断
                sequence = row_data[:sequence_length]
            else:
                # 如果数据不足，进行插值扩展
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(row_data))
                x_new = np.linspace(0, 1, sequence_length)
                f = interp1d(x_old, row_data, kind='linear', bounds_error=False, fill_value=0)
                sequence = f(x_new)
            
            # 重塑为图像格式
            image = sequence.reshape(self.image_size)
            
            # 为每个测井曲线创建通道
            channels = []
            for i, curve_name in enumerate(self.curve_names):
                if i < len(row_data):
                    # 使用该曲线的值创建通道
                    channel = np.full(self.image_size, row_data[i])
                    channels.append(channel)
                else:
                    # 如果曲线数量不足，使用零填充
                    channels.append(np.zeros(self.image_size))
            
            # 组合所有通道
            features = np.stack(channels, axis=0)
            
            return features
            
        except Exception as e:
            print(f"   ⚠️  生成小波特征失败: {e}")
            # 备选方案：使用原始数据
            return self._create_simple_features(row_data)
    
    def _create_simple_features(self, row_data: np.ndarray) -> np.ndarray:
        """创建简单的特征表示"""
        try:
            # 创建固定尺寸的特征
            features = np.zeros((len(self.curve_names), *self.image_size))
            
            # 将每个测井曲线的值填充到对应的通道
            for i, value in enumerate(row_data):
                if i < len(self.curve_names):
                    features[i] = np.full(self.image_size, value)
            
            return features
            
        except Exception as e:
            print(f"   ⚠️  创建简单特征失败: {e}")
            # 返回零填充的默认特征
            return np.zeros((len(self.curve_names), *self.image_size))
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        row_data = self.data[idx]
        label = self.labels_encoded[idx]
        
        if self.enable_wavelet:
            # 生成小波特征
            features = self._generate_wavelet_features(row_data)
        else:
            # 使用简单特征
            features = self._create_simple_features(row_data)
        
        # 转换为张量
        sequence_tensor = torch.FloatTensor(features)
        
        # 最终验证张量形状
        expected_shape = (len(self.curve_names), *self.image_size)
        if sequence_tensor.shape != expected_shape:
            print(f"⚠️  张量形状不匹配: 期望{expected_shape}, 实际{sequence_tensor.shape}")
            # 强制调整形状
            if sequence_tensor.numel() >= np.prod(expected_shape):
                sequence_tensor = sequence_tensor.view(expected_shape)
            else:
                # 如果元素不足，用零填充
                new_tensor = torch.zeros(expected_shape, dtype=sequence_tensor.dtype)
                flat_old = sequence_tensor.flatten()
                flat_new = new_tensor.flatten()
                flat_new[:min(len(flat_old), len(flat_new))] = flat_old[:min(len(flat_old), len(flat_new))]
                sequence_tensor = flat_new.view(expected_shape)
        
        return sequence_tensor, torch.LongTensor([label]).squeeze()


def create_row_based_data_loaders(
    welldata_dir: str = "welldata",
    curve_names: List[str] = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    enable_wavelet: bool = True,
    image_size: Tuple[int, int] = (16, 16),
    enable_cleaning: bool = True,
    num_workers: int = 0,
    test_mode: bool = False
) -> Dict[str, DataLoader]:
    """
    创建基于行的数据加载器
    
    Returns:
        包含训练、验证数据加载器的字典
    """
    
    # 创建数据集
    dataset = RowBasedWellLogDataset(
        welldata_dir=welldata_dir,
        curve_names=curve_names,
        enable_wavelet=enable_wavelet,
        image_size=image_size,
        enable_cleaning=enable_cleaning,
        test_mode=test_mode
    )
    
    if len(dataset) == 0:
        raise ValueError("没有可用的数据")
    
    # 计算分割大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"📊 数据集分割:")
    print(f"   训练集: {len(train_dataset)}")
    print(f"   验证集: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'dataset': dataset
    }


def test_row_based_data_loader():
    """测试基于行的数据加载器"""
    print("🧪 测试基于行的数据加载器...")
    
    try:
        # 创建数据集
        dataset = RowBasedWellLogDataset(
            welldata_dir="welldata",
            test_mode=True
        )
        
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample_data, sample_label = dataset[0]
            print(f"✅ 样本获取成功:")
            print(f"   数据形状: {sample_data.shape}")
            print(f"   标签: {sample_label}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_row_based_data_loader()
