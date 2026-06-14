#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量井数据加载器
处理welldata目录内的所有井数据文件，用于模型训练
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pywt  # 添加pywt导入

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

# 添加项目根目录到Python路径
# 数据清洗器类定义
class WellLogDataCleaner:
    """
    测井数据清洗器
    提供数据去噪、异常值检测和归一化功能
    """
    
    def __init__(self, curve_names: List[str] = None):
        """
        初始化数据清洗器
        Args:
            curve_names: 测井曲线名称列表
        """
        if curve_names is None:
            curve_names = ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
        self.curve_names = curve_names
        
        # 定义每种曲线的清洗策略
        self.cleaning_strategies = {
            'GR': {'denoising': 'gaussian_filter', 'normalization': 'robust'},
            'SP': {'denoising': 'savgol_filter', 'normalization': 'standard'},
            'AC': {'denoising': 'gaussian_filter', 'normalization': 'robust'},
            'DEN': {'denoising': 'gaussian_filter', 'normalization': 'minmax'},
            'CNL': {'denoising': 'gaussian_filter', 'normalization': 'robust'},
            'RT': {'denoising': 'gaussian_filter', 'normalization': 'log_robust'}
        }
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        Args:
            data: 原始数据DataFrame
        Returns:
            清洗后的数据DataFrame
        """
        cleaned_data = data.copy()
        
        for curve in self.curve_names:
            if curve in data.columns:
                print(f"   清洗 {curve} 曲线...")
                
                # 去噪
                denoising_method = self.cleaning_strategies[curve]['denoising']
                cleaned_data[curve] = self._denoise_curve(data[curve], denoising_method)
                
                # 异常值检测和处理
                outliers = self._detect_outliers(cleaned_data[curve])
                if outliers.sum() > 0:
                    print(f"     检测到 {outliers.sum()} 个异常值")
                    cleaned_data.loc[outliers, curve] = np.nan
                    cleaned_data[curve] = cleaned_data[curve].interpolate(method='linear')
                
                # 归一化
                normalization_method = self.cleaning_strategies[curve]['normalization']
                cleaned_data[curve] = self._normalize_curve(cleaned_data[curve], normalization_method)
                print(f"     归一化完成: {normalization_method}")
        
        return cleaned_data
    
    def _denoise_curve(self, curve_data: pd.Series, method: str) -> pd.Series:
        """去噪"""
        if method == 'gaussian_filter':
            from scipy.ndimage import gaussian_filter1d
            return pd.Series(gaussian_filter1d(curve_data, sigma=1), index=curve_data.index)
        elif method == 'savgol_filter':
            from scipy.signal import savgol_filter
            return pd.Series(savgol_filter(curve_data, window_length=5, polyorder=2), index=curve_data.index)
        else:
            return curve_data
    
    def _detect_outliers(self, curve_data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """异常值检测"""
        z_scores = np.abs((curve_data - curve_data.mean()) / curve_data.std())
        return z_scores > threshold
    
    def _normalize_curve(self, curve_data: pd.Series, method: str) -> pd.Series:
        """归一化 - 修复版本，防止NaN/Inf值"""
        try:
            # 数据预处理：处理异常值
            curve_data = curve_data.copy()
            
            # 替换无穷大值
            curve_data = curve_data.replace([np.inf, -np.inf], np.nan)
            
            # 使用中位数填充NaN值
            if curve_data.isna().any():
                median_val = curve_data.median()
                curve_data = curve_data.fillna(median_val)
            
            if method == 'standard':
                mean_val = curve_data.mean()
                std_val = curve_data.std()
                if std_val > 1e-8:  # 防止除零
                    return (curve_data - mean_val) / std_val
                else:
                    return curve_data - mean_val  # 只中心化，不缩放
                    
            elif method == 'minmax':
                min_val = curve_data.min()
                max_val = curve_data.max()
                range_val = max_val - min_val
                if range_val > 1e-8:  # 防止除零
                    return (curve_data - min_val) / range_val
                else:
                    return curve_data - min_val  # 只中心化，不缩放
                    
            elif method == 'robust':
                median_val = curve_data.median()
                q75, q25 = curve_data.quantile(0.75), curve_data.quantile(0.25)
                iqr = q75 - q25
                if iqr > 1e-8:  # 防止除零
                    return (curve_data - median_val) / iqr
                else:
                    return curve_data - median_val  # 只中心化，不缩放
                    
            elif method == 'log_robust':
                # 对数变换后进行稳健归一化
                # 确保所有值都是正数
                positive_data = np.abs(curve_data) + 1e-8
                log_data = np.log(positive_data)
                
                median_val = log_data.median()
                q75, q25 = log_data.quantile(0.75), log_data.quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 1e-8:  # 防止除零
                    return (log_data - median_val) / iqr
                else:
                    return log_data - median_val  # 只中心化，不缩放
            else:
                return curve_data
                
        except Exception as e:
            print(f"     ⚠️  归一化失败 ({method}): {e}，返回原始数据")
            return curve_data


class BatchWellLogDataset(Dataset):
    """
    批量井数据加载器
    处理多个井的数据文件，合并为一个数据集
    """
    
    def __init__(self, 
                 data=None,
                 labels=None,
                 config=None,
                 welldata_dir: str = "welldata",
                 curve_names: List[str] = None,
                 sequence_length: int = 100,
                 enable_wavelet: bool = True,
                 image_size: Tuple[int, int] = None,  # 移除硬编码
                 enable_cleaning: bool = True,
                 test_mode: bool = False):
        """
        初始化批量井数据加载器
        
        Args:
            data: 预加载的数据（如果提供）
            labels: 预加载的标签（如果提供）
            config: 配置对象
            welldata_dir: 井数据目录路径
            curve_names: 测井曲线名称列表
            sequence_length: 序列长度
            enable_wavelet: 是否启用小波变换
            image_size: 图像尺寸 (H, W)
            enable_cleaning: 是否启用数据清洗
            test_mode: 测试模式，只加载少量数据
        """
        # 如果提供了预加载的数据，直接使用
        if data is not None and labels is not None and config is not None:
            self.data = data
            self.labels = labels
            self.config = config
            self.curve_names = config.DATA_CONFIG['curve_names']
            self.sequence_length = config.DATA_CONFIG['sequence_length']
            self.image_size = config.DATA_CONFIG['image_size']
            self.enable_wavelet = enable_wavelet
            self.enable_cleaning = enable_cleaning
            self.test_mode = test_mode
            
            # 标签编码器 - 只对流体类型标签进行编码
            self.label_encoder = LabelEncoder()
            if len(self.labels) > 0:
                # 过滤出真正的流体类型标签，排除井名
                fluid_types = ['油层', '水层', '干层', '差油层']
                filtered_labels = []
                for label in self.labels:
                    if label in fluid_types:
                        filtered_labels.append(label)
                    elif label in ['油水同层', '含油水层', '油水层']:
                        # 将其他油层相关类型归为油层
                        filtered_labels.append('油层')
                    else:
                        # 如果标签不是流体类型，使用默认标签
                        filtered_labels.append('干层')
                
                self.labels_encoded = self.label_encoder.fit_transform(filtered_labels)
            else:
                self.labels_encoded = np.array([])
            
            print(f"   数据集大小: {len(self.data)}")
            print(f"   标签类别: {list(set(self.labels))}")
            print(f"   序列长度: {self.sequence_length}")
            print(f"   图像尺寸: {self.image_size}")
            
        else:
            # 原有的加载逻辑
            self.welldata_dir = welldata_dir
            self.curve_names = curve_names or ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
            self.sequence_length = sequence_length
            self.enable_wavelet = enable_wavelet
            self.image_size = image_size or (16, 16)  # 使用配置或默认值
            self.enable_cleaning = enable_cleaning
            self.test_mode = test_mode
            
            # 数据清洗器
            self.data_cleaner = WellLogDataCleaner(self.curve_names)
            
            # 标签编码器 - 只对流体类型标签进行编码
            self.label_encoder = LabelEncoder()
            
            # 加载所有井数据
            self.data, self.labels = self._load_all_well_data()
            
            # 编码标签 - 只对流体类型标签进行编码
            if len(self.labels) > 0:
                # 过滤出真正的流体类型标签，排除井名
                fluid_types = ['油层', '水层', '干层', '差油层']
                filtered_labels = []
                for label in self.labels:
                    if label in fluid_types:
                        filtered_labels.append(label)
                    elif label in ['油水同层', '含油水层', '油水层']:
                        # 将其他油层相关类型归为油层
                        filtered_labels.append('油层')
                    else:
                        # 如果标签不是流体类型，使用默认标签
                        filtered_labels.append('干层')
                
                self.labels_encoded = self.label_encoder.fit_transform(filtered_labels)
            else:
                self.labels_encoded = np.array([])
            
            print(f"📊 批量数据加载完成:")
            print(f"   总数据量: {len(self.data)}")
            if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
                print(f"   标签类别: {list(self.label_encoder.classes_)}")
            else:
                print(f"   标签类别: {list(set(self.labels))}")
        print(f"   序列长度: {self.sequence_length}")
        print(f"   图像尺寸: {self.image_size}")
    
    def _load_all_well_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """加载所有井的数据"""
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
                    
                    # 生成序列（带标签）
                    sequences_with_labels = self._generate_sequences(well_data, well_name)
                    
                    if len(sequences_with_labels) > 0:
                        seqs, seq_labels = zip(*sequences_with_labels)
                        all_data.extend(seqs)
                        all_labels.extend(seq_labels)
                        
                        if self.test_mode and len(all_data) >= 100:  # 测试模式限制数据量
                            break
                    
            except Exception as e:
                print(f"   ❌ 加载 {well_name} 失败: {e}")
                continue
        
        print(f"✅ 成功加载 {len(all_data)} 个数据序列")
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
            required_curves = ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
            available_curves = [col for col in data.columns if col in required_curves]
            
            if len(available_curves) < 3:  # 至少需要3个测井曲线
                print(f"   ⚠️  {file_path} 缺少必要的测井曲线")
                return None
            
            # 保留可用的测井曲线与解释结论列（若存在）
            label_col = '解释结论'
            keep_cols = available_curves + ([label_col] if label_col in data.columns else [])
            data = data[keep_cols].copy()
            
            # 处理缺失值 - 修复pandas FutureWarning
            data = data.ffill().bfill()
            
            # 移除无效行
            data = data.dropna()
            
            if len(data) < self.sequence_length:
                print(f"   ⚠️  {file_path} 数据量不足 ({len(data)} < {self.sequence_length})")
                return None
            
            return data
            
        except Exception as e:
            print(f"   ❌ 读取 {file_path} 失败: {e}")
            return None
    
    def _clean_well_data(self, data: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """清洗井数据并过滤有效解释数据"""
        try:
            # 先过滤出有解释结论的有效数据（在归一化之前）
            valid_data = self._filter_valid_interpretation_data(data, well_name)
            
            if len(valid_data) == 0:
                print(f"   ⚠️  {well_name} 过滤后没有有效数据")
                return valid_data
            
            # 然后使用数据清洗器进行清洗和归一化
            cleaned_data = self.data_cleaner.clean_data(valid_data)
            
            return cleaned_data
        except Exception as e:
            print(f"   ⚠️  清洗 {well_name} 数据失败: {e}")
            return data
    
    def _filter_valid_interpretation_data(self, data: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """过滤出有解释结论的有效数据 - 放宽过滤条件"""
        try:
            # 大幅放宽各测井曲线的合理值范围
            valid_ranges = {
                'GR': (-50, 300),     # 自然伽马 (API) - 大幅放宽
                'SP': (-100, 100),    # 自然电位 (mV) - 大幅放宽
                'AC': (50, 500),      # 声波时差 (μs/ft) - 大幅放宽
                'DEN': (1.0, 4.0),    # 密度 (g/cm³) - 大幅放宽
                'CNL': (-10, 60),     # 中子孔隙度 (%) - 大幅放宽
                'RT': (0.1, 200)      # 电阻率 (Ω·m) - 大幅放宽
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
            
            # 如果过滤后数据太少，进一步放宽条件
            if len(filtered_data) < len(data) * 0.1:  # 如果过滤后少于10%的数据
                print(f"     ⚠️  {well_name} 过滤后数据过少，使用更宽松的条件")
                
                # 使用更宽松的范围
                loose_ranges = {
                    'GR': (-100, 500),    # 极宽松范围
                    'SP': (-200, 200),    # 极宽松范围
                    'AC': (0, 1000),      # 极宽松范围
                    'DEN': (0.5, 5.0),    # 极宽松范围
                    'CNL': (-20, 80),     # 极宽松范围
                    'RT': (0.01, 1000)    # 极宽松范围
                }
                
                # 重新过滤 - 只对数值列进行过滤
                loose_mask = pd.Series([True] * len(data), index=data.index)
                for curve_name in data.columns:
                    if curve_name in loose_ranges:
                        # 只对数值列进行范围过滤，跳过标签列
                        if curve_name in ['解释结论', '结论', '岩性', '含油性']:
                            continue
                        
                        try:
                            min_val, max_val = loose_ranges[curve_name]
                            curve_mask = (data[curve_name] >= min_val) & (data[curve_name] <= max_val)
                            loose_mask = loose_mask & curve_mask
                        except (TypeError, ValueError) as e:
                            # 如果列包含非数值数据，跳过该列的过滤
                            print(f"     ⚠️  跳过非数值列 {curve_name} 的宽松过滤: {e}")
                            continue
                
                filtered_data = data[loose_mask].copy()
            
            # 异常值过滤：使用更宽松的标准（10倍标准差）- 只对数值列进行过滤
            if len(filtered_data) > 10:
                outlier_mask = pd.Series([True] * len(filtered_data), index=filtered_data.index)
                
                for curve_name in data.columns:
                    if curve_name in filtered_data.columns:
                        # 只对数值列进行异常值过滤，跳过标签列
                        if curve_name in ['解释结论', '结论', '岩性', '含油性']:
                            continue
                        
                        try:
                            mean_val = filtered_data[curve_name].mean()
                            std_val = filtered_data[curve_name].std()
                            
                            # 使用10倍标准差，非常宽松
                            if std_val > 0:
                                z_score = np.abs((filtered_data[curve_name] - mean_val) / std_val)
                                curve_outlier_mask = z_score <= 10.0  # 从5倍改为10倍
                                outlier_mask = outlier_mask & curve_outlier_mask
                        except (TypeError, ValueError) as e:
                            # 如果列包含非数值数据，跳过该列的异常值过滤
                            print(f"     ⚠️  跳过非数值列 {curve_name} 的异常值过滤: {e}")
                            continue
                
                filtered_data = filtered_data[outlier_mask].copy()
            
            print(f"     {well_name} 数据过滤: {len(data)} -> {len(filtered_data)} 行 ({len(filtered_data)/len(data)*100:.1f}%)")
            
            # 如果最终数据仍然太少，返回原始数据
            if len(filtered_data) < len(data) * 0.05:  # 如果少于5%的数据
                print(f"     ⚠️  {well_name} 过滤后数据极少，返回原始数据")
                return data
            
            return filtered_data
            
        except Exception as e:
            print(f"     ⚠️  {well_name} 数据过滤失败: {e}")
            return data
    
    def _clean_label_string(self, label: str) -> str:
        """清理标签字符串，处理重复模式"""
        if not label or label.strip() == '':
            return 'unknown'
            
        label = str(label).strip()
        
        # 检查是否为重复模式（如"水层水层水层..."）
        if len(label) >= 4:
            # 检查2-4字符的重复模式
            for pattern_len in [2, 3, 4]:
                if len(label) >= pattern_len * 2:
                    pattern = label[:pattern_len]
                    # 检查整个字符串是否由该模式重复构成
                    if all(label[i:i+pattern_len] == pattern for i in range(0, len(label), pattern_len) if i + pattern_len <= len(label)):
                        # 如果剩余部分也是该模式的前缀
                        remainder = len(label) % pattern_len
                        if remainder == 0 or label[-remainder:] == pattern[:remainder]:
                            return pattern
        
        # 标准化标签为5个主要流体类型：油层、水层、干层、差油层、油水层
        # 非这5类的归为"无"类
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
        
        return label_mapping.get(label, None)  # 未匹配的标签返回None，将被过滤

    def _generate_sequences(self, data: pd.DataFrame, well_name: Optional[str] = None) -> List[Tuple[np.ndarray, str]]:
        """生成序列数据，并为每个窗口分配标签
        优先使用'解释结论'等解释列作为标签；若不存在则回退到井名。
        仅当窗口内解释标签一致度足够（例如≥70%）时才保留该序列。
        """
        sequences_with_labels: List[Tuple[np.ndarray, str]] = []
        data_length = len(data)

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

        # 滑动窗口生成序列
        step = max(1, self.sequence_length // 2)
        for i in range(0, data_length - self.sequence_length + 1, step):  # 50%重叠
            window_df = data.iloc[i:i + self.sequence_length]
            sequence = window_df[numeric_cols].values

            if sequence.shape[0] != self.sequence_length:
                continue

            # 确定标签
            label: Optional[str] = None
            if label_col is not None:
                labels_window = window_df[label_col].astype(str).str.strip()
                labels_window = labels_window[labels_window.notna() & (labels_window != '') & (labels_window.str.lower() != 'nan')]
                if len(labels_window) > 0:
                    # 清理标签字符串
                    cleaned_labels = [self._clean_label_string(l) for l in labels_window]
                    cleaned_labels = [l for l in cleaned_labels if l != 'unknown']
                    
                    if cleaned_labels:
                        # 获取最常见的清理后标签
                        from collections import Counter
                        label_counts = Counter(cleaned_labels)
                        top_label = label_counts.most_common(1)[0][0]
                        ratio = label_counts[top_label] / len(cleaned_labels)
                        
                        # 要求窗口内≥60%为同一解释结论（放宽）
                        if ratio >= 0.6:
                            label = top_label
            
            # 回退到井名
            if label is None:
                label = well_name if well_name is not None else 'unknown'

            sequences_with_labels.append((sequence, label))

        return sequences_with_labels
    
    def _generate_wavelet_features(self, sequence_data: np.ndarray) -> np.ndarray:
        """生成小波包分解时频图谱特征 - 增强版本"""
        try:
            features = []
            
            for i, curve_name in enumerate(self.curve_names):
                if i < sequence_data.shape[0]:
                    curve_data = sequence_data[i]
                    
                    # 数据验证和预处理
                    if np.isnan(curve_data).any() or np.isinf(curve_data).any():
                        print(f"   ⚠️  {curve_name} 曲线包含NaN/Inf值，使用零填充")
                        curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 数据标准化
                    if np.std(curve_data) > 0:
                        curve_data = (curve_data - np.mean(curve_data)) / np.std(curve_data)
                    else:
                        curve_data = np.zeros_like(curve_data)
                    
                    # 确保数据长度足够
                    if len(curve_data) < self.sequence_length:
                        # 如果数据不足，进行填充
                        padding = np.zeros(self.sequence_length - len(curve_data))
                        curve_data = np.concatenate([curve_data, padding])
                    elif len(curve_data) > self.sequence_length:
                        # 如果数据过多，进行截断
                        curve_data = curve_data[:self.sequence_length]
                    
                    # 生成小波包分解时频图谱
                    wavelet_features = self._generate_wavelet_packet_spectrogram(curve_data, curve_name)
                    features.append(wavelet_features)
                else:
                    # 如果曲线数量不足，使用零填充
                    features.append(np.zeros(self.sequence_length))
            
            # 转换为numpy数组
            features = np.array(features)
            
            # 调整特征尺寸
            features = self._resize_features(features)
            
            return features
            
        except Exception as e:
            print(f"   ⚠️  生成小波特征失败: {e}")
            # 备选方案：使用原始数据
            return self._resize_features(sequence_data)
    
    def _generate_wavelet_packet_spectrogram(self, curve_data: np.ndarray, curve_name: str) -> np.ndarray:
        """生成时频图谱 - 完全修复版本，避免小波包分解问题"""
        try:
            # 数据预处理：确保没有NaN/Inf值
            curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 如果数据全为零，返回简单的时频图谱
            if np.all(curve_data == 0):
                return np.zeros((4, self.sequence_length))
            
            # 使用简单的方法生成时频图谱，避免复杂的小波分解
            try:
                # 方法1：使用滑动窗口生成多尺度特征
                window_sizes = [8, 16, 32, 64]  # 不同的窗口大小
                coeffs_matrix = []
                
                for window_size in window_sizes:
                    if len(curve_data) >= window_size:
                        # 计算滑动窗口的统计特征
                        window_features = []
                        for i in range(0, len(curve_data) - window_size + 1, window_size // 2):
                            window_data = curve_data[i:i + window_size]
                            if len(window_data) == window_size:
                                # 计算窗口的统计特征
                                mean_val = np.mean(window_data)
                                std_val = np.std(window_data)
                                max_val = np.max(window_data)
                                min_val = np.min(window_data)
                                
                                # 组合特征
                                features = np.array([mean_val, std_val, max_val, min_val])
                                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                                window_features.append(features)
                        
                        if len(window_features) > 0:
                            # 将特征插值到固定长度
                            window_features = np.array(window_features)
                            if window_features.ndim == 1:
                                window_features = window_features.reshape(1, -1)
                            
                            # 对每个特征维度进行插值
                            resized_features = []
                            for j in range(window_features.shape[1]):
                                feature_series = window_features[:, j]
                                resized_feature = self._interpolate_to_length(feature_series, self.sequence_length)
                                resized_features.append(resized_feature)
                            
                            # 组合所有特征
                            combined_features = np.array(resized_features)
                            coeffs_matrix.append(combined_features)
                
                # 如果没有生成足够的特征，使用原始数据
                if len(coeffs_matrix) == 0:
                    # 使用原始数据生成简单的时频图谱
                    coeffs_matrix = [curve_data.reshape(1, -1)]
                
                # 转换为时频图谱
                spectrogram = np.vstack(coeffs_matrix)
                
            except Exception as e:
                print(f"     ⚠️  {curve_name} 时频图谱生成失败: {e}，使用原始数据")
                # 使用原始数据生成简单的时频图谱
                spectrogram = curve_data.reshape(1, -1)
            
            # 检查时频图谱是否包含NaN/Inf值
            if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
                print(f"     ⚠️  {curve_name} 时频图谱包含NaN/Inf值，进行清理")
                spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 确保输出尺寸合理
            if spectrogram.shape[0] > 16:  # 限制最大行数
                spectrogram = spectrogram[:16]
            elif spectrogram.shape[0] < 4:  # 确保最小行数
                # 重复最后一行
                last_row = spectrogram[-1] if len(spectrogram) > 0 else curve_data
                last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)
                while spectrogram.shape[0] < 4:
                    spectrogram = np.vstack([spectrogram, last_row])
            
            # 最终检查
            if np.isnan(spectrogram).any() or np.isinf(spectrogram).any():
                print(f"     ⚠️  {curve_name} 最终时频图谱包含NaN/Inf值，使用零填充")
                spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
            
            return spectrogram
            
        except Exception as e:
            print(f"     ⚠️  {curve_name} 时频图谱生成失败: {e}")
            # 返回零填充的默认时频图谱
            return np.zeros((4, self.sequence_length))
    
    def _interpolate_to_length(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """将数据插值到目标长度 - 修复NaN/Inf值问题"""
        try:
            # 预处理：清理NaN/Inf值
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if len(data) == target_length:
                return data
            elif len(data) == 0:
                return np.zeros(target_length)
            
            # 使用线性插值
            x_old = np.linspace(0, 1, len(data))
            x_new = np.linspace(0, 1, target_length)
            
            from scipy.interpolate import interp1d
            f = interp1d(x_old, data, kind='linear', bounds_error=False, fill_value=0)
            result = f(x_new)
            
            # 检查插值结果是否包含NaN/Inf值
            if np.isnan(result).any() or np.isinf(result).any():
                print(f"     ⚠️  插值结果包含NaN/Inf值，使用零填充")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
            return result
            
        except Exception as e:
            print(f"     ⚠️  插值失败: {e}")
            # 简单填充或截断
            if len(data) > target_length:
                result = data[:target_length]
            else:
                padding = np.zeros(target_length - len(data))
                result = np.concatenate([data, padding])
            
            # 确保结果不包含NaN/Inf值
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            return result
    
    def _resize_features(self, features: np.ndarray) -> np.ndarray:
        """调整特征尺寸 - 优化版本"""
        try:
            from scipy.ndimage import zoom
            
            # 确保features是3D数组 (curves, height, width)
            if len(features.shape) == 1:
                # 如果是1D，扩展为2D
                features = features.reshape(1, -1)
            
            if len(features.shape) == 2:
                # 如果是2D，扩展为3D
                features = features.reshape(features.shape[0], features.shape[1], 1)
            
            target_shape = (len(self.curve_names), *self.image_size)
            
            # 数据验证
            if np.isnan(features).any() or np.isinf(features).any():
                print(f"   ⚠️  特征包含NaN/Inf值，进行清理")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 计算缩放因子
            zoom_factors = []
            for i in range(3):
                if features.shape[i] > 0:
                    factor = target_shape[i] / features.shape[i]
                    # 限制缩放因子范围，避免过度缩放
                    factor = max(0.1, min(factor, 10.0))
                else:
                    factor = 1.0
                zoom_factors.append(factor)
            
            # 使用scipy.ndimage.zoom进行缩放
            resized_features = zoom(features, zoom_factors, order=1, prefilter=False)
            
            # 确保输出尺寸正确
            if resized_features.shape != target_shape:
                # 如果尺寸仍然不匹配，使用填充或裁剪
                final_features = np.zeros(target_shape)
                for i in range(min(len(self.curve_names), resized_features.shape[0])):
                    for j in range(min(self.image_size[0], resized_features.shape[1])):
                        for k in range(min(self.image_size[1], resized_features.shape[2])):
                            final_features[i, j, k] = resized_features[i, j, k]
                return final_features
            
            # 最终数据验证
            if np.isnan(resized_features).any() or np.isinf(resized_features).any():
                print(f"   ⚠️  调整后的特征包含NaN/Inf值，使用零填充")
                resized_features = np.nan_to_num(resized_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return resized_features
            
        except Exception as e:
            print(f"   ⚠️  调整特征尺寸失败: {e}")
            # 返回零填充的默认特征
            default_features = np.zeros((len(self.curve_names), *self.image_size))
            return default_features
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        sequence = self.data[idx]
        label = self.labels_encoded[idx]
        
        if self.enable_wavelet:
            # 生成小波特征
            features = self._generate_wavelet_features(sequence)
            
            # 确保特征维度正确
            if features.ndim == 2:
                # 如果是2D，扩展为3D (curves, height, width)
                features = features.reshape(len(self.curve_names), *self.image_size)
            elif features.ndim == 1:
                # 如果是1D，扩展为3D
                features = features.reshape(1, *self.image_size)
                # 如果通道数不足，进行填充
                if features.shape[0] < len(self.curve_names):
                    padding = np.zeros((len(self.curve_names) - features.shape[0], *self.image_size))
                    features = np.concatenate([features, padding], axis=0)
            
            # 确保输出形状为 (curves, height, width)
            if features.shape[0] != len(self.curve_names):
                if features.shape[0] < len(self.curve_names):
                    # 填充缺失的通道
                    padding = np.zeros((len(self.curve_names) - features.shape[0], *self.image_size))
                    features = np.concatenate([features, padding], axis=0)
                else:
                    # 截断多余的通道
                    features = features[:len(self.curve_names)]
            
            sequence_tensor = torch.FloatTensor(features)
        else:
            # 使用原始序列，确保维度正确
            if sequence.ndim == 2:
                # 如果序列是2D (curves, sequence_length)
                if sequence.shape[0] != len(self.curve_names):
                    # 调整通道数
                    if sequence.shape[0] < len(self.curve_names):
                        # 填充缺失的通道
                        padding = np.zeros((len(self.curve_names) - sequence.shape[0], sequence.shape[1]))
                        sequence = np.concatenate([sequence, padding], axis=0)
                    else:
                        # 截断多余的通道
                        sequence = sequence[:len(self.curve_names)]
                
                # 调整序列长度到目标尺寸
                target_length = self.image_size[0] * self.image_size[1]
                if sequence.shape[1] != target_length:
                    if sequence.shape[1] > target_length:
                        # 截断
                        sequence = sequence[:, :target_length]
                    else:
                        # 填充
                        padding = np.zeros((sequence.shape[0], target_length - sequence.shape[1]))
                        sequence = np.concatenate([sequence, padding], axis=1)
                
                # 重塑为 (curves, height, width)
                sequence = sequence.reshape(len(self.curve_names), *self.image_size)
            else:
                # 如果是1D，扩展为3D
                sequence = sequence.reshape(1, -1)
                # 调整到目标尺寸
                target_length = self.image_size[0] * self.image_size[1]
                if sequence.shape[1] != target_length:
                    if sequence.shape[1] > target_length:
                        sequence = sequence[:, :target_length]
                    else:
                        padding = np.zeros((1, target_length - sequence.shape[1]))
                        sequence = np.concatenate([sequence, padding], axis=1)
                
                # 重塑并填充通道
                sequence = sequence.reshape(1, *self.image_size)
                padding = np.zeros((len(self.curve_names) - 1, *self.image_size))
                sequence = np.concatenate([sequence, padding], axis=0)
            
            sequence_tensor = torch.FloatTensor(sequence)
        
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
        
        return sequence_tensor, torch.LongTensor([label]).squeeze()  # 确保标签是1D


def create_batch_data_loaders(
    welldata_dir: str = "welldata",
    curve_names: List[str] = None,
    sequence_length: int = 100,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    test_ratio: float = 0.0,  # 不设置测试集
    enable_wavelet: bool = True,
    image_size: Tuple[int, int] = None,  # 移除硬编码
    enable_cleaning: bool = True,
    num_workers: int = 0,
    test_mode: bool = False
) -> Dict[str, DataLoader]:
    """
    创建批量数据加载器
    
    Returns:
        包含训练、验证数据加载器的字典
    """
    
    # 创建数据集
    dataset = BatchWellLogDataset(
        welldata_dir=welldata_dir,
        curve_names=curve_names,
        sequence_length=sequence_length,
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


def create_batch_data_loaders_with_config(config, welldata_dir="welldata", batch_size=16, test_mode=False):
    """
    使用配置对象创建批量数据加载器
    
    Args:
        config: 配置对象
        welldata_dir: 井数据目录路径
        batch_size: 批次大小
        test_mode: 测试模式
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    try:
        print("📊 创建批量数据加载器...")
        
        # 获取配置参数
        curve_names = config.DATA_CONFIG['curve_names']
        sequence_length = config.DATA_CONFIG['sequence_length']
        image_size = config.DATA_CONFIG['image_size']
        train_ratio = config.DATA_CONFIG['train_ratio']
        val_ratio = config.DATA_CONFIG['val_ratio']
        
        # 数据清洗器
        data_cleaner = WellLogDataCleaner(curve_names)
        
        # 标签编码器
        label_encoder = LabelEncoder()
        
        # 加载所有井数据
        data_files = glob.glob(os.path.join(welldata_dir, "*.txt"))
        
        if not data_files:
            print(f"❌ 在 {welldata_dir} 目录中未找到数据文件")
            return None, None
        
        print(f"🔍 发现 {len(data_files)} 个井数据文件")
        
        all_data = []
        all_labels = []
        
        for file_path in data_files:
            well_name = os.path.basename(file_path).replace('.txt', '')
            print(f"   正在加载: {well_name}")
            
            try:
                # 加载井数据
                well_data = _load_single_well_data(file_path, curve_names, sequence_length)
                
                if well_data is not None and len(well_data) > 0:
                    # 数据清洗
                    well_data = _clean_well_data(well_data, well_name, data_cleaner)
                    
                    # 生成序列
                    sequences = _generate_sequences(well_data, sequence_length)
                    
                    if len(sequences) > 0:
                        all_data.extend(sequences)
                        all_labels.extend([well_name] * len(sequences))
                        
                        if test_mode and len(all_data) >= 100:  # 测试模式限制数据量
                            break
                    
            except Exception as e:
                print(f"   ❌ 加载 {well_name} 失败: {e}")
                continue
        
        if len(all_data) == 0:
            print("❌ 没有成功加载任何数据")
            return None, None
        
        print(f"✅ 成功加载 {len(all_data)} 个数据序列")
        
        # 编码标签
        labels_encoded = label_encoder.fit_transform(all_labels)
        
        # 转换为numpy数组
        all_data = np.array(all_data)
        labels_encoded = np.array(labels_encoded)
        
        # 数据集分割
        total_samples = len(all_data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        # 随机打乱数据
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        
        # 创建数据集
        train_data = all_data[train_indices]
        train_labels = labels_encoded[train_indices]
        val_data = all_data[val_indices]
        val_labels = labels_encoded[val_indices]
        
        print(f"📊 数据集分割:")
        print(f"   训练集: {len(train_data)}")
        print(f"   验证集: {len(val_data)}")
        
        # 创建简单的张量数据集
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(train_data),
            torch.LongTensor(train_labels)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(val_data),
            torch.LongTensor(val_labels)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # Windows下设为0避免多进程问题
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Windows下设为0避免多进程问题
            pin_memory=True
        )
        
        print("✅ 批量数据加载器创建成功")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(val_dataset)} 样本")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ 创建批量数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _load_single_well_data(file_path: str, curve_names: List[str], sequence_length: int) -> Optional[pd.DataFrame]:
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
        available_curves = [col for col in data.columns if col in curve_names]
        
        if len(available_curves) < 3:  # 至少需要3个测井曲线
            print(f"   ⚠️  {file_path} 缺少必要的测井曲线")
            return None
        
        # 只保留可用的测井曲线
        data = data[available_curves].copy()
        
        # 处理缺失值 - 修复pandas FutureWarning
        data = data.ffill().bfill()
        
        # 移除无效行
        data = data.dropna()
        
        if len(data) < sequence_length:
            print(f"   ⚠️  {file_path} 数据量不足 ({len(data)} < {sequence_length})")
            return None
        
        return data
        
    except Exception as e:
        print(f"   ❌ 读取 {file_path} 失败: {e}")
        return None


def _clean_well_data(data: pd.DataFrame, well_name: str, data_cleaner: WellLogDataCleaner) -> pd.DataFrame:
    """清洗井数据"""
    try:
        # 使用数据清洗器
        cleaned_data = data_cleaner.clean_data(data)
        return cleaned_data
    except Exception as e:
        print(f"   ⚠️  清洗 {well_name} 数据失败: {e}")
        return data


def _generate_sequences(data: pd.DataFrame, sequence_length: int) -> List[np.ndarray]:
    """生成序列数据"""
    sequences = []
    data_length = len(data)
    
    # 滑动窗口生成序列
    for i in range(0, data_length - sequence_length + 1, sequence_length // 2):  # 50%重叠
        sequence = data.iloc[i:i + sequence_length].values
        
        if sequence.shape[0] == sequence_length:
            sequences.append(sequence)
    
    return sequences


def test_batch_data_loader():
    """测试批量数据加载器"""
    print("🧪 测试批量数据加载器...")
    
    try:
        # 创建数据集
        dataset = BatchWellLogDataset(
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
    test_batch_data_loader()
