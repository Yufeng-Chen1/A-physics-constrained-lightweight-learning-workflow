#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于时频图谱子区域的数据加载器
将时频图谱的每一段（二维子区域）作为数据集
尺度范围: 5-36个采样点长度
时间窗口: 32个采样点长度
采样时间步长: 1个采样点长度
特征图尺寸: 64*64*6 (6个通道对应6条测井曲线)
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import pywt
from scipy import signal
try:
    from .enhanced_time_frequency_extractor import create_enhanced_extractor
except ImportError:
    from enhanced_time_frequency_extractor import create_enhanced_extractor
from scipy.interpolate import interp1d

# 导入流体类型定义
try:
    from .fluid_types import FluidTypes, get_num_fluid_classes
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fluid_types import FluidTypes, get_num_fluid_classes

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


class SpectrogramWellLogDataset(Dataset):
    """
    基于时频图谱子区域的井数据加载器
    将时频图谱的每一段作为数据集
    支持6种流体类型：油层、水层、干层、差油层、油水同层、含油水层
    """
    
    def __init__(self, 
                 welldata_dir: str = "welldata",
                 curve_names: List[str] = None,
                 scale_range: Tuple[int, int] = (5, 36),
                 time_window: int = 32,
                 time_step: int = 1,
                 feature_size: Tuple[int, int] = (64, 64),
                 enable_cleaning: bool = True,
                 test_mode: bool = False,
                 use_fluid_mapping: bool = True,
                 wavelet_config: Dict = None):
        """
        初始化基于时频图谱的数据加载器
        
        Args:
            welldata_dir: 井数据目录路径
            curve_names: 测井曲线名称列表
            scale_range: 尺度范围 (min_scale, max_scale)
            time_window: 时间窗口长度
            time_step: 采样时间步长
            feature_size: 特征图尺寸 (height, width)
            enable_cleaning: 是否启用数据清洗
            test_mode: 测试模式，只加载少量数据
        """
        self.welldata_dir = welldata_dir
        self.curve_names = curve_names or ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
        self.scale_range = scale_range
        self.time_window = time_window
        self.time_step = time_step
        self.feature_size = feature_size
        self.enable_cleaning = enable_cleaning
        self.test_mode = test_mode
        self.use_fluid_mapping = use_fluid_mapping
        self.wavelet_config = wavelet_config or {}
        
        # 初始化增强特征提取器
        extractor_config = {
            'feature_size': feature_size,
            'time_window': time_window,
            'overlap_ratio': 0.75,
            'decomposition_levels': [2, 3, 4]
        }
        self.enhanced_extractor = create_enhanced_extractor(extractor_config)
        self.use_enhanced_extraction = self.wavelet_config.get('use_enhanced_extraction', True)
        
        # 标签聚合模式：center/majority
        try:
            self._label_mode = str(self.wavelet_config.get('label_mode', 'center')).strip().lower()
            if self._label_mode not in ('center', 'majority'):
                self._label_mode = 'center'
        except Exception:
            self._label_mode = 'center'
        # per-curve window 支持：为每条曲线解析专属窗口，并计算全局滑窗长度
        self._per_curve_windows: Dict[str, int] = {}
        # per-curve output_size 支持：为每条曲线解析目标输出尺寸（先生成到该尺寸，再统一到 feature_size）
        self._per_curve_output_size: Dict[str, Tuple[int, int]] = {}
        try:
            if isinstance(self.wavelet_config, dict) and 'curve_configs' in self.wavelet_config:
                for cname in (self.curve_names or []):
                    cfg = self.wavelet_config['curve_configs'].get(cname, {})
                    tw = int(cfg.get('time_window', self.time_window))
                    self._per_curve_windows[cname] = max(1, tw)
                    osz = cfg.get('output_size', self.feature_size)
                    try:
                        if isinstance(osz, (list, tuple)) and len(osz) == 2:
                            h, w = int(osz[0]), int(osz[1])
                            self._per_curve_output_size[cname] = (max(1, h), max(1, w))
                        else:
                            self._per_curve_output_size[cname] = self.feature_size
                    except Exception:
                        self._per_curve_output_size[cname] = self.feature_size
        except Exception:
            pass
        # 全局滑窗长度：用于对齐标签与步长（取所有曲线窗口的最大值）
        if self._per_curve_windows:
            self._global_time_window = max(self._per_curve_windows.get(c, self.time_window) for c in self.curve_names)
        else:
            self._global_time_window = self.time_window

        # 数据集级标准化统计（在数据加载器创建后由训练集计算并设置）
        self.dataset_channel_mean: Optional[np.ndarray] = None
        self.dataset_channel_std: Optional[np.ndarray] = None
        # 根据重叠率计算 time_step（例如 75% 重叠 -> 步长 = window * (1-0.75) = 0.25*window）
        try:
            if isinstance(self.wavelet_config, dict) and 'overlap_ratio' in self.wavelet_config:
                # 当设置 force_time_step=True 时，保持调用者传入的 time_step 不被重写
                if not bool(self.wavelet_config.get('force_time_step', False)):
                    overlap = float(self.wavelet_config.get('overlap_ratio', 0.0))
                    overlap = min(max(overlap, 0.0), 0.95)
                    # 步长基于全局窗口
                    step = max(1, int(self._global_time_window * (1.0 - overlap)))
                    self.time_step = step
        except Exception:
            pass
        # 多數票閾值（僅影響訓練集，驗證/測試不變）：低於該比例的窗口將丟棄或標為“無”
        try:
            # 支持环境变量覆盖窗口多数票阈值
            try:
                _env_maj = os.environ.get('MAJ_THRESH', '').strip()
                _maj_val = float(_env_maj) if _env_maj else float(self.wavelet_config.get('majority_threshold', 0.0))
            except Exception:
                _maj_val = float(self.wavelet_config.get('majority_threshold', 0.0))
            self._majority_threshold = _maj_val
            self._apply_threshold_discard = bool(self.wavelet_config.get('threshold_discard', True))
        except Exception:
            self._majority_threshold = 0.0
            self._apply_threshold_discard = True
        
        # 井名到流体类型的映射（模拟数据，实际应用中需要真实标注）
        # 19口井的完整映射，按井系列和流体类型分布
        self.well_to_fluid_mapping = {
            # Geng系列井 (12口)
            'geng120': 0,  # 油层
            'geng166': 0,  # 油层
            'geng181': 1,  # 水层
            'geng203': 2,  # 干层
            'geng207': 3,  # 差油层
            'geng217': 4,  # 油水层
            'geng219': 4,  # 油水层
            'geng220': 0,  # 油层
            'geng221': 3,  # 差油层 (根据实际解释结论，差油层占多数)
            'geng343': 2,  # 干层
            'geng60': 2,   # 干层
            'geng86': 3,   # 差油层
            
            # Ji系列井 (3口)
            'ji117': 4,    # 油水层
            'ji121': 1,    # 水层
            'jian39': 3,   # 差油层
            
            # 其他井 (4口)
            'di199-48': 0, # 油层 (根据实际解释结论，油层占多数)
            'di200-47': 3, # 差油层 (根据实际解释结论，差油层占多数)
            'luo160': 2,   # 干层
            'yuan3': 4     # 油水层
        }
        
        # 标签编码器
        self.label_encoder = None  # 不再用于训练，仅保留字段占位
        
        # 加载所有井数据并生成时频图谱
        self.data, self.labels, self.labels_ids, self.well_names = self._load_and_generate_spectrograms()
        
        # 编码标签
        self.labels_encoded = np.array(self.labels_ids, dtype=np.int64) if len(self.labels_ids) > 0 else np.array([])
        
        print(f"📊 基于时频图谱的数据加载完成:")
        print(f"   总数据量: {len(self.data)}")
        print(f"   尺度范围: {self.scale_range}")
        print(f"   时间窗口: {self.time_window}")
        print(f"   时间步长: {self.time_step}")
        print(f"   特征图尺寸: {self.feature_size}")
        print(f"   流体类型映射: {'启用' if self.use_fluid_mapping else '禁用'}")
        
        # 打印标签类别名称
        if len(self.labels) > 0:
            print(f"   标签类别: {sorted(list(set(self.labels)))}")
            if self.use_fluid_mapping:
                print(f"   流体类型总数: {len(set(self.labels_ids))}")
        
        # 显示流体类型分布
        if len(self.labels) > 0:
            from collections import Counter
            label_counts = Counter(self.labels)
            all_names = ['油层', '水层', '干层', '差油层', '油水层', '无']
            print(f"   流体类型分布:")
            for fluid_name in all_names:
                count = label_counts.get(fluid_name, 0)
                percentage = (count / len(self.labels) * 100) if len(self.labels) > 0 else 0.0
                print(f"     {fluid_name}: {count} 样本 ({percentage:.1f}%)")
    
    def set_dataset_normalization_stats(self, channel_mean: np.ndarray, channel_std: np.ndarray) -> None:
        """设置按通道的数据集级标准化统计（mean/std）。"""
        try:
            cm = np.asarray(channel_mean, dtype=np.float32)
            cs = np.asarray(channel_std, dtype=np.float32)
            if cm.shape[0] != len(self.curve_names) or cs.shape[0] != len(self.curve_names):
                print(f"   ⚠️  标准化统计维度不匹配: mean={cm.shape}, std={cs.shape}, channels={len(self.curve_names)}")
            cs[cs < 1e-6] = 1.0
            self.dataset_channel_mean = cm
            self.dataset_channel_std = cs
            print("   ✅ 已设置数据集级标准化统计 (per-channel)")
        except Exception as e:
            print(f"   ⚠️  设置数据集级标准化统计失败: {e}")

    def _load_and_generate_spectrograms(self) -> Tuple[List[np.ndarray], List[str], List[int], List[str]]:
        """加载所有井数据并生成时频图谱"""
        data_files = glob.glob(os.path.join(self.welldata_dir, "*.txt"))
        
        if not data_files:
            print(f"❌ 在 {self.welldata_dir} 目录中未找到数据文件")
            return [], [], [], []
        
        print(f"🔍 发现 {len(data_files)} 个井数据文件")
        
        all_spectrograms = []
        all_labels = []
        all_label_ids = []
        all_well_names = []
        
        for file_path in data_files:
            well_name = os.path.basename(file_path).replace('.txt', '')
            print(f"   正在处理: {well_name}")
            
            try:
                # 加载井数据
                well_data = self._load_single_well_data(file_path)
                
                if well_data is not None and len(well_data) > 0:
                    # 数据清洗
                    if self.enable_cleaning:
                        well_data = self._clean_well_data(well_data, well_name)
                    
                    # 生成时频图谱与窗口标签
                    spectrograms, window_label_ids, window_label_names = self._generate_well_spectrograms(well_data, well_name)
                    
                    if len(spectrograms) > 0:
                        all_spectrograms.extend(spectrograms)
                        
                        if self.use_fluid_mapping:
                            # 仍按井映射赋值
                            fluid_type_id = self.well_to_fluid_mapping.get(well_name, 0)
                            fluid_name = FluidTypes.get_fluid_name(fluid_type_id)
                            all_labels.extend([fluid_name] * len(spectrograms))
                            all_label_ids.extend([fluid_type_id] * len(spectrograms))
                        else:
                            # 使用窗口解释结论
                            all_labels.extend(window_label_names)
                            all_label_ids.extend(window_label_ids)

                        all_well_names.extend([well_name] * len(spectrograms))

                        if self.test_mode and len(all_spectrograms) >= 1000:  # 测试模式限制数据量，但确保有足够的流体类型
                            break
                    
            except Exception as e:
                print(f"   ❌ 处理 {well_name} 失败: {e}")
                continue
        
        print(f"✅ 成功生成 {len(all_spectrograms)} 个时频图谱样本")
        return all_spectrograms, all_labels, all_label_ids, all_well_names
    
    def _load_single_well_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单个井的数据"""
        try:
            # 尝试不同的编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            data = None
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding, sep='\t')
                    # 规范化“深度”列为数值（去除可能的'2235--2338(平均)'等注释）
                    if '深度' in data.columns:
                        try:
                            data['深度'] = (
                                data['深度'].astype(str)
                                .str.replace(r'[^0-9\.\-]+', '', regex=True)
                            )
                        except Exception:
                            pass
                    break
                except UnicodeDecodeError:
                    continue
            
            if data is None:
                print(f"   ❌ 无法读取文件: {file_path}")
                return None
            
            # 标准化曲线别名并处理苏257的RT90优先
            try:
                well_name = os.path.basename(file_path).replace('.txt', '')
                # 若缺少RT，尝试用常见替代列补齐
                if 'RT' not in data.columns:
                    for _cand in ['RT90', 'RT90%1', 'RT60', 'RT30', 'RT20', 'RT10', 'RT06%1', 'RT06', 'RLLD', 'RLLS']:
                        if _cand in data.columns:
                            data['RT'] = data[_cand]
                            break
                # 特例：苏257优先采用RT90
                if well_name in ['苏257', 'su257', 'Su257'] and 'RT90' in data.columns:
                    data['RT'] = data['RT90']
            except Exception:
                pass
            
            # 检查必要的列
            available_curves = [col for col in data.columns if col in self.curve_names]
            
            if len(available_curves) < 3:  # 至少需要3个测井曲线
                print(f"   ⚠️  {file_path} 缺少必要的测井曲线")
                return None
            
            # 保留可用的测井曲线与解释结论列（若存在），并尽量保留深度列用于后续特征
            label_col = '解释结论'
            keep_cols = available_curves + ([label_col] if label_col in data.columns else [])
            if '深度' in data.columns and '深度' not in keep_cols:
                keep_cols = ['深度'] + keep_cols
            data = data[keep_cols].copy()
            
            # 处理缺失值
            data = data.ffill().bfill()
            
            # 移除无效行
            data = data.dropna()
            
            if len(data) < self._global_time_window:  # 至少需要全局时间窗口长度的数据
                print(f"   ⚠️  {file_path} 数据量不足 ({len(data)} < {self._global_time_window})")
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
            
            # 深度列清洗（移除范围/注释等非数字字符）
            try:
                if '深度' in valid_data.columns:
                    valid_data['深度'] = (
                        valid_data['深度'].astype(str)
                        .str.extract(r'(-?\d+(?:\.\d+)?)', expand=False)
                    )
                    valid_data['深度'] = pd.to_numeric(valid_data['深度'], errors='coerce')
            except Exception:
                pass

            # 深度对齐：先将各曲线插值到统一深度网格（先对齐，再变换）
            try:
                if '深度' in valid_data.columns:
                    valid_data = self._align_depth_uniform(valid_data)
            except Exception as _align_e:
                print(f"   ⚠️  深度对齐失败，使用原始深度: {_align_e}")

            # 数据清洗和归一化
            cleaned_data = self._normalize_data(valid_data)
            
            # 仅保留“长4+5”层位，并计算相对深度（0~1）
            try:
                cleaned_data = self._restrict_to_layer_and_rel_depth(cleaned_data, layer_keywords=['长4+5', '长4+5段', '长4+5层', 'Chang4+5', 'CHANG4+5'])
            except Exception as _layer_e:
                print(f"   ⚠️  层位筛选/相对深度计算失败，保持全井: {_layer_e}")

            return cleaned_data
        except Exception as e:
            print(f"   ⚠️  清洗 {well_name} 数据失败: {e}")
            return data
    
    def _filter_valid_data(self, data: pd.DataFrame, well_name: str) -> pd.DataFrame:
        """过滤出有效数据"""
        try:
            # 大幅放宽各测井曲线的合理值范围
            valid_ranges = {
                'GR': (-100, 500),    # 自然伽马 (API)
                'SP': (-200, 200),    # 自然电位 (mV)
                'AC': (0, 1000),      # 声波时差 (μs/ft)
                'DEN': (0.5, 5.0),    # 密度 (g/cm³)
                'CNL': (-20, 80),     # 中子孔隙度 (%)
                'RT': (0.01, 1000)    # 电阻率 (Ω·m)
            }
            
            # 过滤有效数据
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
                        print(f"     ⚠️  跳过非数值列 {curve_name} 的过滤: {e}")
                        continue
            
            # 应用过滤条件
            filtered_data = data[valid_mask].copy()
            
            print(f"     {well_name} 数据过滤: {len(data)} -> {len(filtered_data)} 行 ({len(filtered_data)/len(data)*100:.1f}%)")
            
            # 如果过滤后数据太少，返回原始数据
            if len(filtered_data) < len(data) * 0.05:
                print(f"     ⚠️  {well_name} 过滤后数据极少，返回原始数据")
                return data
            
            return filtered_data
            
        except Exception as e:
            print(f"     ⚠️  {well_name} 数据过滤失败: {e}")
            return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """归一化数据（鲁棒按井归一化 + RT对数变换 + 分位裁剪）。"""
        try:
            cleaned_data = data.copy()

            for curve in self.curve_names:
                if curve in cleaned_data.columns:
                    # 1) 替换无穷并用中位数填充缺失
                    s = cleaned_data[curve].replace([np.inf, -np.inf], np.nan)
                    if s.isna().any():
                        s = s.fillna(s.median())

                    # 2) 特殊处理：电阻率RT重尾，使用log1p
                    if curve == 'RT':
                        try:
                            s = np.log1p(np.clip(s.astype(float), 1e-3, None))
                        except Exception:
                            # 失败则跳过对数，保持原值
                            pass

                    # 3) 分位裁剪（1%~99%），抑制异常值影响
                    try:
                        q1 = s.quantile(0.01)
                        q99 = s.quantile(0.99)
                        if np.isfinite(q1) and np.isfinite(q99) and q99 > q1:
                            s = s.clip(lower=q1, upper=q99)
                    except Exception:
                        pass

                    # 4) 鲁棒缩放：median/IQR（IQR过小回退到std）
                    try:
                        med = s.median()
                        q25 = s.quantile(0.25)
                        q75 = s.quantile(0.75)
                        iqr = float(q75 - q25)
                        denom = iqr if iqr > 1e-8 else float(s.std())
                        if denom > 1e-8:
                            s = (s - med) / denom
                        else:
                            s = s - med
                    except Exception:
                        # 最后回退到标准化
                        try:
                            mean_val = s.mean()
                            std_val = s.std()
                            s = (s - mean_val) / std_val if std_val > 1e-8 else (s - mean_val)
                        except Exception:
                            pass

                    # 5) 轻微裁剪，避免极端值
                    try:
                        s = s.clip(lower=-8.0, upper=8.0)
                    except Exception:
                        pass

                    cleaned_data[curve] = s

            return cleaned_data
        except Exception as e:
            print(f"     ⚠️  归一化失败: {e}")
            return data

    def _align_depth_uniform(self, data: pd.DataFrame) -> pd.DataFrame:
        """按井统一深度网格对齐：
        - 使用井内主导采样步长（深度差的中位数）构建统一网格
        - 对数值曲线进行线性插值；离散列（解释结论/层位）用最近邻映射
        """
        if '深度' not in data.columns:
            return data

        try:
            df = data.copy()
            depth = df['深度'].astype(float).values
            if len(depth) < 3:
                return df
            depth_sorted_idx = np.argsort(depth)
            depth = depth[depth_sorted_idx]
            df = df.iloc[depth_sorted_idx].reset_index(drop=True)

            # 估计主导步长
            diffs = np.diff(depth)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size == 0:
                return df
            step = float(np.median(diffs))
            if not np.isfinite(step) or step <= 0:
                return df

            # 统一网格（包含端点）
            new_depth = np.arange(depth.min(), depth.max() + step * 0.5, step, dtype=float)

            aligned = {'深度': new_depth}
            # 数值曲线插值
            for curve in self.curve_names:
                if curve in df.columns:
                    try:
                        arr = pd.to_numeric(df[curve], errors='coerce').astype(float).values
                        mask = np.isfinite(arr) & np.isfinite(depth)
                        if mask.sum() >= 2:
                            f = interp1d(depth[mask], arr[mask], kind='linear', bounds_error=False, fill_value='extrapolate')
                            aligned[curve] = f(new_depth)
                        else:
                            aligned[curve] = np.interp(new_depth, depth, np.nan_to_num(arr, nan=np.nanmedian(arr)))
                    except Exception:
                        aligned[curve] = np.interp(new_depth, depth, pd.to_numeric(df[curve], errors='coerce').fillna(method='ffill').fillna(method='bfill'))
            # 离散列最近邻
            for disc_col in ['解释结论', '结论', '层位']:
                if disc_col in df.columns:
                    try:
                        # 最近邻：为每个new_depth找到原深度中最接近的索引
                        idx_nn = np.searchsorted(depth, new_depth)
                        idx_nn = np.clip(idx_nn, 1, len(depth) - 1)
                        left = idx_nn - 1
                        right = idx_nn
                        choose_left = (np.abs(new_depth - depth[left]) <= np.abs(new_depth - depth[right]))
                        nn_idx = np.where(choose_left, left, right)
                        aligned[disc_col] = df[disc_col].iloc[nn_idx].astype(str).values
                    except Exception:
                        aligned[disc_col] = None

            aligned_df = pd.DataFrame(aligned)
            return aligned_df
        except Exception as e:
            print(f"   ⚠️  深度对齐异常: {e}")
            return data

    def _restrict_to_layer_and_rel_depth(self, data: pd.DataFrame, layer_keywords: List[str]) -> pd.DataFrame:
        """仅保留指定层位（如“长4+5”）范围内的数据，并计算相对深度列 REL_DEPTH∈[0,1]。
        若缺少层位或深度信息，则原样返回。
        """
        df = data.copy()
        if '层位' not in df.columns or '深度' not in df.columns:
            return df

        try:
            lv = df['层位'].astype(str).fillna('')
            mask = pd.Series(False, index=df.index)
            for kw in layer_keywords:
                mask = mask | lv.str.contains(kw, case=False, regex=False)
            sub = df[mask].copy()
            if len(sub) < 5:
                return df  # 层位样本过少，回退全井

            dmin = float(pd.to_numeric(sub['深度'], errors='coerce').min())
            dmax = float(pd.to_numeric(sub['深度'], errors='coerce').max())
            if not (np.isfinite(dmin) and np.isfinite(dmax)) or dmax <= dmin:
                return df

            # 截取层位范围
            df = df[(df['深度'] >= dmin) & (df['深度'] <= dmax)].copy()
            # 相对深度
            df['REL_DEPTH'] = (pd.to_numeric(df['深度'], errors='coerce') - dmin) / (dmax - dmin)
            df['REL_DEPTH'] = df['REL_DEPTH'].clip(lower=0.0, upper=1.0)
            return df
        except Exception as e:
            print(f"   ⚠️  相对深度计算失败: {e}")
            return data
    
    def _generate_well_spectrograms(self, data: pd.DataFrame, well_name: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """为单个井生成时频图谱及对应窗口标签（基于解释结论）"""
        spectrograms: List[np.ndarray] = []
        window_label_ids: List[int] = []
        window_label_names: List[str] = []
        
        # 仅保留数值曲线列（並嚴格按 self.curve_names 的順序對齊）
        numeric_cols = [c for c in self.curve_names if c in data.columns]
        if len(numeric_cols) == 0:
            return [], [], []
        
        # 获取数值数据
        numeric_data = data[numeric_cols].values
        # 構建曲線名稱到列索引的映射，確保後續按名稱精確對齊
        curve_to_col_idx = {name: idx for idx, name in enumerate(numeric_cols)}
        
        # 获取标签列（若存在）
        label_series = None
        for cand in ['解释结论', '结论']:
            if cand in data.columns:
                label_series = data[cand].astype(str).fillna('').values
                break
        
        # 滑动窗口生成时频图谱
        skipped_unknown = 0
        fallback_mapped = 0
        # 使用全局窗口进行滑动，用于标签与对齐；每曲线内部再取各自窗口
        for start_idx in range(0, len(numeric_data) - self._global_time_window + 1, self.time_step):
            end_idx = start_idx + self._global_time_window
            
            # 为每个曲线生成其专属窗口的时频图谱，并堆叠为 (C,H,W)
            try:
                spectrogram_channels: List[np.ndarray] = []
                # 以全局窗口中心对齐各曲线窗口
                global_center = (start_idx + end_idx) // 2
                for j, curve_name in enumerate(self.curve_names):
                    # 计算该曲线窗口长度（若未配置则退回全局窗口）
                    tw = int(self._per_curve_windows.get(curve_name, self.time_window))
                    tw = max(1, min(tw, self._global_time_window))
                    half = tw // 2
                    # 初始对齐到全局中心
                    s = global_center - half
                    e = s + tw
                    # 限制在全局窗口内
                    if s < start_idx:
                        s = start_idx
                        e = s + tw
                    if e > end_idx:
                        e = end_idx
                        s = e - tw
                    # 边界保护
                    if s < 0:
                        s = 0
                        e = min(tw, len(numeric_data))
                    if e > len(numeric_data):
                        e = len(numeric_data)
                        s = max(0, e - tw)
                    # 取該曲線數據段（按曲線名對齊列索引）
                    if curve_name in curve_to_col_idx:
                        curve_idx = curve_to_col_idx[curve_name]
                        curve_segment = numeric_data[s:e, curve_idx]
                        # 生成該曲線的時頻通道
                        spec = self._generate_wavelet_packet_spectrogram(curve_segment, curve_name)
                        if spec is None:
                            spec = np.zeros(self.feature_size)
                    else:
                        # 缺失該曲線列時使用零填充，保持通道對齊
                        spec = np.zeros(self.feature_size)
                    spectrogram_channels.append(spec)
                # 组合为 (C,H,W)
                spectrogram = np.stack(spectrogram_channels, axis=0)
            except Exception as _e:
                # 出错则跳过该窗
                continue
            
            # 解析窗口标签
            if not self.use_fluid_mapping and label_series is not None:
                if self._label_mode == 'center':
                    center_idx = (start_idx + end_idx) // 2
                    raw_text = str(label_series[center_idx]).strip() if 0 <= center_idx < len(label_series) else ''
                    label_id = self._map_text_label_to_id(raw_text)
                    if label_id is None:
                        label_id = 5  # 无
                    label_name = FluidTypes.get_fluid_name(label_id)
                else:
                    window_labels_text = [str(x).strip() for x in label_series[start_idx:end_idx]]
                    if len(window_labels_text) > 0:
                        from collections import Counter
                        counts = Counter(window_labels_text)
                        label_id = None
                        for cand_text, _cnt in counts.most_common(5):
                            label_id = self._map_text_label_to_id(cand_text)
                            if label_id is not None:
                                break
                        # 多數票閾值：若最高票比例低於閾值，則丟棄窗口（訓練集）或標為“無”（驗證/測試）
                        try:
                            if self._majority_threshold > 0 and len(window_labels_text) > 0:
                                total_votes = sum(counts.values())
                                top_votes = max(counts.values()) if counts else 0
                                ratio = (top_votes / max(total_votes, 1)) if total_votes > 0 else 0.0
                                if ratio < self._majority_threshold:
                                    if self._apply_threshold_discard and not self.test_mode:
                                        # 丟棄當前窗口
                                        continue
                                    else:
                                        label_id = 5
                        except Exception:
                            pass
                        # 可选：仅在配置允许时启用稀有类回退（默认关闭）
                        if bool(self.wavelet_config.get('enable_rare_fallback', False)):
                            rare_set = (4,)
                            if (label_id is None or label_id not in rare_set) and well_name in self.well_to_fluid_mapping:
                                mapped = self.well_to_fluid_mapping[well_name]
                                if mapped in rare_set:
                                    label_id = mapped
                                    fallback_mapped += 1
                        if label_id is None:
                            # 跳过非主要流体类型的样本
                            skipped_unknown += 1
                            continue
                        label_name = FluidTypes.get_fluid_name(label_id)
                    else:
                        # 跳过无法识别的样本
                        skipped_unknown += 1
                        continue
            else:
                # 跳过无标签的样本
                skipped_unknown += 1
                continue
            
            spectrograms.append(spectrogram)
            window_label_ids.append(label_id)
            window_label_names.append(label_name)
        
        if skipped_unknown > 0 or fallback_mapped > 0:
            print(f"     {well_name} 生成 {len(spectrograms)} 个时频图谱（跳过未识别 {skipped_unknown}，回退井映射 {fallback_mapped}）")
        # 显示样本生成统计
        if len(window_label_ids) > 0:
            try:
                import numpy as _np
                label_counts = _np.bincount(window_label_ids, minlength=5)
                print(f"     样本分布: 油层={label_counts[0]}, 水层={label_counts[1]}, 干层={label_counts[2]}, 差油层={label_counts[3]}, 油水层={label_counts[4]}")
            except Exception:
                pass
        else:
            print(f"     {well_name} 生成 {len(spectrograms)} 个时频图谱")
        return spectrograms, window_label_ids, window_label_names

    def _map_text_label_to_id(self, text_label: str) -> Optional[int]:
        """将解释结论文本映射为5类FluidTypes ID：油层(0)、水层(1)、干层(2)、差油层(3)、油水层(4)
        非这5类的标签映射为None，表示不属于主要流体类型"""
        # 处理NaN和非字符串类型
        if text_label is None or (isinstance(text_label, float) and np.isnan(text_label)):
            return None
        
        t = str(text_label).strip().lower()
        t = t.replace(' ', '').replace('\u3000', '')

        # 标准5类流体类型关键词定义
        # 重要：油水层必须在油层之前匹配！
        oil_water_keys = ['油水层', '油水同层', '油水共存', '油水同在', '油水互层', 
                         '含油水层', '含油含水', '含水含油', '弱含油水', '油水过渡层']
        poor_oil_keys = ['差油层', '低产油层', '较差油层', '边底水油层', '弱油层', '薄油层']
        oil_keys = ['油层', '油', '良好油层', '高产油层', '可采油层', '有效油层', '油气层', '含油层', 
                    '致密油层', '致密油', '致密', '致密储层', '低渗油层', '特低渗油层']
        water_keys = ['水层', '水', '底水层', '边水层', '高含水层', '纯水层']
        dry_keys = ['干层', '干', '干层段', '非油层', '非含油层', '非产层']
        
        def contains_any(keys: List[str]) -> bool:
            for k in keys:
                if k in t:
                    return True
            return False
        
        # 按优先级进行匹配：油水层 > 差油层 > 油层 > 水层 > 干层
        if contains_any(oil_water_keys):
            return 4  # 油水层 - 最高优先级
        if contains_any(poor_oil_keys):
            return 3  # 差油层
        if contains_any(oil_keys):
            return 0  # 油层
        if contains_any(water_keys):
            return 1  # 水层
        if contains_any(dry_keys):
            return 2  # 干层
        
        # 所有不属于上述5类的标签（包括空值、未知等）返回None，将在数据预处理时被过滤
        return None  # 非主要流体类型，不参与训练
    
    def _generate_spectrogram(self, window_data: np.ndarray) -> Optional[np.ndarray]:
        """生成时频图谱 - 使用小波包分解"""
        try:
            # 确保数据是数值类型
            window_data = window_data.astype(np.float64)

            # 检查数据有效性
            if np.isnan(window_data).any() or np.isinf(window_data).any():
                return None

            # 为每个测井曲线生成时频图谱
            spectrogram_channels = []

            for i, curve_name in enumerate(self.curve_names):
                if i < window_data.shape[1]:
                    curve_data = window_data[:, i]

                    # 选择特征提取方法
                    if self.use_enhanced_extraction:
                        # 使用增强特征提取器
                        spectrogram = self.enhanced_extractor.extract_multi_scale_features(curve_data, curve_name)
                    else:
                        # 使用原始小波包分解
                        spectrogram = self._generate_wavelet_packet_spectrogram(curve_data, curve_name)

                    if spectrogram is not None:
                        spectrogram_channels.append(spectrogram)
                    else:
                        # 如果生成失败，使用零填充
                        spectrogram_channels.append(np.zeros(self.feature_size))
                else:
                    # 如果曲线数量不足，使用零填充
                    spectrogram_channels.append(np.zeros(self.feature_size))

            # 组合所有通道
            if len(spectrogram_channels) == len(self.curve_names):
                final_spectrogram = np.stack(spectrogram_channels, axis=0)
                return final_spectrogram
            else:
                return None

        except Exception as e:
            print(f"     ⚠️  生成时频图谱失败: {e}")
            return None

    def _generate_wavelet_packet_spectrogram(self, curve_data: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
        """生成小波包分解时频图谱"""
        try:
            # 获取曲线特定配置
            wavelet_type, level, mode = self._get_wavelet_config(curve_name)

            # 若配置为 Morlet/CMOR，使用 CWT 实现高频AWPD路径
            if isinstance(wavelet_type, str) and (wavelet_type.lower() in ["morl", "morlet"] or wavelet_type.lower().startswith("cmor")):
                return self._generate_cwt_spectrogram(curve_data, curve_name)

            # 数据预处理
            curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果数据全为零，返回零填充
            if np.all(curve_data == 0):
                return np.zeros(self.feature_size)

            # 使用小波包分解
            wp = pywt.WaveletPacket(data=curve_data, wavelet=wavelet_type, mode=mode)

            # 检查最大分解层数
            max_level = wp.maxlevel
            actual_level = min(level, max_level)
            
            # 获取指定层数的系数
            coeffs = []
            try:
                nodes = wp.get_level(actual_level, 'natural')
                for node in nodes:
                    coeff = node.data
                    if len(coeff) > 0:
                        coeffs.append(coeff)
            except Exception as level_error:
                print(f"     ⚠️  获取小波包分解层级系数失败: {level_error}")
                # 尝试使用更低的层级
                for test_level in range(actual_level - 1, 0, -1):
                    try:
                        nodes = wp.get_level(test_level, 'natural')
                        for node in nodes:
                            coeff = node.data
                            if len(coeff) > 0:
                                coeffs.append(coeff)
                        break
                    except:
                        continue

            if not coeffs:
                # 如果没有系数，使用连续小波变换作为备选
                return self._generate_cwt_spectrogram(curve_data, curve_name)

            # 将系数转换为矩阵，确保维度兼容
            try:
                # 确保所有系数长度一致
                max_length = max(len(coeff) for coeff in coeffs)
                padded_coeffs = []
                for coeff in coeffs:
                    if len(coeff) < max_length:
                        # 零填充
                        padded = np.pad(coeff, (0, max_length - len(coeff)), mode='constant')
                        padded_coeffs.append(padded)
                    else:
                        padded_coeffs.append(coeff[:max_length])
                
                coeffs_matrix = np.vstack(padded_coeffs)
                
                # 生成时频图谱（先到曲线级目标尺寸，再统一到全局 feature_size）
                target_size = self._per_curve_output_size.get(curve_name, self.feature_size)
                spectrogram_small = self._wavelet_coefficients_to_spectrogram_with_size(coeffs_matrix, target_size)
                spectrogram_final = self._resize_to_size(spectrogram_small, self.feature_size)
                return spectrogram_final
                
            except Exception as matrix_error:
                print(f"     ⚠️  小波系数矩阵处理失败: {matrix_error}")
                return self._generate_cwt_spectrogram(curve_data, curve_name)

        except Exception as e:
            print(f"     ⚠️  小波包分解失败，使用CWT备选: {e}")
            return self._generate_cwt_spectrogram(curve_data, curve_name)

    def _get_wavelet_config(self, curve_name: str) -> Tuple[str, int, str]:
        """获取曲线特定小波配置"""
        # 优先使用传入的小波包分解配置
        if self.wavelet_config and 'curve_configs' in self.wavelet_config:
            curve_configs = self.wavelet_config['curve_configs']
            if curve_name in curve_configs:
                config = curve_configs[curve_name]
                return config.get('wavelet', 'db8'), config.get('level', 2), config.get('mode', 'symmetric')
        
        # 使用默认配置
        curve_specific_wavelets = {
            'AC': {'wavelet': 'db8', 'level': 2, 'mode': 'symmetric'},  # 使用db8替代morlet
            'RT': {'wavelet': 'db8', 'level': 2, 'mode': 'symmetric'},  # 使用db8替代morlet
            'GR': {'wavelet': 'sym8', 'level': 2, 'mode': 'symmetric'}, # 降低分解层数
            'SP': {'wavelet': 'sym8', 'level': 2, 'mode': 'symmetric'}, # 降低分解层数
            'CNL': {'wavelet': 'db8', 'level': 2, 'mode': 'symmetric'}, # 使用db8替代morlet
            'DEN': {'wavelet': 'sym8', 'level': 2, 'mode': 'symmetric'} # 降低分解层数
        }

        if curve_name in curve_specific_wavelets:
            config = curve_specific_wavelets[curve_name]
            return config.get('wavelet', 'db8'), config.get('level', 2), config.get('mode', 'symmetric')
        else:
            return 'db8', 2, 'symmetric'

    def _wavelet_coefficients_to_spectrogram_with_size(self, coeffs_matrix: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """将小波系数转换为指定尺寸的时频图谱"""
        try:
            # 计算能量谱
            use_energy_spectrum = True  # 默认使用能量谱
            if use_energy_spectrum:
                # 使用系数的平方作为能量
                energy_matrix = np.abs(coeffs_matrix) ** 2
            else:
                # 使用系数的绝对值
                energy_matrix = np.abs(coeffs_matrix)

            # 取消每样本归一化，保留原始动态范围；采用数据集级标准化（在 __getitem__ 中应用）

            # 调整到目标尺寸
            spectrogram = self._resize_to_size(energy_matrix, target_size)

            return spectrogram

        except Exception as e:
            print(f"     ⚠️  系数转换失败: {e}")
            return np.zeros(target_size)

    def _generate_cwt_spectrogram(self, curve_data: np.ndarray, curve_name: str) -> Optional[np.ndarray]:
        """生成连续小波变换时频图谱"""
        try:
            # 数据预处理
            curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 如果数据全为零，返回零填充
            if np.all(curve_data == 0):
                return np.zeros(self.feature_size)
            
            # 生成尺度序列（放宽带宽以提升判别信息）
            low, high = self.scale_range
            low = max(low, 5)
            high = min(high, 36)
            if high <= low:
                high = low + 8
            scales = np.arange(low, high + 1)
            
            # 使用Morlet小波进行连续小波变换
            try:
                # 使用pywt进行连续小波变换 - 修复弃用警告
                # 使用明确的参数格式: cmorB-C (B=带宽频率, C=中心频率)
                coefficients, frequencies = pywt.cwt(curve_data, scales, 'cmor1.5-1.0', sampling_period=1.0)
                
                # 取系数的幅度
                spectrogram = np.abs(coefficients)
                
            except Exception as e:
                print(f"       ⚠️  CWT失败，使用STFT: {e}")
                # 备选方案：使用短时傅里叶变换
                f, t, Zxx = signal.stft(curve_data, nperseg=min(16, len(curve_data)//2))
                spectrogram = np.abs(Zxx)
                
                # 调整尺度维度
                if spectrogram.shape[0] < len(scales):
                    # 插值到目标尺度数
                    from scipy.interpolate import interp2d
                    f_interp = interp2d(t, f, spectrogram, kind='linear')
                    new_f = np.linspace(f.min(), f.max(), len(scales))
                    spectrogram = f_interp(t, new_f)
                elif spectrogram.shape[0] > len(scales):
                    # 截断到目标尺度数
                    spectrogram = spectrogram[:len(scales)]
            
            # 取消每样本归一化，保留原始动态范围；采用数据集级标准化（在 __getitem__ 中应用）
            
            # 先到曲线级目标尺寸，再统一到全局 feature_size
            target_size = self._per_curve_output_size.get(curve_name, self.feature_size)
            spectrogram_small = self._resize_to_size(spectrogram, target_size)
            spectrogram = self._resize_to_size(spectrogram_small, self.feature_size)
            
            return spectrogram
            
        except Exception as e:
            print(f"       ⚠️  生成CWT时频图谱失败: {e}")
            return None
    
    def _resize_to_size(self, spectrogram: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """调整时频图谱尺寸到指定大小"""
        try:
            from scipy.ndimage import zoom
            
            # 计算缩放因子
            scale_factor = size[0] / spectrogram.shape[0]
            time_factor = size[1] / spectrogram.shape[1]
            
            # 使用zoom进行缩放
            resized = zoom(spectrogram, (scale_factor, time_factor), order=1)
            
            # 确保输出尺寸正确
            if resized.shape != size:
                # 如果尺寸不匹配，进行裁剪或填充
                final_spectrogram = np.zeros(size)
                h_min = min(resized.shape[0], size[0])
                w_min = min(resized.shape[1], size[1])
                final_spectrogram[:h_min, :w_min] = resized[:h_min, :w_min]
                return final_spectrogram
            
            return resized
            
        except Exception as e:
            print(f"       ⚠️  调整时频图谱尺寸失败: {e}")
            # 返回零填充的默认尺寸
            return np.zeros(size)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """获取单个样本"""
        spectrogram = self.data[idx]
        label = int(self.labels_encoded[idx])
        well_name = self.well_names[idx]
        
        # 转换为张量
        spectrogram_tensor = torch.FloatTensor(spectrogram)
        
        # 应用数据集级标准化（按通道）
        if self.dataset_channel_mean is not None and self.dataset_channel_std is not None:
            try:
                cm = torch.from_numpy(self.dataset_channel_mean).view(-1, 1, 1).type_as(spectrogram_tensor)
                cs = torch.from_numpy(self.dataset_channel_std).view(-1, 1, 1).type_as(spectrogram_tensor)
                spectrogram_tensor = (spectrogram_tensor - cm) / (cs + 1e-6)
            except Exception as e:
                print(f"⚠️  应用数据集级标准化失败: {e}")
        
        # 最终验证张量形状
        expected_shape = (len(self.curve_names), *self.feature_size)
        if spectrogram_tensor.shape != expected_shape:
            print(f"⚠️  张量形状不匹配: 期望{expected_shape}, 实际{spectrogram_tensor.shape}")
            # 强制调整形状
            if spectrogram_tensor.numel() >= np.prod(expected_shape):
                spectrogram_tensor = spectrogram_tensor.view(expected_shape)
            else:
                # 如果元素不足，用零填充
                new_tensor = torch.zeros(expected_shape, dtype=spectrogram_tensor.dtype)
                flat_old = spectrogram_tensor.flatten()
                flat_new = new_tensor.flatten()
                flat_new[:min(len(flat_old), len(flat_new))] = flat_old[:min(len(flat_old), len(flat_new))]
                spectrogram_tensor = flat_new.view(expected_shape)
        
        return spectrogram_tensor, torch.LongTensor([label]).squeeze(), well_name


def create_spectrogram_data_loaders(
    welldata_dir: str = "welldata",
    curve_names: List[str] = None,
    scale_range: Tuple[int, int] = (5, 36),
    time_window: int = 32,
    time_step: int = 1,
    feature_size: Tuple[int, int] = (64, 64),
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.3,
    test_ratio: float = 0.0,  # 新增测试集比例
    enable_cleaning: bool = True,
    num_workers: int = 0,
    test_mode: bool = False,
    use_fluid_mapping: bool = True,
    custom_split: Dict[str, List[str]] = None,  # 新增自定义分割
    use_stratified_split: bool = False,  # 新增分层抽样选项
    random_state: int = 42,  # 新增随机种子
    wavelet_config: Dict = None  # 新增小波包分解配置
) -> Dict[str, DataLoader]:
    """
    创建基于时频图谱的数据加载器
    
    Args:
        custom_split: 自定义分割字典，格式为 {'train': [井名列表], 'val': [井名列表], 'test': [井名列表]}
    
    Returns:
        包含训练、验证、测试数据加载器的字典
    """
    
    # 创建数据集
    dataset = SpectrogramWellLogDataset(
        welldata_dir=welldata_dir,
        curve_names=curve_names,
        scale_range=scale_range,
        time_window=time_window,
        time_step=time_step,
        feature_size=feature_size,
        enable_cleaning=enable_cleaning,
        test_mode=test_mode,
        use_fluid_mapping=use_fluid_mapping,
        wavelet_config=wavelet_config  # 传递小波包分解配置
    )
    
    if len(dataset) == 0:
        raise ValueError("没有可用的数据")
    
    # 如果启用分层抽样，使用分层抽样分割
    if use_stratified_split:
        print(f"🎯 使用分层抽样分割策略 (训练:{train_ratio:.1f}, 验证:{val_ratio:.1f}, 测试:{test_ratio:.1f})")
        
        # 收集所有标签用于分层抽样
        all_labels = []
        for idx in range(len(dataset)):
            _, label, _ = dataset[idx]
            all_labels.append(label.item())
        
        # 使用sklearn进行分层抽样
        try:
            from sklearn.model_selection import train_test_split
            import numpy as _np
            
            # 设置随机种子
            _np.random.seed(random_state)
            torch.manual_seed(random_state)
            
            indices = _np.arange(len(dataset))
            
            # 预检查：类别最小样本数
            try:
                from collections import Counter as _Counter
                _counts = _Counter(all_labels)
                if len(_counts) < 2 or min(_counts.values()) < 2:
                    print("   ⚠️  分层抽样不可用：存在样本数<2的类别或类别数<2。回退到随机分割")
                    raise ValueError("insufficient class counts for stratification")
            except Exception:
                # 若计数失败，继续尝试，在失败时回退
                pass

            try:
                if test_ratio > 0:
                    # 三分割：训练 vs (验证+测试)
                    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
                        indices, all_labels, 
                        test_size=(val_ratio + test_ratio), 
                        random_state=random_state,
                        stratify=all_labels
                    )
                    
                    # 验证 vs 测试
                    val_size = val_ratio / (val_ratio + test_ratio)
                    val_indices, test_indices, val_labels, test_labels = train_test_split(
                        temp_indices, temp_labels,
                        test_size=(1 - val_size),
                        random_state=random_state,
                        stratify=temp_labels
                    )
                else:
                    # 二分割：训练 vs 验证
                    train_indices, val_indices, train_labels, val_labels = train_test_split(
                        indices, all_labels,
                        test_size=val_ratio,
                        random_state=random_state,
                        stratify=all_labels
                    )
                    test_indices = []
            except ValueError as _e:
                print(f"   ⚠️  分层抽样失败: {_e}. 回退到随机分割")
                # 随机分割回退
                if test_ratio > 0:
                    train_size = int(len(dataset) * train_ratio)
                    val_size = int(len(dataset) * val_ratio)
                    test_size = len(dataset) - train_size - val_size
                    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                        dataset, [train_size, val_size, test_size]
                    )
                else:
                    train_size = int(len(dataset) * train_ratio)
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(
                        dataset, [train_size, val_size]
                    )
                    test_dataset = None
            else:
                # 仅在分层成功时创建子集与统计
                # 验证类别一致性
                train_classes = set(train_labels)
                val_classes = set(val_labels)
                test_classes = set(test_labels) if (test_ratio > 0) else set()
                
                print(f"📊 分层抽样结果:")
                print(f"   训练集: {len(train_indices)} 样本, 类别: {sorted(train_classes)}")
                print(f"   验证集: {len(val_indices)} 样本, 类别: {sorted(val_classes)}")
                if test_ratio > 0:
                    print(f"   测试集: {len(test_indices)} 样本, 类别: {sorted(test_classes)}")
                
                # 检查类别一致性
                all_classes = set(all_labels)
                if train_classes == all_classes and val_classes == all_classes and ((test_ratio == 0) or (test_classes == all_classes)):
                    print("   ✅ 所有集合都包含完整的类别集合")
                else:
                    print("   ⚠️  部分集合缺少某些类别，但这是分层抽样的正常结果")
                
                # 创建子集
                train_dataset = torch.utils.data.Subset(dataset, train_indices)
                val_dataset = torch.utils.data.Subset(dataset, val_indices)
                test_dataset = torch.utils.data.Subset(dataset, test_indices) if (test_ratio > 0) else None
            
        except ImportError:
            print("   ⚠️  sklearn未安装，回退到随机分割")
            use_stratified_split = False
    
    # 如果提供了自定义分割，使用自定义分割
    elif custom_split is not None:
        print(f"🎯 使用自定义井分割策略:")
        print(f"   训练井: {custom_split.get('train', [])}")
        print(f"   验证井: {custom_split.get('val', [])}")
        print(f"   测试井: {custom_split.get('test', [])}")
        
        # 根据井名分割数据集
        train_indices = []
        val_indices = []
        test_indices = []
        
        for idx in range(len(dataset)):
            _, _, well_name = dataset[idx]
            if well_name in custom_split.get('train', []):
                train_indices.append(idx)
            elif well_name in custom_split.get('val', []):
                val_indices.append(idx)
            elif well_name in custom_split.get('test', []):
                test_indices.append(idx)
        
        # 打印分割统计信息
        print(f"📊 井分割统计:")
        from collections import Counter
        train_counts = Counter([dataset.well_names[i] for i in train_indices])
        val_counts = Counter([dataset.well_names[i] for i in val_indices])
        test_counts = Counter([dataset.well_names[i] for i in test_indices])
        
        all_wells = set(custom_split.get('train', []) + custom_split.get('val', []) + custom_split.get('test', []))
        for well_name in all_wells:
            count = train_counts.get(well_name, 0) + val_counts.get(well_name, 0) + test_counts.get(well_name, 0)
            split_type = "训练" if well_name in custom_split.get('train', []) else \
                        "验证" if well_name in custom_split.get('val', []) else \
                        "测试"
            print(f"   {well_name}: {count} 样本 -> {split_type}集")
        
        print(f"   训练集总样本: {len(train_indices)}")
        print(f"   验证集总样本: {len(val_indices)}")
        print(f"   测试集总样本: {len(test_indices)}")
        
        # 创建子集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else None
        
    else:
        # 使用比例分割
        total_size = len(dataset)
        
        if test_ratio > 0:
            # 三分割：训练、验证、测试
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * val_ratio)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
        else:
            # 二分割：训练、验证
            train_size = int(total_size * train_ratio)
            val_size = total_size - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            test_dataset = None
    
    print(f"📊 数据集分割:")
    print(f"   训练集: {len(train_dataset)}")
    print(f"   验证集: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"   测试集: {len(test_dataset)}")
    
    # 在创建DataLoader前：计算训练子集的按通道mean/std并设置到基dataset（数据集级标准化）
    try:
        def _get_subset_indices(subset_obj):
            try:
                # torch.utils.data.Subset
                return list(subset_obj.indices)
            except Exception:
                # 不是Subset，可能是原始dataset
                return list(range(len(subset_obj)))

        def _compute_channel_stats(base_dataset: SpectrogramWellLogDataset, subset_indices: list):
            if len(subset_indices) == 0:
                return None, None
            # 使用流式统计避免占用过多内存
            C = len(base_dataset.curve_names)
            sum_c = __import__('numpy').zeros((C,), dtype=__import__('numpy').float64)
            sumsq_c = __import__('numpy').zeros((C,), dtype=__import__('numpy').float64)
            count = 0
            for idx in subset_indices:
                spec_np = base_dataset.data[idx]
                if spec_np is None:
                    continue
                # spec_np: (C, H, W)
                try:
                    ch_flat = spec_np.reshape(C, -1)
                except Exception:
                    # 强制到期望形状
                    H, W = base_dataset.feature_size
                    if spec_np.size >= C * H * W:
                        ch_flat = spec_np.reshape(C, H, W).reshape(C, -1)
                    else:
                        # 跳过异常样本
                        continue
                sum_c += ch_flat.sum(axis=1)
                sumsq_c += (ch_flat ** 2).sum(axis=1)
                count += ch_flat.shape[1]
            if count == 0:
                return None, None
            np_mod = __import__('numpy')
            mean = sum_c / count
            var = np_mod.maximum(sumsq_c / count - mean ** 2, 1e-8)
            std = np_mod.sqrt(var)
            return mean.astype(np_mod.float32), std.astype(np_mod.float32)

        # 依据已构建的拆分对象提取训练索引
        # 先临时构造默认比例拆分以获取一致的变量
        # 实际拆分对象在下方根据分支已生成：train_dataset/val_dataset/test_dataset
        pass
    except Exception as _:
        pass

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
    
    result = {
        'train': train_loader,
        'val': val_loader,
        'dataset': dataset
    }
    
    # 如果有测试集，添加测试数据加载器
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        result['test'] = test_loader

    # 现在真正计算并设置训练集的标准化统计（放在最后，确保train_dataset已存在）
    try:
        train_indices = _get_subset_indices(train_dataset)
        ch_mean, ch_std = _compute_channel_stats(dataset, train_indices)
        if ch_mean is not None and ch_std is not None:
            dataset.set_dataset_normalization_stats(ch_mean, ch_std)
            np_mod = __import__('numpy')
            print(f"   📐 数据集级标准化: mean={np_mod.round(ch_mean, 4)}, std={np_mod.round(ch_std, 4)}")
        else:
            print("   ⚠️  未能计算数据集级标准化统计，保持未标准化输入")
    except Exception as e:
        print(f"   ⚠️  计算/设置数据集级标准化失败: {e}")
    
    return result


def test_spectrogram_data_loader():
    """测试基于时频图谱的数据加载器"""
    print("🧪 测试基于时频图谱的数据加载器...")
    
    try:
        # 创建数据集
        dataset = SpectrogramWellLogDataset(
            welldata_dir="welldata",
            scale_range=(5, 36),
            time_window=32,
            time_step=1,
            feature_size=(64, 64),
            test_mode=True
        )
        
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample_data, sample_label, sample_well_name = dataset[0]
            print(f"✅ 样本获取成功:")
            print(f"   数据形状: {sample_data.shape}")
            print(f"   标签(整数ID): {sample_label}")
            print(f"   井名: {sample_well_name}")
            print(f"   期望形状: (6, 64, 64)")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_spectrogram_data_loader()
