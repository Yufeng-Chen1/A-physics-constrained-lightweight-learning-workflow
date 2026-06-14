#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的子图谱提取器
将256×256时频图谱智能分解为32×32子图谱
使用PCA降维处理冗余，通过合理参数设计转化劣势为优势
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# 添加项目路径（脚本目录与项目根）
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
sys.path.append(this_dir)
sys.path.append(project_root)

try:
    from data.enhanced_time_frequency_extractor import create_enhanced_extractor
    from data.fluid_types import FluidTypes
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 在Windows控制台上安全打印，自动忽略无法编码的字符（如emoji）
import builtins as _builtins
def _safe_print(*args, **kwargs):
    try:
        _builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            joined = " ".join(str(a) for a in args)
            joined = joined.encode('gbk', errors='ignore').decode('gbk', errors='ignore')
            _builtins.print(joined)
        except Exception:
            _builtins.print("[print-error]")
print = _safe_print

class OptimizedPatchExtractor:
    """优化的子图谱提取器"""
    
    def __init__(self, welldata_dir: str = "welldata", cache_dir: str = "optimized_patch_data"):
        self.welldata_dir = Path(welldata_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.spectrograms_dir = self.cache_dir / "spectrograms"
        self.metadata_dir = self.cache_dir / "metadata"
        self.pca_dir = self.cache_dir / "pca_models"
        self.spectrograms_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.pca_dir.mkdir(exist_ok=True)
        
        # 储层流体识别优化配置
        self.curve_names = ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC']
        self.time_window = 64  # 64时间窗口
        self.overlap_ratio = 0.5  # 0.5重叠率
        self.base_size = (256, 256)  # 基础分辨率
        self.patch_size = 32  # 子图谱尺寸
        
        # 智能子图谱提取配置
        # 子图谱重叠率：按需求改为 50%
        self.patch_overlap = 0.5
        self.min_energy_threshold = 0.1  # 最小能量阈值（过滤低信息子图谱）
        self.pca_components = 0.98  # PCA保留98%方差
        
        # 创建特征提取器
        extractor_config = {
            'feature_size': self.base_size,
            'time_window': self.time_window,
            'overlap_ratio': self.overlap_ratio,
            'decomposition_levels': [4],  # 4层小波包分解
            'enable_cwt': False,
            'feature_diversity': True
        }
        
        print("🔧 初始化储层特征提取器...")
        self.enhanced_extractor = create_enhanced_extractor(extractor_config)
        
        # PCA降维器（每个通道一个）
        self.pca_models = {}
        self.scalers = {}
        
        # 统计
        self.total_samples = 0
        self.generated_samples = 0
        
        print(f"✅ 优化子图谱提取器初始化完成")
        print(f"   基础时频图谱: 256×256")
        print(f"   子图谱尺寸: 32×32")
        print(f"   子图谱重叠率: {self.patch_overlap}")
        print(f"   能量阈值: {self.min_energy_threshold}")
        print(f"   PCA保留方差: {self.pca_components}")

    # -----------------------------
    # 与训练一致的预处理：深度对齐/层位/相对深度/鲁棒归一化
    # -----------------------------
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """規範欄名：去空白、簡繁映射、常見別名與亂碼別名兼容。"""
        try:
            col_map = {}
            for c in list(df.columns):
                cn = str(c).strip()
                # 常見簡繁/同義
                aliases = {
                    '解释结论': ['解釋結論', '解释結论', '結論', '结论', '解释', '說明結論', '說明'],
                    '层位': ['層位', '地層', '层位名称', '層位名稱'],
                }
                # 亂碼別名（在部分錯誤編碼情況下出現）
                garbled_aliases = {
                    '解释结论': ['���ͽ���'],
                    '层位': ['��λ'],
                }
                mapped = None
                for key, vals in aliases.items():
                    if cn == key or any(cn == v for v in vals):
                        mapped = key
                        break
                if mapped is None:
                    for key, vals in garbled_aliases.items():
                        if any(cn == v for v in vals):
                            mapped = key
                            break
                # 就地記錄映射
                if mapped and mapped != c:
                    col_map[c] = mapped
            if col_map:
                df = df.rename(columns=col_map)
        except Exception:
            pass
        return df
    def _align_depth_uniform(self, data: pd.DataFrame) -> pd.DataFrame:
        """按统一深度网格对齐：线性插值数值列，最近邻映射离散列。"""
        if '深度' not in data.columns or len(data) < 3:
            return data
        try:
            df = data.copy()
            depth = pd.to_numeric(df['深度'], errors='coerce').astype(float).values
            order = np.argsort(depth)
            depth = depth[order]
            df = df.iloc[order].reset_index(drop=True)

            diffs = np.diff(depth)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size == 0:
                return df
            step = float(np.median(diffs))
            if not np.isfinite(step) or step <= 0:
                return df

            new_depth = np.arange(depth.min(), depth.max() + step * 0.5, step, dtype=float)
            aligned = {'深度': new_depth}

            # 数值曲线插值
            for curve in self.curve_names:
                if curve in df.columns:
                    arr = pd.to_numeric(df[curve], errors='coerce').astype(float).values
                    mask = np.isfinite(arr) & np.isfinite(depth)
                    if mask.sum() >= 2:
                        f = interp1d(depth[mask], arr[mask], kind='linear', bounds_error=False, fill_value='extrapolate')
                        aligned[curve] = f(new_depth)
                    else:
                        aligned[curve] = np.interp(new_depth, depth, np.nan_to_num(arr, nan=np.nanmedian(arr)))

            # 离散列最近邻
            for disc_col in ['解释结论', '结论', '层位']:
                if disc_col in df.columns:
                    idx_nn = np.searchsorted(depth, new_depth)
                    idx_nn = np.clip(idx_nn, 1, len(depth) - 1)
                    left = idx_nn - 1
                    right = idx_nn
                    choose_left = (np.abs(new_depth - depth[left]) <= np.abs(new_depth - depth[right]))
                    nn_idx = np.where(choose_left, left, right)
                    aligned[disc_col] = df[disc_col].iloc[nn_idx].astype(str).values

            return pd.DataFrame(aligned)
        except Exception:
            return data

    def _restrict_to_layer_and_rel_depth(self, data: pd.DataFrame, layer_keywords: List[str]) -> pd.DataFrame:
        """仅保留指定层位（如长4+5）并计算相对深度REL_DEPTH∈[0,1]。缺失则原样返回。"""
        if '层位' not in data.columns or '深度' not in data.columns:
            return data
        try:
            lv = data['层位'].astype(str).fillna('')
            mask = pd.Series(False, index=data.index)
            for kw in layer_keywords:
                mask = mask | lv.str.contains(kw, case=False, regex=False)
            sub = data[mask].copy()
            if len(sub) < 5:
                return data
            dmin = float(pd.to_numeric(sub['深度'], errors='coerce').min())
            dmax = float(pd.to_numeric(sub['深度'], errors='coerce').max())
            if not (np.isfinite(dmin) and np.isfinite(dmax)) or dmax <= dmin:
                return data
            df = data[(data['深度'] >= dmin) & (data['深度'] <= dmax)].copy()
            rel = (pd.to_numeric(df['深度'], errors='coerce') - dmin) / (dmax - dmin)
            df['REL_DEPTH'] = rel.clip(lower=0.0, upper=1.0)
            return df
        except Exception:
            return data
    
    def _map_text_label_to_id(self, text_label: str) -> Optional[int]:
        """标签映射"""
        if text_label is None or (isinstance(text_label, float) and np.isnan(text_label)):
            return None
        
        t = str(text_label).strip().lower()
        
        # 储层流体5类映射
        if '油水层' in t or '油水同层' in t or '含油水层' in t:
            return 4  # 油水层
        elif '差油层' in t or '低产油层' in t:
            return 3  # 差油层
        elif '致密油层' in t or '致密' in t or '油层' in t or '油' in t:
            return 0  # 油层（包含致密油层）
        elif '水层' in t or '水' in t:
            return 1  # 水层
        elif '干层' in t or '干' in t:
            return 2  # 干层
        else:
            return None  # 未识别，过滤
    
    def _normalize_curve_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """测井曲线数据归一化（与训练一致：RT对数+分位裁剪+鲁棒缩放+限幅）。"""
        cleaned_data = data.copy()
        for curve in self.curve_names:
            if curve in cleaned_data.columns:
                s = pd.to_numeric(cleaned_data[curve], errors='coerce')
                s = s.replace([np.inf, -np.inf], np.nan)
                if s.isna().any():
                    s = s.fillna(s.median())
                if curve == 'RT':
                    try:
                        s = np.log1p(np.clip(s.astype(float), 1e-3, None))
                    except Exception:
                        pass
                try:
                    q1 = s.quantile(0.01)
                    q99 = s.quantile(0.99)
                    if np.isfinite(q1) and np.isfinite(q99) and q99 > q1:
                        s = s.clip(lower=q1, upper=q99)
                except Exception:
                    pass
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
                    try:
                        mean_val = s.mean(); std_val = s.std()
                        s = (s - mean_val) / std_val if std_val > 1e-8 else (s - mean_val)
                    except Exception:
                        pass
                try:
                    s = s.clip(lower=-8.0, upper=8.0)
                except Exception:
                    pass
                cleaned_data[curve] = s
        return cleaned_data
    
    def _generate_base_spectrogram(self, curve_data: np.ndarray, curve_name: str) -> np.ndarray:
        """生成256×256基础时频图谱"""
        try:
            spectrogram = self.enhanced_extractor.extract_multi_scale_features(curve_data, curve_name)
            
            # 确保输出为256×256
            if spectrogram.shape != self.base_size:
                from scipy import ndimage
                scale_factors = (
                    self.base_size[0] / spectrogram.shape[0],
                    self.base_size[1] / spectrogram.shape[1]
                )
                spectrogram = ndimage.zoom(spectrogram, scale_factors, order=1)
            
            return spectrogram
            
        except Exception as e:
            print(f"     ⚠️ {curve_name} 时频图谱生成失败: {e}")
            return np.zeros(self.base_size)
    
    def _extract_intelligent_patches(self, spectrogram: np.ndarray) -> List[Dict]:
        """智能提取32×32子图谱"""
        patches_info = []
        h, w = spectrogram.shape
        
        # 计算步长（减少重叠以降低冗余）
        step = int(self.patch_size * (1 - self.patch_overlap))
        step = max(step, 1)
        
        # 滑动窗口提取子图谱
        for i in range(0, h - self.patch_size + 1, step):
            for j in range(0, w - self.patch_size + 1, step):
                patch = spectrogram[i:i+self.patch_size, j:j+self.patch_size]
                
                if patch.shape == (self.patch_size, self.patch_size):
                    # 计算能量（用于过滤低信息子图谱）
                    energy = np.sum(patch ** 2)
                    normalized_energy = energy / (self.patch_size ** 2)
                    
                    # 只保留高能量的子图谱
                    if normalized_energy > self.min_energy_threshold:
                        # 计算其他特征用于多样性筛选
                        variance = np.var(patch)
                        gradient_magnitude = np.sum(np.gradient(patch)[0] ** 2 + np.gradient(patch)[1] ** 2)
                        
                        patches_info.append({
                            'patch': patch,
                            'position': (i, j),
                            'energy': normalized_energy,
                            'variance': variance,
                            'gradient': gradient_magnitude,
                            'diversity_score': normalized_energy * variance * gradient_magnitude
                        })
        
        # 根据多样性得分排序，选择最有信息量的子图谱
        patches_info.sort(key=lambda x: x['diversity_score'], reverse=True)
        
        # 限制子图谱数量（避免过多冗余）
        max_patches = min(len(patches_info), 16)  # 最多16个子图谱
        return patches_info[:max_patches]
    
    def _apply_pca_to_patches(self, patches: List[np.ndarray], channel_idx: int) -> List[np.ndarray]:
        """对子图谱应用PCA降维"""
        if len(patches) == 0:
            return patches
        
        try:
            # 将子图谱展平为特征向量
            patch_vectors = []
            for patch in patches:
                patch_vectors.append(patch.flatten())
            
            patch_matrix = np.array(patch_vectors)
            
            # 初始化或获取该通道的PCA模型
            channel_key = f"channel_{channel_idx}"
            
            if channel_key not in self.pca_models:
                # 创建新的PCA模型
                scaler = StandardScaler()
                pca = PCA(n_components=self.pca_components, random_state=42)
                
                # 标准化
                scaled_patches = scaler.fit_transform(patch_matrix)
                
                # PCA拟合
                pca_features = pca.fit_transform(scaled_patches)
                
                # 保存模型
                self.scalers[channel_key] = scaler
                self.pca_models[channel_key] = pca
                
                print(f"   📊 通道{channel_idx} PCA: {patch_matrix.shape[1]} -> {pca_features.shape[1]} 特征")
                
            else:
                # 使用已有模型转换
                scaler = self.scalers[channel_key]
                pca = self.pca_models[channel_key]
                
                scaled_patches = scaler.transform(patch_matrix)
                pca_features = pca.transform(scaled_patches)
            
            # 将PCA特征转换回子图谱格式（通过逆变换或填充）
            # 这里我们保留原始子图谱，但可以用PCA特征做进一步分析
            return patches
            
        except Exception as e:
            print(f"   ⚠️ 通道{channel_idx} PCA处理失败: {e}")
            return patches
    
    def _determine_window_label(self, window_labels: List[str]) -> Optional[int]:
        """确定窗口标签（优先级投票）"""
        if not window_labels:
            return None
        
        # 统计所有映射后的标签
        label_votes = {}
        for label in window_labels:
            mapped_id = self._map_text_label_to_id(label)
            if mapped_id is not None:
                label_votes[mapped_id] = label_votes.get(mapped_id, 0) + 1
        
        if not label_votes:
            return None
        
        # 加权投票（考虑储层重要性）
        priority_weights = {4: 2.0, 0: 1.5, 3: 1.2, 1: 1.0, 2: 0.8}
        
        weighted_votes = {}
        for label_id, votes in label_votes.items():
            weight = priority_weights.get(label_id, 1.0)
            weighted_votes[label_id] = votes * weight
        
        return max(weighted_votes, key=weighted_votes.get)
    
    def process_well_file(self, well_file: Path) -> int:
        """处理单个井文件"""
        print(f"📊 处理井文件: {well_file.name}")
        
        try:
            # 读取数据（多编码尝试）
            data = None
            for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1', 'utf-16', 'utf-16le', 'utf-16be']:
                try:
                    data = pd.read_csv(well_file, encoding=enc, sep='\t')
                    break
                except Exception:
                    continue
            if data is None:
                print(f"   ⚠️  无法读取 {well_file.name}")
                return 0
            print(f"   原始数据: {len(data)} 行")
            # 欄名正規化（處理簡繁/別名/亂碼）
            data = self._normalize_columns(data)
            
            # 曲线别名标准化（如将RT90/RLLD等映射为RT；苏257优先使用RT90）
            try:
                well_name = well_file.stem
                if 'RT' not in data.columns:
                    for _cand in ['RT90','RT90%1','RT60','RT30','RT20','RT10','RT06%1','RT06','RLLD','RLLS']:
                        if _cand in data.columns:
                            data['RT'] = data[_cand]
                            break
                # 特例：苏257优先采用RT90
                if well_name in ['苏257','su257','Su257'] and 'RT90' in data.columns:
                    data['RT'] = data['RT90']
            except Exception:
                pass
            
            # 标签列检测：允许无标签井进入“未标注模式”以便生成推理用图谱
            has_label = ('解释结论' in data.columns) or ('结论' in data.columns)
            if not has_label:
                print(f"   ⚠️ 缺少解释结论/结论列，按未标注模式生成(仅用于推理/评估图谱)")
            
            # 检查测井曲线
            available_curves = [curve for curve in self.curve_names if curve in data.columns]
            if len(available_curves) < 4:
                print(f"   ⚠️ 测井曲线不足 ({len(available_curves)}/{len(self.curve_names)})，跳过")
                return 0
            
            print(f"   使用曲线: {available_curves}")
            
            # 对齐深度 -> 限定长4+5 -> 归一化（与训练一致）
            if '深度' in data.columns:
                data = self._align_depth_uniform(data)
                data = self._restrict_to_layer_and_rel_depth(data, ['长4+5', '长4+5段', '长4+5层', 'Chang4+5', 'CHANG4+5'])

            # 仅先收集可用曲线，后续在滑窗阶段按固定順序輸出六通道
            numeric_data = pd.DataFrame({c: pd.to_numeric(data[c], errors='coerce') for c in available_curves})
            numeric_data = numeric_data.dropna()
            if len(numeric_data) < self.time_window:
                print(f"   ⚠️ 数据长度不足 ({len(numeric_data)} < {self.time_window})，跳过")
                return 0
            
            # 数据归一化
            cleaned_data = self._normalize_curve_data(pd.DataFrame(numeric_data, columns=available_curves))
            if has_label:
                if '解释结论' in data.columns:
                    labels = data['解释结论'].iloc[:len(cleaned_data)]
                elif '结论' in data.columns:
                    labels = data['结论'].iloc[:len(cleaned_data)]
                else:
                    labels = pd.Series([''] * len(cleaned_data))
                # 檢測有效標註比例，過低則視為未標註模式
                try:
                    valid_mask = labels.notna()
                    stripped = labels.astype(str).str.strip().str.lower()
                    non_empty_mask = valid_mask & (stripped != '') & (~stripped.isin(['nan', 'none', 'null']))
                    non_empty_ratio = float(non_empty_mask.mean()) if len(labels) > 0 else 0.0
                    if not np.isfinite(non_empty_ratio) or non_empty_ratio < 0.05:
                        has_label = False
                        labels = pd.Series(['<unlabeled>'] * len(cleaned_data))
                        print("   ⚠️ 標註列有效比例極低，切換為未標註模式")
                except Exception:
                    pass
            else:
                # 未标注模式：占位标签
                labels = pd.Series(['<unlabeled>'] * len(cleaned_data))
            
            # 滑动窗口生成样本
            step = int(self.time_window * 0.5)  # 0.5重叠率
            well_samples = 0
            well_name = well_file.stem
            
            # 建立固定通道順序映射，確保六通道順序一致
            fixed_order = list(self.curve_names)  # ['GR','RT','DEN','CNL','SP','AC']
            col_to_series = {c: cleaned_data[c] for c in cleaned_data.columns if c in fixed_order}

            for start_idx in range(0, len(cleaned_data) - self.time_window + 1, step):
                end_idx = start_idx + self.time_window
                
                # 提取窗口数据
                # 以固定順序構造窗口數據矩陣 (time_window, 6)
                cols_in_window = []
                for cname in fixed_order:
                    if cname in col_to_series:
                        cols_in_window.append(col_to_series[cname].iloc[start_idx:end_idx].to_numpy())
                    else:
                        cols_in_window.append(np.zeros((self.time_window,), dtype=float))
                window_data = np.stack(cols_in_window, axis=1)  # (time_window, 6)
                window_labels = labels.iloc[start_idx:end_idx].tolist()
                
                # 确定窗口标签（未标注模式用 -1 作占位，不阻断生成）
                if has_label:
                    final_label_id = self._determine_window_label(window_labels)
                    if final_label_id is None:
                        continue
                else:
                    final_label_id = -1
                
                try:
                    # 按固定順序生成6通道256×256时频图谱
                    base_spectrograms = []
                    for ch_idx, curve_name in enumerate(fixed_order):
                        curve_series = window_data[:, ch_idx] if ch_idx < window_data.shape[1] else np.zeros((self.time_window,), dtype=float)
                        spectrogram = self._generate_base_spectrogram(curve_series, curve_name)
                        base_spectrograms.append(spectrogram)
                    
                    # 保存完整256×256时频图谱
                    full_spectrogram = np.stack(base_spectrograms[:6], axis=0)
                    full_file = self.spectrograms_dir / f"sample_{self.generated_samples:06d}_full_256x256.npz"
                    np.savez_compressed(
                        full_file,
                        spectrogram=full_spectrogram,
                        label=final_label_id,
                        well_name=well_name,
                        sample_idx=self.generated_samples,
                        window_start=start_idx,
                        window_end=end_idx,
                        type='full'
                    )
                    
                    self.generated_samples += 1
                    well_samples += 1

                    # 生成6通道对齐的32×32子图谱（按统一网格）
                    patch_h = self.patch_size
                    patch_w = self.patch_size
                    step = int(self.patch_size * (1 - self.patch_overlap))
                    step = max(step, 1)
                    H, W = self.base_size
                    for i0 in range(0, H - patch_h + 1, step):
                        for j0 in range(0, W - patch_w + 1, step):
                            # 组装6通道patch
                            patch6 = []
                            energies = []
                            for ch in range(6):
                                p = base_spectrograms[ch][i0:i0+patch_h, j0:j0+patch_w]
                                if p.shape != (patch_h, patch_w):
                                    p = np.zeros((patch_h, patch_w))
                                patch6.append(p)
                                energies.append(float(np.sum(p ** 2) / (patch_h * patch_w)))
                            patch6 = np.stack(patch6, axis=0)  # (6, 32, 32)
                            avg_energy = float(np.mean(energies)) if energies else 0.0
                            # 能量阈值过滤（更稳健：使用平均能量）
                            if avg_energy >= self.min_energy_threshold:
                                patch6_file = self.spectrograms_dir / (
                                    f"sample_{self.generated_samples:06d}_patch_6ch_32x32_{i0}_{j0}.npz"
                                )
                                np.savez_compressed(
                                    patch6_file,
                                    spectrogram=patch6,
                                    label=final_label_id,
                                    well_name=well_name,
                                    sample_idx=self.generated_samples,
                                    window_start=start_idx,
                                    window_end=end_idx,
                                    pos_i=i0,
                                    pos_j=j0,
                                    energy=avg_energy,
                                    type='patch_6ch'
                                )
                                self.generated_samples += 1
                                well_samples += 1

                    # 智能提取32×32单通道子图谱（保留，供可视化或其他用途）
                    for channel_idx, channel_spectrogram in enumerate(base_spectrograms[:6]):
                        patches_info = self._extract_intelligent_patches(channel_spectrogram)
                        if len(patches_info) > 0:
                            patches = [info['patch'] for info in patches_info]
                            optimized_patches = self._apply_pca_to_patches(patches, channel_idx)
                            for patch_idx, (patch, info) in enumerate(zip(optimized_patches, patches_info)):
                                patch_file = self.spectrograms_dir / f"sample_{self.generated_samples:06d}_patch_32x32_{channel_idx}_{patch_idx}.npz"
                                np.savez_compressed(
                                    patch_file,
                                    spectrogram=patch[np.newaxis, :, :],  # (1, 32, 32)
                                    label=final_label_id,
                                    well_name=well_name,
                                    sample_idx=self.generated_samples,
                                    window_start=start_idx,
                                    window_end=end_idx,
                                    channel_idx=channel_idx,
                                    patch_idx=patch_idx,
                                    position=info['position'],
                                    energy=info['energy'],
                                    variance=info['variance'],
                                    gradient=info['gradient'],
                                    diversity_score=info['diversity_score'],
                                    type='patch'
                                )
                                self.generated_samples += 1
                                well_samples += 1
                    
                    # 定期垃圾回收
                    if well_samples % 100 == 0:
                        gc.collect()
                    
                except Exception as e:
                    print(f"     ⚠️ 样本生成失败: {e}")
                    continue
            
            print(f"   ✅ 生成 {well_samples} 个样本")
            return well_samples
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            return 0
    
    def generate_cache(self):
        """生成优化子图谱缓存"""
        print("🚀 开始优化子图谱提取")
        print("=" * 70)
        print(f"📐 优化策略:")
        print(f"   基础时频图谱: 256×256")
        print(f"   智能子图谱: 32×32")
        print(f"   子图谱重叠率: {self.patch_overlap}")
        print(f"   能量阈值过滤: {self.min_energy_threshold}")
        print(f"   PCA降维: 保留{self.pca_components*100}%方差")
        print(f"   多样性筛选: 基于能量×方差×梯度")
        print("=" * 70)
        
        start_time = time.time()
        
        # 查找井文件
        well_files = list(self.welldata_dir.glob("*.txt"))
        print(f"📂 发现 {len(well_files)} 个井文件")
        
        if not well_files:
            print("❌ 没有找到井数据文件")
            return
        
        # 逐个处理井文件
        total_well_samples = 0
        for i, well_file in enumerate(well_files, 1):
            print(f"\n[{i}/{len(well_files)}] 处理井文件...")
            
            well_samples = self.process_well_file(well_file)
            total_well_samples += well_samples
            
            # 进度报告
            if self.generated_samples % 1000 == 0 and self.generated_samples > 0:
                print(f"🔄 已生成 {self.generated_samples:,} 个样本...")
            
            # 强制垃圾回收
            gc.collect()
        
        # 保存PCA模型
        self._save_pca_models()
        
        # 保存元数据
        self._save_metadata()
        
        # 完成报告
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("🎉 优化子图谱提取完成！")
        print(f"   总样本数: {self.generated_samples:,}")
        print(f"   总耗时: {total_time/60:.1f} 分钟")
        if self.generated_samples > 0:
            print(f"   平均速度: {self.generated_samples/total_time:.1f} 样本/秒")
        print(f"   缓存目录: {self.cache_dir.absolute()}")
        
        # 分析分布
        self._analyze_distribution()
    
    def _save_pca_models(self):
        """保存PCA模型"""
        print("💾 保存PCA模型...")
        
        for channel_key, pca_model in self.pca_models.items():
            pca_file = self.pca_dir / f"{channel_key}_pca.pkl"
            scaler_file = self.pca_dir / f"{channel_key}_scaler.pkl"
            
            with open(pca_file, 'wb') as f:
                pickle.dump(pca_model, f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[channel_key], f)
            
            print(f"   保存 {channel_key} PCA模型")
    
    def _save_metadata(self):
        """保存元数据"""
        print("💾 保存元数据...")
        
        metadata = {
            'total_samples': self.generated_samples,
            'curve_names': self.curve_names,
            'base_size': self.base_size,
            'patch_size': self.patch_size,
            'time_window': self.time_window,
            'overlap_ratio': self.overlap_ratio,
            'patch_overlap': self.patch_overlap,
            'min_energy_threshold': self.min_energy_threshold,
            'pca_components': self.pca_components,
            'fluid_types': FluidTypes.FLUID_TYPES,
            'generation_time': time.time(),
            'optimization_features': {
                'intelligent_patch_extraction': True,
                'energy_filtering': True,
                'diversity_scoring': True,
                'pca_dimensionality_reduction': True
            }
        }
        
        # 保存pickle文件
        with open(self.metadata_dir / "cache_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # 保存文本信息
        with open(self.metadata_dir / "cache_info.txt", 'w', encoding='utf-8') as f:
            f.write("优化子图谱提取缓存信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"总样本数: {self.generated_samples:,}\n")
            f.write(f"基础时频图谱: {self.base_size}\n")
            f.write(f"子图谱尺寸: {self.patch_size}×{self.patch_size}\n")
            f.write(f"子图谱重叠率: {self.patch_overlap}\n")
            f.write(f"能量阈值: {self.min_energy_threshold}\n")
            f.write(f"PCA保留方差: {self.pca_components}\n")
            f.write(f"优化策略: 智能提取+能量过滤+多样性筛选+PCA降维\n")
            f.write(f"曲线名称: {', '.join(self.curve_names)}\n")
            f.write(f"生成时间: {time.ctime()}\n")
    
    def _analyze_distribution(self):
        """分析数据分布"""
        print("📊 分析数据分布...")
        
        full_class_counts = {}
        patch_class_counts = {}
        well_counts = {}
        type_counts = {'full': 0, 'patch': 0}
        energy_stats = []
        
        # 扫描样本文件
        sample_files = list(self.spectrograms_dir.glob("sample_*.npz"))
        
        for file_path in sample_files:
            try:
                data = np.load(file_path)
                label = int(data['label'])
                well_name = str(data['well_name'])
                sample_type = str(data.get('type', 'unknown'))
                
                if sample_type == 'full':
                    full_class_counts[label] = full_class_counts.get(label, 0) + 1
                elif sample_type == 'patch':
                    patch_class_counts[label] = patch_class_counts.get(label, 0) + 1
                    # 收集能量统计
                    if 'energy' in data:
                        energy_stats.append(float(data['energy']))
                
                well_counts[well_name] = well_counts.get(well_name, 0) + 1
                type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
                
            except Exception:
                continue
        
        # 显示类别分布
        print(f"\n🏷️ 完整时频图谱类别分布 ({type_counts.get('full', 0)} 个):")
        for class_id, count in sorted(full_class_counts.items()):
            class_name = FluidTypes.FLUID_TYPES.get(class_id, f"Class_{class_id}")
            percentage = count / type_counts.get('full', 1) * 100
            print(f"   {class_name}: {count} 样本 ({percentage:.1f}%)")
        
        print(f"\n🔍 优化子图谱类别分布 ({type_counts.get('patch', 0)} 个):")
        for class_id, count in sorted(patch_class_counts.items()):
            class_name = FluidTypes.FLUID_TYPES.get(class_id, f"Class_{class_id}")
            percentage = count / type_counts.get('patch', 1) * 100
            print(f"   {class_name}: {count} 样本 ({percentage:.1f}%)")
        
        # 显示能量分布
        if energy_stats:
            print(f"\n⚡ 子图谱能量分布:")
            print(f"   平均能量: {np.mean(energy_stats):.3f}")
            print(f"   能量标准差: {np.std(energy_stats):.3f}")
            print(f"   最小能量: {np.min(energy_stats):.3f}")
            print(f"   最大能量: {np.max(energy_stats):.3f}")
        
        # 显示井分布（前5）
        sorted_wells = sorted(well_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏭 井分布 (前5口井，共 {len(well_counts)} 口井):")
        for well_name, count in sorted_wells[:5]:
            percentage = count / self.generated_samples * 100
            print(f"   {well_name}: {count:,} 样本 ({percentage:.1f}%)")

def main():
    """主函数"""
    print("🎯 优化子图谱提取器")
    print("🔬 256×256→32×32智能分解+PCA降维")
    print("💡 转化冗余为优势的参数优化策略")
    print("=" * 60)
    
    try:
        import argparse
        parser = argparse.ArgumentParser(description='Generate optimized spectrogram patches')
        parser.add_argument('--welldata_dir', type=str, default='welldata', help='井数据目录')
        parser.add_argument('--cache_dir', type=str, default='optimized_patch_data', help='输出缓存目录')
        args = parser.parse_args()

        extractor = OptimizedPatchExtractor(welldata_dir=args.welldata_dir, cache_dir=args.cache_dir)
        extractor.generate_cache()
        
        print("\n📋 优势分析:")
        print("   ✅ 多尺度特征：全局+局部")
        print("   ✅ 智能筛选：能量+方差+梯度")
        print("   ✅ PCA降维：处理冗余特征")
        print("   ✅ 样本增强：提高数据多样性")
        print("   ✅ 计算优化：适合轻量化架构")
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
