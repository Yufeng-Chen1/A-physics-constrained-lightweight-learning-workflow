#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的时间-频率特征提取器
实现多尺度小波包分解和丰富的时频联合特征提取
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class EnhancedTimeFrequencyExtractor:
    """增强的时间-频率特征提取器"""
    
    def __init__(self, config: Dict):
        """
        初始化时频特征提取器
        Args:
            config: 配置字典
        """
        self.config = config
        self.feature_size = config.get('feature_size', (64, 64))
        self.time_window = config.get('time_window', 32)
        self.overlap_ratio = config.get('overlap_ratio', 0.75)  # 75%重叠
        self.decomposition_levels = config.get('decomposition_levels', [4])  # 4层小波包分解
        
        # 小波配置
        self.wavelets = {
            'high_freq': ['db8', 'db10'],  # 高频曲线用多个小波
            'low_freq': ['sym8', 'coif4']   # 低频曲线用多个小波
        }
        
        print(f"🔧 增强时频特征提取器初始化完成")
        print(f"   特征尺寸: {self.feature_size}")
        print(f"   时间窗口: {self.time_window}")
        print(f"   重叠比例: {self.overlap_ratio}")
        print(f"   分解层数: {self.decomposition_levels}")

    def extract_multi_scale_features(self, curve_data: np.ndarray, curve_name: str) -> np.ndarray:
        """
        提取多尺度时频特征
        Args:
            curve_data: 测井曲线数据
            curve_name: 曲线名称
        Returns:
            多尺度时频特征矩阵
        """
        try:
            # 数据预处理
            curve_data = self._preprocess_data(curve_data)
            
            # 确定曲线类型和小波配置
            curve_type = self._get_curve_type(curve_name)
            wavelets = self.wavelets[curve_type]
            
            # 多尺度、多小波特征提取
            all_features = []
            
            for wavelet in wavelets:
                for level in self.decomposition_levels:
                    # 小波包分解
                    wp_features = self._extract_wavelet_packet_features(
                        curve_data, wavelet, level
                    )
                    all_features.append(wp_features)
            
            # 连续小波变换特征（对于高频曲线）
            if curve_type == 'high_freq':
                cwt_features = self._extract_cwt_features(curve_data)
                all_features.append(cwt_features)
            
            # 合并所有特征
            combined_features = self._combine_features(all_features)
            
            # 调整到目标尺寸
            final_features = self._resize_features(combined_features, self.feature_size)
            
            return final_features
            
        except Exception as e:
            print(f"⚠️  多尺度特征提取失败: {e}")
            return self._generate_fallback_features(curve_data)

    def _extract_wavelet_packet_features(self, data: np.ndarray, wavelet: str, level: int) -> np.ndarray:
        """提取4层小波包分解特征，生成高分辨率时频图谱"""
        try:
            # 小波包分解（4层）
            wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric')
            max_level = min(level, wp.maxlevel, 4)  # 确保最多4层
            
            # 获取第4层的所有16个子带
            nodes = wp.get_level(max_level, 'natural')
            coeffs = []
            
            for node in nodes:
                if len(node.data) > 0:
                    coeffs.append(node.data)
            
            if not coeffs:
                return np.zeros((16, len(data) // 16))  # 4层分解得到16个子带
            
            # 构建高分辨率系数矩阵
            coeffs_matrix = self._build_high_res_coeffs_matrix(coeffs, target_freq_bins=256)
            
            # 提取储层流体识别专用时频特征
            time_freq_features = self._extract_fluid_time_frequency_features(coeffs_matrix)
            
            return time_freq_features
            
        except Exception as e:
            print(f"⚠️  小波包特征提取失败: {e}")
            return np.zeros((256, 256))  # 返回256x256的零矩阵

    def _extract_time_frequency_features(self, coeffs_matrix: np.ndarray) -> np.ndarray:
        """提取时频联合特征"""
        try:
            # 计算能量谱
            energy_matrix = np.abs(coeffs_matrix) ** 2
            
            # 时间窗口分析
            step_size = int(self.time_window * (1 - self.overlap_ratio))
            num_windows = (energy_matrix.shape[1] - self.time_window) // step_size + 1
            
            features = []
            
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + self.time_window
                
                if end_idx <= energy_matrix.shape[1]:
                    window_data = energy_matrix[:, start_idx:end_idx]
                    
                    # 提取丰富的统计特征
                    window_features = self._extract_window_features(window_data)
                    features.append(window_features)
            
            if not features:
                return energy_matrix
            
            # 组合特征
            features_matrix = np.column_stack(features)
            
            return features_matrix
            
        except Exception as e:
            print(f"⚠️  时频特征提取失败: {e}")
            return coeffs_matrix
    
    def _build_high_res_coeffs_matrix(self, coeffs: List[np.ndarray], target_freq_bins: int = 256) -> np.ndarray:
        """构建高分辨率系数矩阵"""
        try:
            if not coeffs:
                return np.zeros((target_freq_bins, 64))
            
            # 确定最短长度
            min_length = min(len(coeff) for coeff in coeffs)
            
            # 截取或填充系数到统一长度
            aligned_coeffs = []
            for coeff in coeffs:
                if len(coeff) >= min_length:
                    aligned_coeffs.append(coeff[:min_length])
                else:
                    # 零填充
                    padded = np.zeros(min_length)
                    padded[:len(coeff)] = coeff
                    aligned_coeffs.append(padded)
            
            # 堆叠成矩阵
            coeffs_matrix = np.stack(aligned_coeffs, axis=0)
            
            # 调整到目标频率分辨率
            if coeffs_matrix.shape[0] < target_freq_bins:
                # 上采样到256个频率分量
                from scipy import ndimage
                scale_factor = target_freq_bins / coeffs_matrix.shape[0]
                coeffs_matrix = ndimage.zoom(coeffs_matrix, (scale_factor, 1), order=1)
            elif coeffs_matrix.shape[0] > target_freq_bins:
                # 下采样到256个频率分量
                indices = np.linspace(0, coeffs_matrix.shape[0]-1, target_freq_bins, dtype=int)
                coeffs_matrix = coeffs_matrix[indices, :]
            
            return coeffs_matrix
            
        except Exception as e:
            print(f"⚠️  高分辨率系数矩阵构建失败: {e}")
            return np.random.randn(target_freq_bins, 64) * 0.1
    
    def _extract_fluid_time_frequency_features(self, coeffs_matrix: np.ndarray) -> np.ndarray:
        """提取储层流体识别专用时频特征"""
        try:
            # 计算能量谱密度
            energy_matrix = np.abs(coeffs_matrix) ** 2
            
            # 0.5重叠率滑动窗口分析
            window_size = self.time_window
            step_size = int(window_size * 0.5)  # 0.5重叠率
            
            # 计算可以创建的窗口数量
            if energy_matrix.shape[1] < window_size:
                # 如果数据不够长，直接返回插值后的矩阵
                return self._resize_to_target(energy_matrix, (256, 256))
            
            num_windows = (energy_matrix.shape[1] - window_size) // step_size + 1
            target_time_bins = 256
            
            # 时频特征矩阵
            time_freq_features = np.zeros((256, target_time_bins))
            
            # 时间窗口特征提取
            for i in range(min(num_windows, target_time_bins)):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                if end_idx <= energy_matrix.shape[1]:
                    window_data = energy_matrix[:, start_idx:end_idx]
                    
                    # 计算窗口内的统计特征
                    window_features = self._compute_window_statistics(window_data)
                    
                    # 映射到时间索引
                    time_idx = int(i * target_time_bins / num_windows) if num_windows > 0 else i
                    time_idx = min(time_idx, target_time_bins - 1)
                    
                    time_freq_features[:, time_idx] = window_features
            
            # 插值填充空白时间点
            time_freq_features = self._interpolate_missing_times(time_freq_features)
            
            # 应用储层特征增强
            enhanced_features = self._enhance_reservoir_features(time_freq_features)
            
            return enhanced_features
            
        except Exception as e:
            print(f"⚠️  储层流体时频特征提取失败: {e}")
            return np.random.randn(256, 256) * 0.1
    
    def _compute_window_statistics(self, window_data: np.ndarray) -> np.ndarray:
        """计算窗口统计特征"""
        try:
            features = []
            
            # 对每个频率分量计算统计量
            for freq_idx in range(window_data.shape[0]):
                freq_data = window_data[freq_idx, :]
                
                # 基本统计量
                mean_val = np.mean(freq_data)
                std_val = np.std(freq_data)
                energy_val = np.sum(freq_data ** 2)
                
                # 选择主要特征（能量为主）
                features.append(energy_val)
            
            features_array = np.array(features)
            
            # 确保输出256个特征
            if len(features_array) < 256:
                # 插值扩展
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(features_array))
                x_new = np.linspace(0, 1, 256)
                f = interp1d(x_old, features_array, kind='linear', fill_value='extrapolate')
                features_array = f(x_new)
            elif len(features_array) > 256:
                # 下采样
                indices = np.linspace(0, len(features_array)-1, 256, dtype=int)
                features_array = features_array[indices]
            
            return features_array
            
        except Exception as e:
            return np.random.randn(256) * 0.1
    
    def _interpolate_missing_times(self, features: np.ndarray) -> np.ndarray:
        """插值填充缺失的时间点"""
        try:
            # 检查每个时间点是否有数据
            for t in range(features.shape[1]):
                if np.all(features[:, t] == 0):
                    # 找到最近的非零列进行插值
                    left_col = t - 1
                    right_col = t + 1
                    
                    while left_col >= 0 and np.all(features[:, left_col] == 0):
                        left_col -= 1
                    while right_col < features.shape[1] and np.all(features[:, right_col] == 0):
                        right_col += 1
                    
                    if left_col >= 0 and right_col < features.shape[1]:
                        # 线性插值
                        alpha = 0.5
                        features[:, t] = alpha * features[:, left_col] + (1-alpha) * features[:, right_col]
                    elif left_col >= 0:
                        features[:, t] = features[:, left_col]
                    elif right_col < features.shape[1]:
                        features[:, t] = features[:, right_col]
            
            return features
            
        except Exception:
            return features
    
    def _enhance_reservoir_features(self, features: np.ndarray) -> np.ndarray:
        """增强储层特征"""
        try:
            # 应用储层特征增强滤波
            enhanced = features.copy()
            
            # 平滑处理以突出储层特征
            from scipy.ndimage import gaussian_filter
            enhanced = gaussian_filter(enhanced, sigma=0.5)
            
            # 增强对比度
            enhanced = np.tanh(enhanced * 2.0) * 0.5 + 0.5
            
            # 能量归一化
            energy = np.sqrt(np.sum(enhanced ** 2))
            if energy > 1e-8:
                enhanced = enhanced / energy
            
            return enhanced
            
        except Exception:
            return features
    
    def _resize_to_target(self, matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """调整矩阵到目标形状"""
        try:
            from scipy import ndimage
            scale_factors = (target_shape[0] / matrix.shape[0], target_shape[1] / matrix.shape[1])
            resized = ndimage.zoom(matrix, scale_factors, order=1)
            return resized
        except Exception:
            return np.random.randn(*target_shape) * 0.1

    def _extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """提取时间窗口内的多维特征"""
        try:
            # 基础统计特征
            mean_vals = np.mean(window_data, axis=1)
            std_vals = np.std(window_data, axis=1)
            max_vals = np.max(window_data, axis=1)
            min_vals = np.min(window_data, axis=1)
            
            # 高阶统计特征
            skew_vals = []
            kurt_vals = []
            energy_vals = []
            
            for i in range(window_data.shape[0]):
                row_data = window_data[i, :]
                if len(row_data) > 3:  # 避免计算错误
                    skew_vals.append(skew(row_data))
                    kurt_vals.append(kurtosis(row_data))
                    energy_vals.append(np.sum(row_data ** 2))
                else:
                    skew_vals.append(0)
                    kurt_vals.append(0)
                    energy_vals.append(np.sum(row_data ** 2))
            
            # 频率域特征（每个频率成分的能量分布）
            freq_energy = np.sum(window_data, axis=1)
            
            # 时间域特征（每个时间点的能量分布）
            time_energy = np.sum(window_data, axis=0)
            time_centroid = np.sum(time_energy * np.arange(len(time_energy))) / (np.sum(time_energy) + 1e-8)
            
            # 组合所有特征
            features = np.concatenate([
                mean_vals,
                std_vals, 
                max_vals,
                min_vals,
                skew_vals,
                kurt_vals,
                energy_vals,
                freq_energy,
                [time_centroid]  # 时间质心
            ])
            
            return features
            
        except Exception as e:
            print(f"⚠️  窗口特征提取失败: {e}")
            return np.mean(window_data, axis=1)

    def _extract_cwt_features(self, data: np.ndarray) -> np.ndarray:
        """提取连续小波变换特征"""
        try:
            # 生成尺度序列
            scales = np.arange(1, 33)  # 32个尺度
            
            # 连续小波变换
            coefficients, frequencies = pywt.cwt(data, scales, 'cmor1.5-1.0', sampling_period=1.0)
            
            # 计算时频特征
            cwt_features = np.abs(coefficients)
            
            # 瞬时频率特征
            instantaneous_freq = self._compute_instantaneous_frequency(coefficients)
            
            # 合并CWT特征
            combined_cwt = np.vstack([cwt_features, instantaneous_freq[np.newaxis, :]])
            
            return combined_cwt
            
        except Exception as e:
            print(f"⚠️  CWT特征提取失败: {e}")
            return np.zeros((32, len(data)))

    def _compute_instantaneous_frequency(self, coefficients: np.ndarray) -> np.ndarray:
        """计算瞬时频率"""
        try:
            # 计算相位
            phase = np.angle(coefficients)
            
            # 计算瞬时频率（相位的时间导数）
            inst_freq = np.diff(phase, axis=1) / (2 * np.pi)
            
            # 取均值作为整体瞬时频率
            mean_inst_freq = np.mean(inst_freq, axis=0)
            
            # 补齐长度
            if len(mean_inst_freq) < coefficients.shape[1]:
                mean_inst_freq = np.pad(mean_inst_freq, (0, 1), mode='edge')
            
            return mean_inst_freq
            
        except Exception as e:
            print(f"⚠️  瞬时频率计算失败: {e}")
            return np.zeros(coefficients.shape[1])

    def _get_curve_type(self, curve_name: str) -> str:
        """确定曲线类型"""
        high_freq_curves = ['AC', 'RT']
        return 'high_freq' if curve_name in high_freq_curves else 'low_freq'

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """数据预处理"""
        # 处理异常值
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 平滑处理
        if len(data) > 5:
            data = signal.savgol_filter(data, window_length=5, polyorder=2)
        
        return data

    def _build_coeffs_matrix(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """构建系数矩阵"""
        if not coeffs:
            return np.array([[0]])
        
        # 找到最大长度
        max_length = max(len(coeff) for coeff in coeffs)
        
        # 填充到相同长度
        padded_coeffs = []
        for coeff in coeffs:
            if len(coeff) < max_length:
                padded = np.pad(coeff, (0, max_length - len(coeff)), mode='constant')
            else:
                padded = coeff[:max_length]
            padded_coeffs.append(padded)
        
        return np.vstack(padded_coeffs)

    def _combine_features(self, feature_list: List[np.ndarray]) -> np.ndarray:
        """合并多个特征矩阵"""
        if not feature_list:
            return np.zeros(self.feature_size)
        
        # 找到最大尺寸
        max_height = max(f.shape[0] for f in feature_list)
        max_width = max(f.shape[1] for f in feature_list)
        
        # 调整所有特征到相同尺寸
        resized_features = []
        for features in feature_list:
            resized = self._resize_features(features, (max_height, max_width))
            resized_features.append(resized)
        
        # 平均合并
        combined = np.mean(resized_features, axis=0)
        
        return combined

    def _resize_features(self, features: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整特征矩阵尺寸"""
        try:
            from scipy.ndimage import zoom
            
            if features.shape == target_size:
                return features
            
            zoom_factors = (target_size[0] / features.shape[0], 
                          target_size[1] / features.shape[1])
            
            resized = zoom(features, zoom_factors, order=1)
            
            # 确保精确尺寸
            if resized.shape[0] > target_size[0]:
                resized = resized[:target_size[0], :]
            if resized.shape[1] > target_size[1]:
                resized = resized[:, :target_size[1]]
            
            if resized.shape != target_size:
                temp = np.zeros(target_size)
                temp[:resized.shape[0], :resized.shape[1]] = resized
                resized = temp
            
            return resized
            
        except Exception as e:
            print(f"⚠️  特征尺寸调整失败: {e}")
            return np.zeros(target_size)

    def _generate_fallback_features(self, data: np.ndarray) -> np.ndarray:
        """生成备选特征"""
        try:
            # 使用简单的滑动窗口统计特征
            window_size = min(8, len(data) // 4)
            features = []
            
            for i in range(0, len(data) - window_size + 1, window_size // 2):
                window = data[i:i + window_size]
                feature_vec = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    np.median(window)
                ]
                features.append(feature_vec)
            
            if not features:
                return np.zeros(self.feature_size)
            
            features_matrix = np.array(features).T
            return self._resize_features(features_matrix, self.feature_size)
            
        except Exception as e:
            print(f"⚠️  备选特征生成失败: {e}")
            return np.zeros(self.feature_size)


def create_enhanced_extractor(config: Dict) -> EnhancedTimeFrequencyExtractor:
    """创建增强的时频特征提取器"""
    return EnhancedTimeFrequencyExtractor(config)


if __name__ == "__main__":
    # 测试代码
    test_config = {
        'feature_size': (64, 64),
        'time_window': 32,
        'overlap_ratio': 0.75,
        'decomposition_levels': [2, 3, 4]
    }
    
    extractor = create_enhanced_extractor(test_config)
    
    # 生成测试数据
    test_data = np.random.randn(100) + np.sin(np.linspace(0, 10*np.pi, 100))
    
    # 提取特征
    features = extractor.extract_multi_scale_features(test_data, 'AC')
    
    print(f"✅ 测试完成！")
    print(f"   输入数据长度: {len(test_data)}")
    print(f"   输出特征尺寸: {features.shape}")
    print(f"   特征范围: [{np.min(features):.4f}, {np.max(features):.4f}]")
