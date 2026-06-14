#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小波包分解时频图谱生成器 - 专门用于生成64×64×6的时频图谱
"""

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import warnings
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')

class WaveletSpectrogramGenerator:
    """小波包分解时频图谱生成器"""
    
    def __init__(self, config: Dict):
        """
        初始化时频图谱生成器
        Args:
            config: 配置字典，包含小波包分解参数
        """
        self.config = config
        self.wavelet_config = config.get('WAVELET_CONFIG', {})
        
        # 小波包分解参数
        self.scale_range = self.wavelet_config.get('scale_range', (5, 36))
        self.time_window = self.wavelet_config.get('time_window', 32)
        self.time_step = self.wavelet_config.get('time_step', 1)
        self.output_size = self.wavelet_config.get('output_size', (64, 64))
        self.decomposition_level = self.wavelet_config.get('decomposition_level', 4)
        self.curve_specific_wavelets = self.wavelet_config.get('curve_specific_wavelets', {})
        self.wavelet_mode = self.wavelet_config.get('wavelet_mode', 'symmetric')
        self.normalize_coefficients = self.wavelet_config.get('normalize_coefficients', True)
        self.use_energy_spectrum = self.wavelet_config.get('use_energy_spectrum', True)
        
        # 默认小波基
        self.default_wavelet = self.wavelet_config.get('wavelet_type', 'db8')
        
        print(f"🔧 小波包分解时频图谱生成器初始化完成")
        print(f"   尺度范围: {self.scale_range}")
        print(f"   时间窗口: {self.time_window}")
        print(f"   时间步长: {self.time_step}")
        print(f"   输出尺寸: {self.output_size}")
        print(f"   分解层数: {self.decomposition_level}")
    
    def generate_spectrogram(self, curve_data: np.ndarray, curve_name: str) -> np.ndarray:
        """
        为单个曲线生成时频图谱
        Args:
            curve_data: 曲线数据
            curve_name: 曲线名称
        Returns:
            时频图谱，形状为 (64, 64)
        """
        try:
            # 数据预处理
            curve_data = self._preprocess_curve_data(curve_data)
            
            # 获取曲线专用的小波基配置
            wavelet_type, level = self._get_curve_wavelet_config(curve_name)
            
            # 生成小波包分解时频图谱
            spectrogram = self._generate_wavelet_packet_spectrogram(
                curve_data, wavelet_type, level
            )
            
            # 调整尺寸到目标输出尺寸
            spectrogram = self._resize_spectrogram(spectrogram)
            
            # 归一化
            if self.normalize_coefficients:
                spectrogram = self._normalize_spectrogram(spectrogram)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  生成 {curve_name} 时频图谱失败: {e}")
            # 返回零填充的默认时频图谱
            return np.zeros(self.output_size)
    
    def _preprocess_curve_data(self, curve_data: np.ndarray) -> np.ndarray:
        """预处理曲线数据"""
        # 清理NaN/Inf值
        curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        if np.std(curve_data) > 0:
            curve_data = (curve_data - np.mean(curve_data)) / np.std(curve_data)
        else:
            curve_data = np.zeros_like(curve_data)
        
        # 确保数据长度足够
        min_length = max(self.scale_range[1], self.time_window) * 2
        if len(curve_data) < min_length:
            # 如果数据不足，进行填充
            padding = np.zeros(min_length - len(curve_data))
            curve_data = np.concatenate([curve_data, padding])
        
        return curve_data
    
    def _get_curve_wavelet_config(self, curve_name: str) -> Tuple[str, int]:
        """获取曲线专用的小波基配置"""
        if curve_name in self.curve_specific_wavelets:
            config = self.curve_specific_wavelets[curve_name]
            return config.get('wavelet', self.default_wavelet), config.get('level', self.decomposition_level)
        else:
            return self.default_wavelet, self.decomposition_level
    
    def _generate_wavelet_packet_spectrogram(self, curve_data: np.ndarray, 
                                           wavelet_type: str, level: int) -> np.ndarray:
        """生成小波包分解时频图谱"""
        try:
            # 使用小波包分解
            wp = pywt.WaveletPacket(data=curve_data, wavelet=wavelet_type, mode=self.wavelet_mode)
            
            # 获取指定层数的系数
            coeffs = []
            for node in wp.get_level(level, 'natural'):
                coeff = node.data
                if len(coeff) > 0:
                    coeffs.append(coeff)
            
            if not coeffs:
                # 如果没有系数，使用原始数据
                return self._generate_fallback_spectrogram(curve_data)
            
            # 将系数转换为矩阵
            coeffs_matrix = np.vstack(coeffs)
            
            # 生成时频图谱
            spectrogram = self._coefficients_to_spectrogram(coeffs_matrix)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  小波包分解失败: {e}")
            return self._generate_fallback_spectrogram(curve_data)
    
    def _coefficients_to_spectrogram(self, coeffs_matrix: np.ndarray) -> np.ndarray:
        """将小波系数转换为时频图谱"""
        try:
            # 计算能量谱
            if self.use_energy_spectrum:
                # 使用系数的平方作为能量
                energy_matrix = np.abs(coeffs_matrix) ** 2
            else:
                # 使用系数的绝对值
                energy_matrix = np.abs(coeffs_matrix)
            
            # 应用滑动时间窗口
            spectrogram = self._apply_time_windows(energy_matrix)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  系数转换失败: {e}")
            return np.zeros(self.output_size)
    
    def _apply_time_windows(self, energy_matrix: np.ndarray) -> np.ndarray:
        """应用时间窗口生成时频图谱"""
        try:
            height, width = energy_matrix.shape
            
            # 计算时间窗口的数量
            num_windows = (width - self.time_window) // self.time_step + 1
            
            # 初始化时频图谱
            spectrogram = np.zeros((height, num_windows))
            
            # 应用滑动窗口
            for i in range(num_windows):
                start_idx = i * self.time_step
                end_idx = start_idx + self.time_window
                
                if end_idx <= width:
                    window_data = energy_matrix[:, start_idx:end_idx]
                    # 计算窗口内的统计特征
                    spectrogram[:, i] = np.mean(window_data, axis=1)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  时间窗口应用失败: {e}")
            return energy_matrix
    
    def _generate_fallback_spectrogram(self, curve_data: np.ndarray) -> np.ndarray:
        """生成备选的时频图谱（当小波包分解失败时）"""
        try:
            # 使用短时傅里叶变换作为备选
            if len(curve_data) >= self.time_window:
                # 计算短时傅里叶变换
                f, t, Sxx = signal.spectrogram(
                    curve_data, 
                    fs=1.0, 
                    window='hann',
                    nperseg=self.time_window,
                    noverlap=self.time_window // 2
                )
                
                # 转换为对数尺度
                spectrogram = np.log10(Sxx + 1e-10)
                
                return spectrogram
            else:
                # 如果数据太短，使用简单的统计特征
                return self._generate_statistical_spectrogram(curve_data)
                
        except Exception as e:
            print(f"⚠️  备选时频图谱生成失败: {e}")
            return self._generate_statistical_spectrogram(curve_data)
    
    def _generate_statistical_spectrogram(self, curve_data: np.ndarray) -> np.ndarray:
        """生成基于统计特征的时频图谱"""
        try:
            # 使用不同尺度的统计特征
            scales = np.linspace(self.scale_range[0], self.scale_range[1], 8)
            spectrogram = np.zeros((len(scales), len(curve_data)))
            
            for i, scale in enumerate(scales):
                scale = int(scale)
                if scale <= len(curve_data):
                    # 计算滑动窗口的统计特征
                    for j in range(len(curve_data) - scale + 1):
                        window_data = curve_data[j:j + scale]
                        spectrogram[i, j] = np.std(window_data)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  统计特征时频图谱生成失败: {e}")
            return np.zeros((8, len(curve_data)))
    
    def _resize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """调整时频图谱尺寸到目标输出尺寸"""
        try:
            target_height, target_width = self.output_size
            
            # 使用双线性插值调整尺寸
            if spectrogram.shape != self.output_size:
                # 转换为torch张量进行插值
                spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0)
                
                # 使用双线性插值调整尺寸
                resized_tensor = F.interpolate(
                    spectrogram_tensor, 
                    size=self.output_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # 转换回numpy数组
                spectrogram = resized_tensor.squeeze().numpy()
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  时频图谱尺寸调整失败: {e}")
            # 使用简单的填充或截断
            return self._simple_resize(spectrogram)
    
    def _simple_resize(self, spectrogram: np.ndarray) -> np.ndarray:
        """简单的尺寸调整方法"""
        target_height, target_width = self.output_size
        current_height, current_width = spectrogram.shape
        
        # 创建目标尺寸的数组
        resized = np.zeros(self.output_size)
        
        # 复制数据
        copy_height = min(current_height, target_height)
        copy_width = min(current_width, target_width)
        
        resized[:copy_height, :copy_width] = spectrogram[:copy_height, :copy_width]
        
        return resized
    
    def _normalize_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """归一化时频图谱"""
        try:
            # 避免除零
            if np.max(spectrogram) > 0:
                spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️  时频图谱归一化失败: {e}")
            return spectrogram
    
    def generate_multi_curve_spectrograms(self, sequence_data: np.ndarray, 
                                        curve_names: List[str]) -> np.ndarray:
        """
        为多个曲线生成时频图谱
        Args:
            sequence_data: 序列数据，形状为 (curves, sequence_length)
            curve_names: 曲线名称列表
        Returns:
            多通道时频图谱，形状为 (6, 64, 64)
        """
        try:
            spectrograms = []
            
            for i, curve_name in enumerate(curve_names):
                if i < sequence_data.shape[0]:
                    curve_data = sequence_data[i]
                    spectrogram = self.generate_spectrogram(curve_data, curve_name)
                    spectrograms.append(spectrogram)
                else:
                    # 如果曲线数量不足，使用零填充
                    spectrograms.append(np.zeros(self.output_size))
            
            # 确保有6个通道
            while len(spectrograms) < 6:
                spectrograms.append(np.zeros(self.output_size))
            
            # 截断到6个通道
            if len(spectrograms) > 6:
                spectrograms = spectrograms[:6]
            
            # 堆叠为多通道时频图谱
            multi_channel_spectrogram = np.stack(spectrograms, axis=0)
            
            return multi_channel_spectrogram
            
        except Exception as e:
            print(f"⚠️  多曲线时频图谱生成失败: {e}")
            # 返回零填充的默认时频图谱
            return np.zeros((6, *self.output_size))
    
    def save_spectrogram_image(self, spectrogram: np.ndarray, save_path: str, 
                              title: str = "时频图谱"):
        """保存时频图谱图像"""
        try:
            plt.figure(figsize=(10, 8))
            
            if len(spectrogram.shape) == 3:
                # 多通道时频图谱，显示第一个通道
                plt.imshow(spectrogram[0], cmap='viridis', aspect='auto')
                plt.title(f"{title} - 通道1")
            else:
                # 单通道时频图谱
                plt.imshow(spectrogram, cmap='viridis', aspect='auto')
                plt.title(title)
            
            plt.colorbar(label='强度')
            plt.xlabel('时间')
            plt.ylabel('频率/尺度')
            plt.tight_layout()
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 时频图谱已保存: {save_path}")
            
        except Exception as e:
            print(f"⚠️  保存时频图谱图像失败: {e}")


def create_wavelet_spectrogram_generator(config: Dict) -> WaveletSpectrogramGenerator:
    """创建小波包分解时频图谱生成器"""
    return WaveletSpectrogramGenerator(config)
