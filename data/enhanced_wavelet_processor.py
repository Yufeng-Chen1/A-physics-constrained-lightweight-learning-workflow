#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的小波包分解处理器
实现分类型小波包分解策略，支持高频/低频曲线的不同处理方式
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import random

warnings.filterwarnings('ignore')

class EnhancedWaveletProcessor:
    """增强的小波包分解处理器"""
    
    def __init__(self, config: Dict):
        """
        初始化小波包分解处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.curve_configs = self._create_curve_configs()
        
        print("🔧 增强小波包分解处理器初始化完成")
        print("   高频曲线 (AC/RT): Morlet小波, 5层分解, 16采样点窗口")
        print("   低频曲线 (GR/SP/DEN): Sym8小波, 4层分解, 32采样点窗口")
        print("   重叠率: 75%, 补偿模式: periodic")
    
    def _create_curve_configs(self) -> Dict:
        """创建测井曲线特定配置"""
        return {
            # 统一输出尺寸为64x64，确保所有曲线输出一致
            'AC': {
                'type': 'high_freq',
                'wavelet': 'morlet',
                'level': 5,
                'time_window': 16,
                'output_size': (64, 64),
                'sigma': 1.0,
                'frequencies': 8
            },
            'RT': {
                'type': 'high_freq',
                'wavelet': 'morlet',
                'level': 5,
                'time_window': 16,
                'output_size': (64, 64),
                'sigma': 1.0,
                'frequencies': 8
            },
            # 低频曲线配置
            'GR': {
                'type': 'low_freq',
                'wavelet': 'sym8',
                'level': 4,
                'time_window': 32,
                'output_size': (64, 64),
                'mode': 'symmetric'
            },
            'SP': {
                'type': 'low_freq',
                'wavelet': 'sym8',
                'level': 4,
                'time_window': 32,
                'output_size': (64, 64),
                'mode': 'symmetric'
            },
            'DEN': {
                'type': 'low_freq',
                'wavelet': 'sym8',
                'level': 4,
                'time_window': 32,
                'output_size': (64, 64),
                'mode': 'symmetric'
            },
            'CNL': {
                'type': 'low_freq',
                'wavelet': 'sym8',
                'level': 4,
                'time_window': 32,
                'output_size': (64, 64),
                'mode': 'symmetric'
            }
        }
    
    def process_curve_data(self, curve_data: np.ndarray, curve_name: str, 
                          enable_augmentation: bool = False) -> np.ndarray:
        """
        处理单条测井曲线数据
        
        Args:
            curve_data: 曲线数据
            curve_name: 曲线名称
            enable_augmentation: 是否启用数据增强
            
        Returns:
            处理后的时频图谱
        """
        try:
            # 获取曲线配置
            curve_config = self.curve_configs.get(curve_name, self.curve_configs['GR'])
            
            # 数据预处理
            curve_data = self._preprocess_data(curve_data)
            
            # 应用数据增强（仅训练集）
            if enable_augmentation:
                curve_data = self._apply_data_augmentation(curve_data, curve_config)
            
            # 小波包分解
            if curve_config['type'] == 'high_freq':
                spectrogram = self._process_high_freq_curve(curve_data, curve_config)
            else:
                spectrogram = self._process_low_freq_curve(curve_data, curve_config)
            
            # 调整到目标尺寸
            spectrogram = self._resize_spectrogram(spectrogram, curve_config['output_size'])
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️ 处理曲线 {curve_name} 失败: {e}")
            # 返回零填充的默认时频图谱
            return np.zeros(curve_config['output_size'])
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """数据预处理"""
        # 处理NaN和Inf值
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 如果数据全为零，添加微小噪声
        if np.all(data == 0):
            data = np.random.normal(0, 0.01, len(data))
        
        return data
    
    def _apply_data_augmentation(self, data: np.ndarray, curve_config: Dict) -> np.ndarray:
        """应用数据增强"""
        # 时间偏移: ±2时间步
        if len(data) > 4:
            shift = random.randint(-2, 2)
            if shift != 0:
                data = np.roll(data, shift)
        
        # 加噪: 高斯白噪声 (SNR=20dB)
        snr_db = 20
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(data))
        data = data + noise
        
        return data
    
    def _process_high_freq_curve(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """处理高频曲线 (AC/RT)"""
        try:
            # 使用Morlet小波进行连续小波变换
            scales = np.arange(1, config['level'] + 1)

            # 计算连续小波变换 - 使用PyWavelets库，更稳定可靠
            wavelet = 'morl'  # 使用标准的Morlet小波
            cwt_coeffs, frequencies = pywt.cwt(data, scales, wavelet)

            # 转换为时频图谱
            spectrogram = np.abs(cwt_coeffs)

            # 应用频率掩蔽 (20%比例)
            if random.random() < 0.2:
                spectrogram = self._apply_frequency_masking(spectrogram)

            return spectrogram

        except Exception as e:
            print(f"⚠️ 高频曲线处理失败: {e}")
            return self._generate_fallback_spectrogram(data, config)
    
    def _process_low_freq_curve(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """处理低频曲线 (GR/SP/DEN/CNL)"""
        try:
            # 使用Sym8小波进行小波包分解
            wp = pywt.WaveletPacket(data, config['wavelet'], mode='periodic')
            
            # 获取指定层数的系数，确保不超过最大分解层数
            max_level = wp.maxlevel
            actual_level = min(config['level'], max_level)
            
            coeffs = []
            for node in wp.get_level(actual_level, 'natural'):
                if len(node.data) > 0:
                    coeffs.append(node.data)
            
            if not coeffs:
                return self._generate_fallback_spectrogram(data, config)
            
            # 转换为时频图谱
            coeffs_matrix = np.vstack(coeffs)
            spectrogram = np.abs(coeffs_matrix) ** 2
            
            # 应用频率掩蔽 (20%比例)
            if random.random() < 0.2:
                spectrogram = self._apply_frequency_masking(spectrogram)
            
            return spectrogram
            
        except Exception as e:
            print(f"⚠️ 低频曲线处理失败: {e}")
            return self._generate_fallback_spectrogram(data, config)
    
    def _apply_frequency_masking(self, spectrogram: np.ndarray) -> np.ndarray:
        """应用频率掩蔽 - 动态和随机版本"""
        height, width = spectrogram.shape
        mask_ratio = 0.2  # 掩蔽比例
        num_masks = random.randint(1, 3)  # 随机生成1到3个掩蔽块

        for _ in range(num_masks):
            mask_h = random.randint(int(height * 0.05), int(height * mask_ratio))
            mask_w = random.randint(int(width * 0.05), int(width * mask_ratio))

            start_h = random.randint(0, height - mask_h) if height > mask_h else 0
            start_w = random.randint(0, width - mask_w) if width > mask_w else 0

            end_h = start_h + mask_h
            end_w = start_w + mask_w

            spectrogram[start_h:end_h, start_w:end_w] = 0

        return spectrogram

    def _resize_spectrogram(self, spectrogram: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整时频图谱尺寸"""
        try:
            if spectrogram.shape == target_size:
                return spectrogram
            
            # 使用双线性插值调整尺寸
            zoom_factors = (target_size[0] / spectrogram.shape[0], 
                          target_size[1] / spectrogram.shape[1])
            resized = zoom(spectrogram, zoom_factors, order=1)
            
            # 确保尺寸正确
            if resized.shape != target_size:
                # 如果尺寸不匹配，进行裁剪或填充
                result = np.zeros(target_size)
                min_h = min(resized.shape[0], target_size[0])
                min_w = min(resized.shape[1], target_size[1])
                result[:min_h, :min_w] = resized[:min_h, :min_w]
                return result
            
            return resized
            
        except Exception as e:
            print(f"⚠️ 尺寸调整失败: {e}")
            return np.zeros(target_size)
    
    def _generate_fallback_spectrogram(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """生成备选时频图谱 - 优化版本"""
        target_size = config.get('output_size', (64, 64))
        if data.size == 0 or len(data) < 4:  # 数据过短或为空
            print(f"⚠️  数据过短或为空 ({len(data)}), 返回零填充的图谱")
            return np.zeros(target_size, dtype=np.float32)

        try:
            # 使用滑动窗口生成统计特征
            window_size = config.get('time_window', 32)
            if len(data) < window_size:
                # 如果数据太短，进行简单的线性插值或重复填充以达到最小窗口大小
                if len(data) > 1:
                    x = np.linspace(0, 1, len(data))
                    f = interp1d(x, data, kind='linear', fill_value="extrapolate")
                    x_new = np.linspace(0, 1, window_size)
                    data = f(x_new)
                else:
                    data = np.full(window_size, data[0] if data.size > 0 else 0.0)

            features = []
            step_size = max(1, window_size // 4) # 确保步长至少为1
            for i in range(0, len(data) - window_size + 1, step_size):
                window_data = data[i:i + window_size]
                if len(window_data) == window_size:
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    max_val = np.max(window_data)
                    min_val = np.min(window_data)
                    features.append([mean_val, std_val, max_val, min_val])

            if not features:
                return np.zeros(target_size, dtype=np.float32)

            features_matrix = np.array(features).T
            spectrogram = np.abs(features_matrix)

            return self._resize_spectrogram(spectrogram, target_size)

        except Exception as e:
            print(f"⚠️  备选时频图谱生成失败: {e}")
            return np.zeros(target_size, dtype=np.float32)
    
    def process_batch_data(self, batch_data: np.ndarray, curve_names: List[str], 
                          enable_augmentation: bool = False) -> np.ndarray:
        """
        处理批次数据
        
        Args:
            batch_data: 批次数据 (batch_size, sequence_length, num_curves)
            curve_names: 曲线名称列表
            enable_augmentation: 是否启用数据增强
            
        Returns:
            处理后的时频图谱 (batch_size, num_curves, height, width)
        """
        # 处理不同格式的输入数据
        if len(batch_data.shape) == 3:
            # 格式: (batch_size, sequence_length, num_curves)
            batch_size, seq_len, num_curves = batch_data.shape
            processed_batch = []

            for i in range(batch_size):
                sample_spectrograms = []
                for j, curve_name in enumerate(curve_names):
                    curve_data = batch_data[i, :, j]
                    spectrogram = self.process_curve_data(curve_data, curve_name, enable_augmentation)
                    sample_spectrograms.append(spectrogram)

                processed_batch.append(np.stack(sample_spectrograms, axis=0))

        elif len(batch_data.shape) == 4:
            # 格式: (batch_size, num_curves, height, width) - 每个通道已经是图像格式
            batch_size, num_curves, height, width = batch_data.shape
            processed_batch = []

            for i in range(batch_size):
                sample_spectrograms = []
                for j, curve_name in enumerate(curve_names):
                    # 直接传递2D图像数据进行处理
                    # 注意: process_curve_data 目前处理1D数据，需要调整其逻辑或在这里做适配
                    # 暂时保持原始行为，但会打印警告
                    curve_image = batch_data[i, j, :, :]
                    
                    # 将2D图像展平为1D序列，以便 process_curve_data 处理
                    # 这可能不是理想的，但符合当前 process_curve_data 的输入期望
                    # 如果 process_curve_data 适配处理2D图像，这里需要修改
                    curve_data_1d = curve_image.flatten()
                    spectrogram = self.process_curve_data(curve_data_1d, curve_name, enable_augmentation)
                    sample_spectrograms.append(spectrogram)

                processed_batch.append(np.stack(sample_spectrograms, axis=0))

        else:
            raise ValueError(f"不支持的输入数据格式: {batch_data.shape}")

        return np.array(processed_batch)


# 测试函数
def test_enhanced_wavelet_processor():
    """测试增强小波包分解处理器"""
    print("🧪 测试增强小波包分解处理器")
    print("=" * 50)
    
    # 创建测试配置
    config = {
        'curve_names': ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC'],
        'enable_augmentation': True
    }
    
    # 创建处理器
    processor = EnhancedWaveletProcessor(config)
    
    # 生成测试数据
    np.random.seed(42)
    test_data = np.random.randn(100, 6)  # 100个采样点，6条曲线
    
    # 测试单条曲线处理
    for curve_name in config['curve_names']:
        curve_data = test_data[:, 0]  # 使用第一列作为测试数据
        spectrogram = processor.process_curve_data(curve_data, curve_name, enable_augmentation=True)
        print(f"   {curve_name}: {spectrogram.shape}")
    
    print("✅ 测试完成")

if __name__ == "__main__":
    test_enhanced_wavelet_processor()
