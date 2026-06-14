#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
极简调试配置 - 用于快速诊断问题
"""

class MinimalDebugConfig:
    """极简调试配置"""

    # 数据配置 - 极简设置
    DATA_CONFIG = {
        'curve_names': ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC'],
        'num_curves': 6,
        'num_classes': 6,
        'sequence_length': 64,  # 减少序列长度
        'batch_size': 16,  # 极小批次大小
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'enable_wavelet': False,  # 禁用小波变换
        'enable_cleaning': False,  # 禁用数据清理
        'image_size': (64, 64),
        'num_workers': 0,
        'pin_memory': False,  # 禁用内存固定
        'shuffle': True,
        'drop_last': True,
        'overlap_ratio': 0.0,  # 禁用重叠
        'use_all_wells': True,
        'min_sequence_length': 32,
        'max_sequence_length': 64,
        # 禁用复杂功能
        'enable_pca': False,
        'enable_bsmote': False,
        'enable_class_weights': False
    }

    # 训练配置 - 极简设置
    TRAIN_CONFIG = {
        'epochs': 10,  # 只训练10轮用于调试
        'learning_rate': 0.0001,  # 极小学习率
        'weight_decay': 1e-4,
        'use_focal_loss': False,  # 禁用Focal Loss
        'gradient_clip': True,
        'gradient_clip_value': 1.0,
        'use_amp': False,  # 禁用混合精度
        'debug_mode': True,  # 启用调试
        'early_stopping_patience': 50,
        'min_delta': 0.01
    }

    # Transformer配置 - 极简设置
    TRANSFORMER_CONFIG = {
        'd_model': 64,  # 减少模型尺寸
        'nhead': 4,     # 减少注意力头数
        'num_layers': 1,  # 只用1层
        'dim_feedforward': 128,  # 减少前馈维度
        'dropout': 0.1,
        'use_positional_encoding': True
    }

    # 数据路径
    DATA_PATHS = {
        'welldata_dir': 'welldata'
    }

    # 其他配置保持默认
    WAVELET_CONFIG = {}
    CLEANING_CONFIG = {}
    CNN_CONFIG = {}
