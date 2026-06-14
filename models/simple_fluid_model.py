#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的流体识别模型
专门用于处理64x64的时频图谱输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFluidIdentificationModel(nn.Module):
    """
    简化的流体识别模型
    专门用于处理小波包分解后的时频图谱
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']  # 6个测井曲线
        # 使用模型配置中的實際類別數（由訓練器檢測後覆蓋），回退到DATA_CONFIG
        try:
            num_classes = int(config.MODEL_CONFIG.get('num_classes', config.DATA_CONFIG['num_classes']))
        except Exception:
            num_classes = config.DATA_CONFIG['num_classes']
        
        # 简化的CNN特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # 第四层卷积
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 (B, C, H, W) - 批次大小, 通道数(6), 高度(64), 宽度(64)
        Returns:
            分类结果 (B, num_classes)
        """
        # 特征提取
        features = self.feature_extractor(x)
        
        # 展平特征
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        return output


def create_simple_model(config):
    """创建简化的流体识别模型"""
    return SimpleFluidIdentificationModel(config)
