#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化流体识别模型
专门用于高效训练和推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFluidIdentificationModel(nn.Module):
    """
    轻量化流体识别模型
    使用深度可分离卷积和注意力机制，大幅减少参数量
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']  # 6个测井曲线
        num_classes = config.DATA_CONFIG['num_classes']    # 11个类别
        
        # 轻量化特征提取器 - 使用深度可分离卷积
        self.feature_extractor = nn.Sequential(
            # 第一层：深度可分离卷积
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.Conv2d(input_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # 第二层：深度可分离卷积
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # 第三层：深度可分离卷积
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 轻量化分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
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


class EfficientFluidIdentificationModel(nn.Module):
    """
    高效流体识别模型
    使用MobileNet风格的架构
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']
        num_classes = config.DATA_CONFIG['num_classes']
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # 高效残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(16, 32, stride=2),
            self._make_residual_block(32, 64, stride=2),
            self._make_residual_block(64, 128, stride=2)
        ])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_residual_block(self, in_channels, out_channels, stride):
        """创建高效残差块"""
        return nn.Sequential(
            # 深度可分离卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 第二个深度可分离卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def _initialize_weights(self):
        """初始化权重"""
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
        """前向传播"""
        x = self.initial_conv(x)
        
        for block in self.residual_blocks:
            x = F.relu(x + block(x))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


def create_lightweight_model(config):
    """创建轻量化模型"""
    return LightweightFluidIdentificationModel(config)


def create_efficient_model(config):
    """创建高效模型"""
    return EfficientFluidIdentificationModel(config)
