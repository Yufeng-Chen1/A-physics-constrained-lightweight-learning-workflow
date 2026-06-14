#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案A增強模型：AWPD-MDSC-TAM + 跨模態注意力 + 輔助特徵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.cross_modal_attention import CrossModalAttention, AuxiliaryFeatureFusion
from models.lightweight_mdsc_tam import LightweightMDSC, LightweightTAM


class PlanAEnhancedModel(nn.Module):
    """
    方案A完整模型架構
    
    流程:
    1. 6通道時頻圖 → MDSC → 主特徵
    2. 熱圖+小波特徵 → 輔助特徵融合 → 輔助特徵
    3. 跨模態注意力：輔助特徵引導主特徵
    4. TAM注意力增強
    5. 分類頭輸出
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.input_channels = config.get('input_channels', 6)  # 6通道時頻圖
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        
        # ====================================================================
        # 主分支：MDSC (處理6通道時頻圖)
        # ====================================================================
        self.mdsc = LightweightMDSC(
            input_channels=self.input_channels,
            config=config
        )
        
        # MDSC輸出通道數（從配置獲取）
        self.mdsc_out_channels = config.get('fusion_channels', 256)
        
        # ====================================================================
        # 輔助分支：處理熱圖和小波特徵
        # ====================================================================
        self.aux_fusion = AuxiliaryFeatureFusion(
            heatmap_channels=1,
            wpt_channels=256,
            output_channels=64  # 輔助特徵維度
        )
        
        # ====================================================================
        # 跨模態交叉注意力：輔助特徵引導主特徵
        # ====================================================================
        self.cross_modal_attn = CrossModalAttention(
            main_dim=self.mdsc_out_channels,   # 主特徵維度
            aux_dim=64,                        # 輔助特徵維度
            num_heads=4,
            dropout=0.1
        )
        
        # ====================================================================
        # 通道注意力（簡化版TAM）
        # ====================================================================
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mdsc_out_channels, self.mdsc_out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mdsc_out_channels // 16, self.mdsc_out_channels, 1),
            nn.Sigmoid()
        )
        
        # ====================================================================
        # 全局池化和特徵/分類頭
        # ====================================================================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 特徵提取器（共享，用於SupCon Loss和分類）
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.mdsc_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.GELU()
        )
        
        # 分類頭（從特徵到類別）
        self.classifier = nn.Linear(128, self.num_classes)
        
    def forward(self, x, heatmap=None, wpt_features=None, return_features=False):
        """
        前向傳播
        
        Args:
            x: 6通道時頻圖 (B, 6, H, W)
            heatmap: 熱圖 (B, 1, H, W) - 可選
            wpt_features: 小波特徵 (B, 256, H, W) - 可選
            return_features: 是否返回特徵向量（用於SupCon Loss）
        
        Returns:
            logits: 分類logits (B, num_classes)
            features: 特徵向量 (B, 128) - 僅當return_features=True時
        """
        # 1. 主分支：MDSC處理6通道時頻圖
        main_features = self.mdsc(x)  # (B, 256, H', W')
        
        # 2. 輔助分支（如果提供）
        if heatmap is not None and wpt_features is not None:
            # 融合熱圖和小波特徵
            aux_features = self.aux_fusion(heatmap, wpt_features)  # (B, 64, H', W')
            
            # 確保輔助特徵與主特徵空間維度匹配
            if aux_features.shape[2:] != main_features.shape[2:]:
                aux_features = F.interpolate(
                    aux_features, 
                    size=main_features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 3. 跨模態交叉注意力：輔助特徵引導主特徵
            main_features, _ = self.cross_modal_attn(main_features, aux_features)
        
        # 4. 通道注意力增強
        channel_attn = self.channel_attention(main_features)
        enhanced_features = main_features * channel_attn  # (B, 256, H', W')
        
        # 5. 全局池化
        pooled = self.global_pool(enhanced_features)  # (B, 256, 1, 1)
        pooled = pooled.flatten(1)  # (B, 256)
        
        # 6. 特徵提取（用於SupCon Loss）
        features = self.feature_extractor(pooled)  # (B, 128)
        
        # 7. 分類（直接使用features）
        logits = self.classifier(features)  # (B, 5)
        
        # 總是返回兩個值，保持一致性（用於HybridLoss）
        return logits, features
    
    def get_num_params(self):
        """計算模型參數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_plan_a_model(config=None, device='cuda'):
    """
    創建方案A增強模型
    
    Args:
        config: 模型配置字典
        device: 設備
    
    Returns:
        model: 方案A模型實例
    """
    if config is None:
        config = {
            'input_channels': 6,
            'num_classes': 5,
            'conv_channels': [32, 64, 128],
            'kernel_sizes': [3, 5, 7],
            'fusion_channels': 256,
            'tam_reduction': 16,
            'tam_spatial_kernel': 7,
            'dropout_rate': 0.3
        }
    
    model = PlanAEnhancedModel(config).to(device)
    
    # 打印模型信息
    num_params = model.get_num_params()
    print(f"🔧 方案A增強模型創建成功")
    print(f"   總參數量: {num_params:,}")
    print(f"   模型大小: {num_params * 4 / 1024 / 1024:.2f} MB")
    print(f"   輸入通道: {config['input_channels']}")
    print(f"   輸出類別: {config['num_classes']}")
    print(f"   ✅ 集成跨模態注意力")
    print(f"   ✅ 集成輔助特徵融合")
    print(f"   ✅ 支持SupCon Loss")
    
    return model


if __name__ == "__main__":
    # 測試代碼
    print("=" * 70)
    print("測試方案A增強模型")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    config = {
        'input_channels': 6,
        'num_classes': 5,
        'conv_channels': [32, 64, 128],
        'kernel_sizes': [3, 5, 7],
        'fusion_channels': 256,
        'tam_reduction': 16,
        'tam_spatial_kernel': 7,
        'dropout_rate': 0.3
    }
    
    model = create_plan_a_model(config, device)
    
    # 測試前向傳播
    print("\n測試前向傳播:")
    B, H, W = 4, 64, 64
    
    # 輸入數據
    x = torch.randn(B, 6, H, W).to(device)
    heatmap = torch.randn(B, 1, H, W).to(device)
    wpt_features = torch.randn(B, 256, H, W).to(device)
    
    # 1. 僅使用主特徵
    print("\n1. 僅使用主特徵（6通道時頻圖）:")
    logits = model(x)
    print(f"   輸入: {x.shape}")
    print(f"   輸出logits: {logits.shape}")
    
    # 2. 使用主特徵 + 輔助特徵
    print("\n2. 使用主特徵 + 輔助特徵:")
    logits, features = model(x, heatmap, wpt_features, return_features=True)
    print(f"   輸入時頻圖: {x.shape}")
    print(f"   輸入熱圖: {heatmap.shape}")
    print(f"   輸入小波特徵: {wpt_features.shape}")
    print(f"   輸出logits: {logits.shape}")
    print(f"   輸出features: {features.shape}")
    
    print("\n✅ 所有測試通過！")

