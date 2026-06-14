#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨模態交叉注意力模塊 (Cross-Modal Cross Attention)
用於輔助特徵引導主特徵學習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    跨模態交叉注意力機制
    
    輔助特徵作為Query，主特徵作為Key和Value，
    實現輔助特徵對主特徵的引導和增強
    """
    
    def __init__(self, main_dim, aux_dim, num_heads=4, dropout=0.1):
        """
        Args:
            main_dim: 主特徵維度（來自MDSC）
            aux_dim: 輔助特徵維度（熱圖+小波特徵）
            num_heads: 注意力頭數
            dropout: Dropout比例
        """
        super().__init__()
        
        self.main_dim = main_dim
        self.aux_dim = aux_dim
        self.num_heads = num_heads
        self.head_dim = main_dim // num_heads
        
        assert main_dim % num_heads == 0, "main_dim必須能被num_heads整除"
        
        # Query來自輔助特徵
        self.query_proj = nn.Linear(aux_dim, main_dim)
        
        # Key和Value來自主特徵
        self.key_proj = nn.Linear(main_dim, main_dim)
        self.value_proj = nn.Linear(main_dim, main_dim)
        
        # 輸出投影
        self.out_proj = nn.Linear(main_dim, main_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, main_features, aux_features):
        """
        Args:
            main_features: 主特徵 (B, main_dim, H, W)
            aux_features: 輔助特徵 (B, aux_dim, H, W)
        
        Returns:
            enhanced_features: 增強後的主特徵 (B, main_dim, H, W)
            attention_weights: 注意力權重 (B, num_heads, HW, HW)
        """
        B, C_main, H, W = main_features.shape
        _, C_aux, _, _ = aux_features.shape
        
        # 展平空間維度: (B, C, H, W) -> (B, HW, C)
        main_flat = main_features.flatten(2).permute(0, 2, 1)  # (B, HW, main_dim)
        aux_flat = aux_features.flatten(2).permute(0, 2, 1)    # (B, HW, aux_dim)
        
        # 計算Query, Key, Value
        Q = self.query_proj(aux_flat)   # (B, HW, main_dim)
        K = self.key_proj(main_flat)    # (B, HW, main_dim)
        V = self.value_proj(main_flat)  # (B, HW, main_dim)
        
        # 多頭分割: (B, HW, main_dim) -> (B, num_heads, HW, head_dim)
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 計算注意力分數: (B, num_heads, HW, HW)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 應用注意力: (B, num_heads, HW, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # 合併多頭: (B, num_heads, HW, head_dim) -> (B, HW, main_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, H*W, self.main_dim)
        
        # 輸出投影
        output = self.out_proj(attn_output)  # (B, HW, main_dim)
        
        # 恢復空間維度: (B, HW, main_dim) -> (B, main_dim, H, W)
        output = output.permute(0, 2, 1).view(B, self.main_dim, H, W)
        
        # 殘差連接
        enhanced_features = main_features + output
        
        return enhanced_features, attn_weights


class AuxiliaryFeatureFusion(nn.Module):
    """
    輔助特徵融合模塊
    
    將熱圖(1通道)和小波特徵(256通道)融合為統一的輔助特徵表示
    """
    
    def __init__(self, heatmap_channels=1, wpt_channels=256, output_channels=64):
        super().__init__()
        
        # 熱圖處理分支
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(heatmap_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 小波特徵處理分支（降維）
        self.wpt_conv = nn.Sequential(
            nn.Conv2d(wpt_channels, 128, 1),  # 1×1卷積降維
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 融合層
        self.fusion = nn.Sequential(
            nn.Conv2d(64, output_channels, 1),  # 32+32=64
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, heatmap, wpt_features):
        """
        Args:
            heatmap: 熱圖 (B, 1, H, W)
            wpt_features: 小波特徵 (B, 256, H, W)
        
        Returns:
            fused_aux: 融合後的輔助特徵 (B, output_channels, H, W)
        """
        # 處理兩個分支
        h_feat = self.heatmap_conv(heatmap)      # (B, 32, H, W)
        w_feat = self.wpt_conv(wpt_features)     # (B, 32, H, W)
        
        # 拼接並融合
        concat_feat = torch.cat([h_feat, w_feat], dim=1)  # (B, 64, H, W)
        fused_aux = self.fusion(concat_feat)              # (B, output_channels, H, W)
        
        return fused_aux


if __name__ == "__main__":
    # 測試代碼
    print("=" * 70)
    print("測試跨模態交叉注意力模塊")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建測試數據
    B, H, W = 4, 64, 64
    main_dim = 256
    aux_dim = 64
    
    main_features = torch.randn(B, main_dim, H, W).to(device)
    aux_features = torch.randn(B, aux_dim, H, W).to(device)
    
    # 測試CrossModalAttention
    print("\n1. 測試CrossModalAttention")
    cma = CrossModalAttention(main_dim, aux_dim, num_heads=4).to(device)
    enhanced, attn = cma(main_features, aux_features)
    print(f"   輸入主特徵: {main_features.shape}")
    print(f"   輸入輔助特徵: {aux_features.shape}")
    print(f"   輸出增強特徵: {enhanced.shape}")
    print(f"   注意力權重: {attn.shape}")
    
    # 測試AuxiliaryFeatureFusion
    print("\n2. 測試AuxiliaryFeatureFusion")
    heatmap = torch.randn(B, 1, H, W).to(device)
    wpt_feat = torch.randn(B, 256, H, W).to(device)
    
    fusion = AuxiliaryFeatureFusion(output_channels=64).to(device)
    fused = fusion(heatmap, wpt_feat)
    print(f"   輸入熱圖: {heatmap.shape}")
    print(f"   輸入小波特徵: {wpt_feat.shape}")
    print(f"   輸出融合特徵: {fused.shape}")
    
    print("\n✅ 所有測試通過！")



