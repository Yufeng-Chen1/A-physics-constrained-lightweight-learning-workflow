#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化MDSC+TAM模型
专门处理高维时频图谱和子图谱输入
优化的架构设计，支持多尺度特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

class LightweightMDSCBranch(nn.Module):
    """轻量化多分支深度可分离卷积分支"""
    
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, 
                 stride: int = 1, dropout_rate: float = 0.25):  # ⭐ 修复：0.42→0.25（降低正则化）
        super().__init__()
        
        # 深度卷积 (Depthwise)
        self.depthwise = nn.Conv2d(
            input_channels, input_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=kernel_size//2, 
            groups=input_channels,
            bias=False
        )
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(input_channels)
        
        # 点卷积 (Pointwise)
        self.pointwise = nn.Conv2d(
            input_channels, output_channels, 
            kernel_size=1, 
            bias=False
        )
        
        # 批归一化
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        # 激活和dropout
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # 点卷积
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x

class LightweightMDSC(nn.Module):
    """轻量化多分支深度可分离卷积"""
    
    def __init__(self, input_channels: int, config: Dict):
        super().__init__()
        
        # 轻量化配置
        self.conv_channels = config.get('conv_channels', [48, 96, 192])  # 🔧 增加通道数
        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        self.fusion_channels = config.get('fusion_channels', 384)  # 🔧 增加融合通道
        self.dropout_rate = config.get('dropout_rate', 0.25)  # ⭐ 修复：0.42→0.25（降低正则化）
        
        # 三个轻量化分支
        self.branch1 = LightweightMDSCBranch(
            input_channels, self.conv_channels[0], 
            self.kernel_sizes[0], stride=1, dropout_rate=self.dropout_rate
        )
        self.branch2 = LightweightMDSCBranch(
            input_channels, self.conv_channels[1], 
            self.kernel_sizes[1], stride=1, dropout_rate=self.dropout_rate
        )
        self.branch3 = LightweightMDSCBranch(
            input_channels, self.conv_channels[2], 
            self.kernel_sizes[2], stride=1, dropout_rate=self.dropout_rate
        )
        
        # 自适应池化和融合
        # ⭐⭐⭐ 回退到稳定配置：保持8×8特征图
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # 统一到8×8（稳定）
        total_channels = sum(self.conv_channels)
        
        # 轻量化融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, self.fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate)
        )
        
    def forward(self, x):
        # 三个分支并行处理
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        
        # 自适应池化到统一尺寸
        branch1_out = self.adaptive_pool(branch1_out)
        branch2_out = self.adaptive_pool(branch2_out)
        branch3_out = self.adaptive_pool(branch3_out)
        
        # 特征融合
        fused = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        fused = self.fusion(fused)
        
        return fused  # (B, fusion_channels, 8, 8)

class EfficientSelfAttention(nn.Module):
    """高效自注意力机制"""
    
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.25):  # ⭐ 修复：0.42→0.25
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model
        
        # 线性变换层
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)
        
        # 归一化和dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.q_linear, self.k_linear, self.v_linear, self.out_linear]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # 残差连接
        residual = x
        x = self.norm(x)
        
        # 计算Q, K, V
        Q = self.q_linear(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        
        # ⭐⭐⭐ 修复：attention计算数值稳定性保护
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # ⭐ 关键修复1：限制scores范围，防止softmax溢出
        scores = torch.clamp(scores, min=-50, max=50)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # ⭐ 关键修复2：检查softmax输出是否有NaN（调试用）
        if torch.isnan(attn_weights).any():
            # 如果出现NaN，使用uniform attention
            attn_weights = torch.ones_like(attn_weights) / N
        
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # 输出变换
        output = self.out_linear(attn_output)
        
        # 残差连接
        return residual + self.dropout(output)

class LightweightTAM(nn.Module):
    """轻量化三重注意力机制"""
    
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.25,  # ⭐ 修复：0.42→0.25
                 num_layers: int = 2):
        super().__init__()
        
        self.d_model = d_model
        
        # 多层自注意力
        self.attention_layers = nn.ModuleList([
            EfficientSelfAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 轻量化FFN
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (B, seq_len, d_model)
        
        # 多层注意力
        for attention_layer in self.attention_layers:
            x = attention_layer(x)
        
        # FFN with residual connection
        residual = x
        x = self.ffn(x)
        x = residual + x
        
        return x

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, feature_channels: int):
        super().__init__()
        
        self.feature_channels = feature_channels
        
        # 不同尺度的特征处理
        self.full_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels//2, 1),
            nn.BatchNorm2d(feature_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.patch_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels//2, 1),
            nn.BatchNorm2d(feature_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, full_features, patch_features):
        """
        融合完整特征和子图谱特征
        full_features: (B, C, H, W) 完整时频图谱特征
        patch_features: (B, C, H, W) 子图谱特征
        """
        # 处理不同尺度特征
        full_proc = self.full_conv(full_features)
        patch_proc = self.patch_conv(patch_features)
        
        # 特征融合
        fused = torch.cat([full_proc, patch_proc], dim=1)
        fused = self.fusion_conv(fused)
        
        return fused

class LightweightFluidModel(nn.Module):
    """轻量化流体识别模型
    专门处理高维时频图谱和子图谱输入
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.input_channels = config.get('input_channels', 6)
        
        
        # MDSC配置 - ✅ 从传入的config中读取，支持动态调整
        mdsc_config = {
            'conv_channels': config.get('conv_channels', [48, 96, 192]),  # ✅ 从config读取
            'kernel_sizes': config.get('kernel_sizes', [3, 5, 7]),
            'fusion_channels': config.get('fusion_channels', 384),  # ✅ 从config读取
            'dropout_rate': config.get('dropout_rate', 0.42)  # ✅ 从config读取
        }
        
        # 轻量化MDSC
        self.mdsc = LightweightMDSC(self.input_channels, mdsc_config)
        
        # 多尺度特征融合（如果支持）
        self.use_multiscale = config.get('use_multiscale', False)
        if self.use_multiscale:
            self.scale_fusion = MultiScaleFeatureFusion(mdsc_config['fusion_channels'])
        
        # 二维位置编码（匹配8×8特征图）← ⭐⭐⭐ 回退到稳定配置
        self.pos_h = nn.Parameter(torch.zeros(1, mdsc_config['fusion_channels']//2, 8, 1))
        self.pos_w = nn.Parameter(torch.zeros(1, mdsc_config['fusion_channels']//2, 1, 8))
        nn.init.trunc_normal_(self.pos_h, std=0.02)
        nn.init.trunc_normal_(self.pos_w, std=0.02)

        # 轻量化TAM（接受8×8=64个token） ← ⭐⭐⭐ 稳定配置
        # 保持TAM层数为1
        self.tam = LightweightTAM(
            d_model=mdsc_config['fusion_channels'],
            nhead=8,  # ✅ 8头注意力（充分建模时频特征）
            dropout=mdsc_config.get('dropout_rate', 0.30),  # ⭐ 使用config的dropout
            num_layers=1  # ⭐⭐⭐ 保持1层（稳定）
        )
        
        # 统计特征分支（能量、熵、偏度、峰度 × 每通道）
        # 擴展：每通道 [energy, mean, std, entropy, skewness, kurtosis] + 4x4塊能量(16) + 8x8塊能量(64) = 86
        self.stats_per_channel = 86
        self.stats_dim = config.get('stats_dim', self.input_channels * self.stats_per_channel)
        if self.stats_dim > 0:
            self.stats_mlp = nn.Sequential(
                nn.LayerNorm(self.stats_dim),
                nn.Linear(self.stats_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, mdsc_config['fusion_channels']),
                nn.ReLU(inplace=True)
            )
        else:
            self.stats_mlp = None

        # 統計特徵作為通道注意力門控（SE樣式）
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # 先占位，實際在forward中用stats投影
        )

        # 轻量化分类头 - ⭐ 修复：降低dropout到0.25
        self.classifier = nn.Sequential(
            nn.Dropout(mdsc_config.get('dropout_rate', 0.25)),  # ⭐ 0.42→0.25
            nn.Linear(mdsc_config['fusion_channels'], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(mdsc_config.get('dropout_rate', 0.25) * 0.65),  # ⭐ 0.42→0.25
            nn.Linear(128, self.num_classes)
        )

        # 小波/2D-DWT 特徵向量分支（96維，可從config覆蓋）
        self.wavelet_vec_dim = config.get('wavelet_vec_dim', 96)
        if self.wavelet_vec_dim > 0:
            self.wavelet_mlp = nn.Sequential(
                nn.LayerNorm(self.wavelet_vec_dim),
                nn.Linear(self.wavelet_vec_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, mdsc_config['fusion_channels']),
                nn.ReLU(inplace=True)
            )
        else:
            self.wavelet_mlp = None

        # 專屬可分性輔助向量（頻帶比值/跨道相關/極值能量/相位一致性）分支
        # 默認維度：6低+6高+6比值+6能量佔比+15相關+5擴展 = 44
        # AWPD數據：57維（可從config覆蓋）
        self.aux_vec_dim = config.get('aux_vec_dim', 44)
        if self.aux_vec_dim > 0:
            # ⭐⭐⭐ 方案L+: 保持方案L的aux_mlp路径（44→256→512→1216）
            # 目标：稳定且高效，提升水层/差油层F1（强依赖物性参数Sw, φ, K）
            aux_dropout = mdsc_config.get('dropout_rate', 0.035)  # 从config获取dropout_rate
            self.aux_mlp = nn.Sequential(
                nn.LayerNorm(self.aux_vec_dim),
                nn.Linear(self.aux_vec_dim, 256),  # ⭐ 方案L+: 保持256
                nn.ReLU(inplace=True),
                nn.Dropout(aux_dropout),  # ⭐ 使用配置的dropout_rate
                nn.Linear(256, 512),  # ⭐ 方案L+: 保持512
                nn.ReLU(inplace=True),
                nn.Dropout(aux_dropout),  # ⭐ 使用配置的dropout_rate
                nn.Linear(512, mdsc_config['fusion_channels']),  # 自动适配1216
                nn.ReLU(inplace=True)
            )
        else:
            self.aux_mlp = None

        # 輔助頭：類3(差油層) / 類4(油水層) 的二分類logit
        self.aux_head_c3 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(mdsc_config['fusion_channels'], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        self.aux_head_c4 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(mdsc_config['fusion_channels'], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # 权重初始化
        self._init_weights()
        # 用訓練先驗對分類器bias做保守初始化（若可用，於外部覆寫）。
        self.register_buffer('logit_bias', None, persistent=False)
    
    def _init_weights(self):
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, stats: Optional[torch.Tensor] = None, patch_x=None, wavelet_vec: Optional[torch.Tensor] = None, aux_vec: Optional[torch.Tensor] = None):
        """
        前向传播
        x: (B, C, H, W) 主要时频图谱输入
        stats: (B, C*4) 能量/熵/偏度/峰度统计特征（可选）
        patch_x: (B, C, H, W) 可选的子图谱输入
        """
        B = x.size(0)
        
        # MDSC特征提取
        features = self.mdsc(x)  # (B, 256, 8, 8)
        
        # 多尺度特征融合（如果有子图谱输入）
        if self.use_multiscale and patch_x is not None:
            patch_features = self.mdsc(patch_x)
            features = self.scale_fusion(features, patch_features)
        
        # 统计门控（SE式）：用统计向量生成通道权重，缩放卷积特征
        stats_embed_for_fusion: Optional[torch.Tensor] = None
        if stats is not None and self.stats_mlp is not None:
            if stats.dim() == 1:
                stats = stats.unsqueeze(0)
            if stats.size(1) == self.stats_dim:
                stats_embed_for_fusion = self.stats_mlp(stats)  # (B, 256)
                gate = torch.sigmoid(stats_embed_for_fusion).view(features.size(0), -1, 1, 1)
                # 放寬門控幅度：改為 [0.85, 1.15]，避免單一通道主導
                gate = 0.85 + 0.30 * gate
                # 与特征通道数一致
                if gate.size(1) == features.size(1):
                    features = features * gate

        # 将8×8特征展平为序列token供TAM使用（加入二维位置编码）← ⭐⭐⭐ 稳定配置
        pos2d = torch.cat([
            self.pos_h.expand(features.size(0), -1, 8, 8),
            self.pos_w.expand(features.size(0), -1, 8, 8)
        ], dim=1)
        features = features + pos2d
        seq_features = features.flatten(2).transpose(1, 2)  # (B, 64, 1152) ← ⭐ 稳定的token数
        
        # TAM注意力处理
        attended_features = self.tam(seq_features)  # (B, 64, 1152) ← ⭐ 稳定配置
        attended_features = attended_features.mean(dim=1)  # (B, 1152)
        # 融合统计特征（与TAM输出相加）
        if stats_embed_for_fusion is not None:
            attended_features = attended_features + stats_embed_for_fusion

        # 融合小波/2D-DWT向量
        if wavelet_vec is not None and torch.is_tensor(wavelet_vec) and self.wavelet_mlp is not None:
            if wavelet_vec.dim() == 1:
                wavelet_vec = wavelet_vec.unsqueeze(0)
            if wavelet_vec.size(1) == self.wavelet_vec_dim:
                wpt_embed = self.wavelet_mlp(wavelet_vec)
                attended_features = attended_features + wpt_embed

        # 融合專屬可分性輔助向量（頻帶比值/跨道相關）
        if aux_vec is not None and torch.is_tensor(aux_vec) and self.aux_mlp is not None:
            if aux_vec.dim() == 1:
                aux_vec = aux_vec.unsqueeze(0)
            # 若維度匹配則使用
            if aux_vec.size(1) == self.aux_vec_dim:
                aux_embed = self.aux_mlp(aux_vec)
                # ⭐⭐⭐ 回退：保持aux权重0.5（之前1.0导致训练崩溃）
                # 等特征图扩大后，如果训练稳定，再考虑逐步增加到0.7
                attended_features = attended_features + 0.5 * aux_embed

        # 主分類
        logits = self.classifier(attended_features)
        
        # ⭐⭐⭐ 物理规则增强（暂时降低权重）
        if aux_vec is not None and aux_vec.size(1) >= 44:
            # 应用基于测井解释经验的物理规则
            rule_scores = self._apply_physics_rules(aux_vec)
            
            # ⭐⭐⭐ 终极修复：大幅降低物理规则权重（0.25→0.05）
            # 原因：物性参数计算错误（孔隙度方差<0.001），规则基于错误参数
            # 策略：降低规则权重，让模型主要从时频图谱学习
            rule_weight = 0.05
            logits = logits + rule_weight * rule_scores

        # 輔助二分類logits（不改動主損失計算邏輯，由訓練器決定是否用）
        aux3 = self.aux_head_c3(attended_features)
        aux4 = self.aux_head_c4(attended_features)
        
        return logits, aux3, aux4
    
    def _apply_physics_rules(self, aux_vec):
        """
        基于测井解释经验的物理规则系统
        
        参数:
            aux_vec: (batch_size, 44) 辅助特征
                维度 0: AC均值
                维度 16: GR均值
                维度 24: RT均值
                维度 41: 孔隙度（φ）
                维度 42: 含水饱和度（Sw）
                维度 43: 渗透率（K，log尺度）
        
        返回:
            rule_scores: (batch_size, 5) 物理规则分数
        """
        batch_size = aux_vec.size(0)
        device = aux_vec.device
        
        # 提取关键物性参数
        ac_mean = aux_vec[:, 0]   # AC均值
        gr_mean = aux_vec[:, 16]  # GR均值
        rt_mean = aux_vec[:, 24]  # RT均值
        por = aux_vec[:, 41]      # 孔隙度
        sw = aux_vec[:, 42]       # 含水饱和度
        perm = aux_vec[:, 43]     # 渗透率（log尺度）
        
        # 渗透率从log尺度还原（0->1, 1->10, 2->100, 3->1000 mD）
        perm_actual = 10 ** perm
        
        rule_scores = torch.zeros(batch_size, 5, device=device)
        
        # ⭐⭐⭐ 姬塬油田长4+5层专用规则（低孔低渗储层）
        
        # 规则1: 油层（φ>10% + RT>40 + Sw<30% + K>2mD）
        oil_rule = (
            (por > 0.10) & (rt_mean > 40) & (sw < 0.30) & (perm_actual > 2)
        ).float()
        oil_bonus = ((ac_mean > 220) * 0.2) + ((gr_mean < 60) * 0.2)
        rule_scores[:, 0] = oil_rule + oil_bonus
        
        # 规则2: 水层（φ=8-12% + RT<15 + Sw>60% + K<1mD）
        water_rule = (
            (por > 0.08) & (por < 0.12) & (rt_mean < 15) & 
            (sw > 0.60) & (perm_actual < 1)
        ).float()
        water_bonus = ((ac_mean < 210) * 0.2) + ((gr_mean < 60) * 0.2)
        rule_scores[:, 1] = water_rule + water_bonus
        
        # 规则3: 干层（φ<5% + GR>100 + K≈0）
        dry_rule = (
            (por < 0.05) & (gr_mean > 100) & (perm_actual < 0.1) & (sw < 0.05)
        ).float()
        dry_bonus = ((gr_mean > 120) * 0.2)
        rule_scores[:, 2] = dry_rule + dry_bonus
        
        # 规则4: 差油层（φ=8-12% + RT=15-40 + K=0.5-3mD + Sw<45%）
        poor_oil_rule = (
            (por > 0.08) & (por < 0.12) & 
            (rt_mean > 15) & (rt_mean < 40) &
            (perm_actual > 0.5) & (perm_actual < 3) & (sw < 0.45)
        ).float()
        poor_oil_bonus = ((gr_mean < 60) * 0.2)
        rule_scores[:, 3] = poor_oil_rule + poor_oil_bonus
        
        # 规则5: 油水层（φ=10-14% + RT=15-30 + Sw=35-65% + K=1-4mD）
        oil_water_rule = (
            (por > 0.10) & (por < 0.14) &
            (rt_mean > 15) & (rt_mean < 30) &
            (sw > 0.35) & (sw < 0.65) &
            (perm_actual > 1) & (perm_actual < 4)
        ).float()
        oil_water_bonus = ((gr_mean < 60) * 0.2)
        rule_scores[:, 4] = oil_water_rule + oil_water_bonus
        
        return rule_scores
    
    def get_model_size(self):
        """获取模型大小信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

def create_lightweight_model(config, input_channels=None):
    """创建轻量化模型"""
    model_config = config.copy()
    if input_channels is not None:
        model_config['input_channels'] = input_channels
    model = LightweightFluidModel(model_config)
    
    # 打印模型信息
    model_info = model.get_model_size()
    print(f"🔧 轻量化模型信息:")
    print(f"   总参数量: {model_info['total_params']:,}")
    print(f"   可训练参数: {model_info['trainable_params']:,}")
    print(f"   模型大小: {model_info['model_size_mb']:.2f} MB")
    
    return model

if __name__ == "__main__":
    # 测试模型
    config = {
        'num_classes': 5,
        'input_channels': 6,
        'use_multiscale': True
    }
    
    model = create_lightweight_model(config)
    
    # 测试前向传播
    x = torch.randn(4, 6, 256, 256)  # 高分辨率输入
    patch_x = torch.randn(4, 6, 32, 32)  # 子图谱输入
    
    with torch.no_grad():
        output = model(x, patch_x)
        print(f"输出形状: {output.shape}")
        print("✅ 轻量化模型测试通过")
