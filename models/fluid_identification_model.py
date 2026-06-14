#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的流体识别模型
结合CNN、Transformer和注意力机制，提高模型表达能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x的形状是 (batch_size, seq_len, d_model)
        # self.pe的形状是 (max_len, 1, d_model)
        # 需要正确匹配维度
        seq_len = x.size(1)
        # 确保序列长度不超过位置编码的最大长度
        if seq_len > self.pe.size(0):
            # 如果序列长度超过最大长度，使用重复的位置编码
            pe = self.pe.repeat((seq_len // self.pe.size(0)) + 1, 1, 1)
            pe = pe[:seq_len, :, :]
        else:
            pe = self.pe[:seq_len, :, :]
        
        return x + pe.transpose(0, 1)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # 改进权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """更激进的权重初始化"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            # 使用更激进的初始化
            nn.init.xavier_uniform_(module.weight, gain=3.0)  # 增加gain
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.2)  # 进一步增加偏置初始值
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑并线性变换
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class MDSCBranch(nn.Module):
    """多分支深度可分离卷积 (MDSC) 的单个分支"""
    
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, dropout_rate=0.3):
        super().__init__()
        
        padding = kernel_size // 2
        
        # 深度可分离卷积
        self.depthwise = nn.Conv2d(
            input_channels, input_channels, 
            kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=input_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            input_channels, output_channels, 
            kernel_size=1, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # 逐点卷积
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class MDSC(nn.Module):
    """多分支深度可分离卷积 (Multi-branch Deep Separable Convolution)"""
    
    def __init__(self, input_channels, config):
        super().__init__()
        
        # 从配置中获取分支参数
        conv_channels = config.get('conv_channels', [64, 128, 256])
        kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        pool_sizes = config.get('pool_sizes', [1, 2, 2])
        fusion_channels = config.get('fusion_channels', 512)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        # 三个分支
        self.branch1 = MDSCBranch(input_channels, conv_channels[0], kernel_sizes[0], stride=pool_sizes[0], dropout_rate=dropout_rate)
        self.branch2 = MDSCBranch(input_channels, conv_channels[1], kernel_sizes[1], stride=pool_sizes[1], dropout_rate=dropout_rate)
        self.branch3 = MDSCBranch(input_channels, conv_channels[2], kernel_sizes[2], stride=pool_sizes[2], dropout_rate=dropout_rate)
        
        # 融合层: 1x1卷积将所有分支输出融合
        total_channels = sum(conv_channels)
        self.fusion = nn.Conv2d(total_channels, fusion_channels, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(fusion_channels)
        self.fusion_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 三个分支并行处理
        branch1_out = self.branch1(x)  # stride=1: 64x64
        branch2_out = self.branch2(x)  # stride=2: 32x32  
        branch3_out = self.branch3(x)  # stride=2: 32x32
        
        # 统一所有分支的尺寸到16x16（为了匹配位置编码）
        target_size = (16, 16)
        branch1_out = F.adaptive_avg_pool2d(branch1_out, target_size)
        branch2_out = F.adaptive_avg_pool2d(branch2_out, target_size)  
        branch3_out = F.adaptive_avg_pool2d(branch3_out, target_size)
        
        # 拼接所有分支
        fused = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        
        # 融合层
        fused = self.fusion(fused)
        fused = self.fusion_bn(fused)
        fused = self.fusion_relu(fused)
        
        return fused


class SubbandSE(nn.Module):
    """子带通道注意力（SE风格），在通道维上进行加权，强调判别性子带。"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class MultiScaleTFEncoder(nn.Module):
    """多尺度时频编码器（基于MDSC输出做多尺度融合 + 子带注意力）。
    - 从 MDSC 特征图出发，构建 16×16、8×8、4×4 三个尺度表示；
    - 将低尺度上采样至 16×16 后与原尺度拼接；
    - 通过 1×1 卷积降维回 d_model；
    - 子带SE注意力增强判别子带；
    - 返回 2D 特征图 (B, d_model, 16, 16)。
    """
    def __init__(self, mdsc_module: nn.Module, d_model: int = 512):
        super().__init__()
        self.mdsc = mdsc_module
        in_channels = d_model * 1  # mdsc融合通道=512
        fused_channels = d_model * 3  # 16×16, 8×8, 4×4 三尺度拼接
        self.reduce = nn.Conv2d(fused_channels, d_model, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(d_model)
        self.act = nn.ReLU(inplace=True)
        self.subband_se = SubbandSE(d_model, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础特征 (B, 512, 16, 16)
        f16 = self.mdsc(x)
        # 低尺度特征
        f8 = F.adaptive_avg_pool2d(f16, (8, 8))
        f4 = F.adaptive_avg_pool2d(f16, (4, 4))
        # 上采样至16×16
        f8u = F.interpolate(f8, size=(16, 16), mode='bilinear', align_corners=False)
        f4u = F.interpolate(f4, size=(16, 16), mode='bilinear', align_corners=False)
        # 多尺度融合 -> 降维
        fused = torch.cat([f16, f8u, f4u], dim=1)
        fused = self.reduce(fused)
        fused = self.bn(fused)
        fused = self.act(fused)
        # 子带注意力
        fused = self.subband_se(fused)
        return fused


class TripleAttentionMechanism(nn.Module):
    """三重注意力机制 (TAM): Self-Attention + Cross-Attention"""
    
    def __init__(self, d_model, nhead=4, dropout=0.3):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Self-Attention
        self.self_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Cross-Attention (跨频率注意力)
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (B, seq_len, d_model)
        
        # Self-Attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-Attention (使用自身作为key和value，模拟跨频率注意力)
        cross_out, _ = self.cross_attention(x, x, x)
        x = self.norm2(x + self.dropout(cross_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x


class PositionalEncoding2D(nn.Module):
    """二维正弦位置编码（适配时频图）"""
    
    def __init__(self, d_model, height=16, width=16):
        super().__init__()
        
        self.d_model = d_model
        pe = torch.zeros(d_model, height, width)
        
        # 创建位置网格
        y_position = torch.arange(0, height).unsqueeze(1).repeat(1, width).float()
        x_position = torch.arange(0, width).unsqueeze(0).repeat(height, 1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # 应用正弦和余弦编码
        for i in range(0, d_model, 4):
            if i < d_model:
                pe[i, :, :] = torch.sin(y_position * div_term[i//2])
            if i+1 < d_model:
                pe[i+1, :, :] = torch.cos(y_position * div_term[i//2])
            if i+2 < d_model:
                pe[i+2, :, :] = torch.sin(x_position * div_term[i//2])
            if i+3 < d_model:
                pe[i+3, :, :] = torch.cos(x_position * div_term[i//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, d_model, H, W)
        
    def forward(self, x):
        # x shape: (B, d_model, H, W)
        return x + self.pe


class EnhancedCNN(nn.Module):
    """增强的CNN特征提取器 - 保持向后兼容"""
    
    def __init__(self, input_channels, conv_channels, kernel_sizes, pool_sizes, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(conv_channels, kernel_sizes, pool_sizes)):
            # 卷积层
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ])
            
            # 池化层
            if pool_size > 1:
                layers.append(nn.MaxPool2d(pool_size, pool_size))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return x


class FluidIdentificationModel(nn.Module):
    """
    AWPD-MDSC-TAM 流体识别模型
    集成自适应小波包分解(AWPD)、多分支深度可分离卷积(MDSC)和三重注意力机制(TAM)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']  # 6个测井曲线
        num_classes = config.DATA_CONFIG['num_classes']

        # 啟用 cuDNN/TF32 以提升卷積與矩陣計算效率
        try:
            import torch
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        
        # MDSC: 多分支深度可分离卷积
        base_mdsc = MDSC(input_channels, config.CNN_CONFIG)
        use_mstf = bool(getattr(config, 'MODEL_CONFIG', {}).get('use_multiscale_tf', True))
        if use_mstf:
            self.backbone = MultiScaleTFEncoder(base_mdsc, d_model=config.TRANSFORMER_CONFIG['d_model'])
        else:
            self.backbone = base_mdsc
        
        # 二维位置编码（适配时频图）
        d_model = config.TRANSFORMER_CONFIG['d_model']
        self.pos_encoding_2d = PositionalEncoding2D(d_model, height=16, width=16)
        
        # TAM: 三重注意力机制
        self.tam = TripleAttentionMechanism(
            d_model=d_model,
            nhead=config.TRANSFORMER_CONFIG['nhead'],
            dropout=config.TRANSFORMER_CONFIG['dropout']
        )
        
        # 多层TAM堆叠
        self.tam_layers = nn.ModuleList([
            TripleAttentionMechanism(
                d_model=d_model,
                nhead=config.TRANSFORMER_CONFIG['nhead'],
                dropout=config.TRANSFORMER_CONFIG['dropout']
            ) for _ in range(config.TRANSFORMER_CONFIG['num_layers'])
        ])
        
        # 分类头 - 按照方案规格
        classifier_config = getattr(config, 'CLASSIFIER_CONFIG', {})
        fc1_out = classifier_config.get('fc1_out', 256)
        fc2_out = classifier_config.get('fc2_out', 128)
        dropout1 = classifier_config.get('dropout1', 0.3)
        dropout2 = classifier_config.get('dropout2', 0.0)
        activation1 = classifier_config.get('activation1', 'relu')
        activation2 = classifier_config.get('activation2', 'gelu')
        
        # 构建分类头
        layers = []
        
        # 第一层: Linear(512->256) + ReLU + Dropout(0.3)
        layers.extend([
            nn.Linear(d_model, fc1_out),
            nn.ReLU() if activation1 == 'relu' else nn.GELU(),
            nn.Dropout(dropout1)
        ])
        
        # 第二层: Linear(256->128) + GELU
        layers.extend([
            nn.Linear(fc1_out, fc2_out),
            nn.GELU() if activation2 == 'gelu' else nn.ReLU()
        ])
        
        if dropout2 > 0:
            layers.append(nn.Dropout(dropout2))
        
        # 输出层: Linear(128->6)
        layers.append(nn.Linear(fc2_out, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
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
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        AWPD-MDSC-TAM 前向传播
        Args:
            x: 输入张量 (B, C, H, W) - 批次大小, 通道数(6), 高度(64), 宽度(64)
        Returns:
            分类结果 (B, num_classes)
        """
        batch_size = x.size(0)
        
        # MDSC: 多分支深度可分离卷积特征提取
        # 输入: (B, 6, 64, 64) -> 输出: (B, 512, 16, 16)
        mdsc_features = self.backbone(x)
        
        # 添加二维位置编码
        mdsc_features = self.pos_encoding_2d(mdsc_features)
        
        # 将2D特征图展平为序列: (B, 512, 16, 16) -> (B, 256, 512)
        sequence_features = mdsc_features.flatten(2).transpose(1, 2)
        
        # TAM: 三重注意力机制处理序列
        for tam_layer in self.tam_layers:
            sequence_features = tam_layer(sequence_features)
        
        # 全局平均池化: (B, 256, 512) -> (B, 512)
        global_features = sequence_features.mean(dim=1)
        
        # 分类头: (B, 512) -> (B, 6)
        output = self.classifier(global_features)
        
        return output


class FluidIdentificationModelLegacy(nn.Module):
    """
    原始增强的流体识别模型 - 保持向后兼容
    结合CNN特征提取、Transformer序列建模和注意力机制
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']  # 6个测井曲线
        num_classes = config.DATA_CONFIG['num_classes']
        sequence_length = config.DATA_CONFIG['sequence_length']

        # 啟用 cuDNN/TF32 以提升卷積與矩陣計算效率
        try:
            import torch
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        
        # CNN特征提取器
        self.cnn = EnhancedCNN(
            input_channels=input_channels,
            conv_channels=config.CNN_CONFIG['conv_channels'],
            kernel_sizes=config.CNN_CONFIG['kernel_sizes'],
            pool_sizes=config.CNN_CONFIG['pool_sizes'],
            dropout_rate=config.CNN_CONFIG['dropout_rate']
        )
        
        # 固定下採樣到 16×16 並做 patch embedding，將序列長度從 4096 降至 256
        self.input_downsample = nn.AvgPool2d(kernel_size=4, stride=4)  # 64x64 -> 16x16
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=config.TRANSFORMER_CONFIG['d_model'],
            kernel_size=1,
            bias=False
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.TRANSFORMER_CONFIG['d_model'], max_len=5000)
        
        # Transformer编码器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                config.TRANSFORMER_CONFIG['d_model'],
                config.TRANSFORMER_CONFIG['nhead'],
                config.TRANSFORMER_CONFIG['dim_feedforward'],
                config.TRANSFORMER_CONFIG['dropout']
            ) for _ in range(config.TRANSFORMER_CONFIG['num_layers'])
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.TRANSFORMER_CONFIG['d_model'], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 增强分类器 - 提高学习能力
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # 减少dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),  # 进一步减少dropout
            nn.Linear(128, 64),
            nn.ReLU(),
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 (B, C, H, W) - 批次大小, 通道数(6), 高度(64), 宽度(64)
        Returns:
            分类结果 (B, num_classes)
        """
        batch_size = x.size(0)
        
        # 固定下採樣 + Patch Embedding，得到序列 (B, 256, d_model)
        x_ds = self.input_downsample(x)                 # (B, C, 16, 16)
        tokens_2d = self.patch_embed(x_ds)              # (B, d_model, 16, 16)
        sequence_features = tokens_2d.flatten(2).transpose(1, 2)  # (B, 256, d_model)
        
        # 添加位置编码
        if self.config.TRANSFORMER_CONFIG['use_positional_encoding']:
            sequence_features = self.pos_encoding(sequence_features)
        
        # Transformer编码
        for transformer_layer in self.transformer_layers:
            sequence_features = transformer_layer(sequence_features)
        
        # 全局平均池化
        sequence_features = sequence_features.mean(dim=1)  # (B, d_model)
        
        # 特征融合
        fused_features = self.feature_fusion(sequence_features)
        
        # 分类
        output = self.classifier(fused_features)
        
        return output


class LightweightFluidIdentificationModel(nn.Module):
    """
    轻量化流体识别模型 - 备选方案
    使用深度可分离卷积和注意力机制
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        input_channels = config.DATA_CONFIG['num_curves']
        num_classes = config.DATA_CONFIG['num_classes']
        
        # 轻量化特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层：深度可分离卷积
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels),
            nn.Conv2d(input_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二层：深度可分离卷积
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三层：深度可分离卷积
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 轻量化分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """更激进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 增加偏置初始值
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=2.0)  # 增加gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 增加偏置初始值
    
    def forward(self, x):
        """前向传播"""
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output


def create_enhanced_model(config):
    """创建增强模型"""
    return FluidIdentificationModel(config)


def create_lightweight_model(config):
    """创建轻量化模型"""
    return LightweightFluidIdentificationModel(config) 