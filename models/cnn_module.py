import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightCNNModule(nn.Module):
    """
    輕量級CNN模組 - 優化版本
    使用深度可分離卷積和注意力機制減少參數量
    """
    
    def __init__(self, input_channels: int = 6, conv_channels: list = None, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [16, 32, 64]  # 減少通道數
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.dropout_rate = dropout_rate
        
        # 構建輕量級卷積層
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        in_channels = input_channels
        for out_channels in conv_channels:
            # 使用深度可分離卷積減少參數量
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1)
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            self.dropouts.append(nn.Dropout2d(dropout_rate))
            self.attention_layers.append(ChannelAttention(out_channels))
            in_channels = out_channels
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        Args:
            x: 輸入張量 (B, C, H, W)
        Returns:
            特徵張量 (B, feature_dim)
        """
        for i, (conv, bn, dropout, attention) in enumerate(zip(
            self.conv_layers, self.batch_norms, self.dropouts, self.attention_layers)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            x = attention(x)  # 應用通道注意力
            
            # 池化操作
            if i < len(self.conv_channels) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class CNNModule(nn.Module):
    """
    卷積神經網絡模組
    用於提取測井曲線的空間特徵
    """
    
    def __init__(self, input_channels: int = 6, conv_channels: list = None, 
                 kernel_sizes: list = None, pool_sizes: list = None, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2, 2]
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.pool_sizes = pool_sizes
        self.dropout_rate = dropout_rate
        
        # 構建卷積層
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_channels = input_channels
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                         padding=kernel_size//2)
            )
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            self.dropouts.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特徵維度計算
        self.feature_dim = conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        Args:
            x: 輸入張量 (B, C, H, W)
        Returns:
            CNN特徵張量
        """
        for i, (conv, bn, dropout) in enumerate(zip(self.conv_layers, self.batch_norms, self.dropouts)):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # 池化操作
            if i < len(self.pool_sizes):
                x = F.max_pool2d(x, kernel_size=self.pool_sizes[i], stride=self.pool_sizes[i])
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        return x


class ResidualCNNModule(nn.Module):
    """
    殘差卷積神經網絡模組
    使用殘差連接提高特徵提取能力
    """
    
    def __init__(self, input_channels: int = 6, base_channels: int = 32, 
                 num_blocks: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        
        # 初始卷積層
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 殘差塊
        self.residual_blocks = nn.ModuleList()
        in_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            self.residual_blocks.append(
                ResidualBlock(in_channels, out_channels, stride=2 if i > 0 else 1)
            )
            in_channels = out_channels
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = in_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        x = self.initial_conv(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class ResidualBlock(nn.Module):
    """殘差塊"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下採樣層（如果需要）
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class AttentionCNNModule(nn.Module):
    """
    注意力卷積神經網絡模組
    使用通道注意力和空間注意力機制
    """
    
    def __init__(self, input_channels: int = 6, conv_channels: list = None, 
                 dropout_rate: float = 0.3):
        super().__init__()
        
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.dropout_rate = dropout_rate
        
        # 構建帶注意力的卷積層
        self.conv_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        in_channels = input_channels
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.attention_layers.append(ChannelAttention(out_channels))
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
            self.dropouts.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = conv_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        for i, (conv, attention, bn, dropout) in enumerate(
            zip(self.conv_layers, self.attention_layers, self.batch_norms, self.dropouts)
        ):
            x = conv(x)
            x = attention(x)  # 應用通道注意力
            x = bn(x)
            x = F.relu(x)
            x = dropout(x)
            
            # 池化操作
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        return x


class ChannelAttention(nn.Module):
    """通道注意力機制"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        return x * attention 