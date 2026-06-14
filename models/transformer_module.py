import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置編碼"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        return x + self.pe[:x.size(0), :]


class LightweightTransformerModule(nn.Module):
    """
    輕量級Transformer模組 - 優化版本
    使用更少的層數和更小的維度減少參數量
    """
    
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.1, 
                 max_seq_length: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # 輕量級Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 輕量級輸入投影層 - 動態適應輸入維度
        self.input_projection = None  # 將在forward中動態創建
        
        # 輕量級輸出投影層
        self.output_projection = nn.Linear(d_model, d_model // 2)
        
        # 輕量級分類頭
        self.classifier = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 6)  # 6個流體類型
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向傳播
        Args:
            x: 輸入張量 (B, L, D) - B批次大小, L序列長度, D特徵維度
            mask: 注意力掩碼
        Returns:
            Transformer輸出
        """
        # 檢查輸入維度
        if len(x.shape) == 4:  # (B, C, H, W)
            # 如果是4D張量，重塑為2D
            B, C, H, W = x.shape
            x = x.view(B, C, H * W)  # (B, C, H*W)
            x = x.transpose(1, 2)  # (B, H*W, C)
        
        # 動態創建輸入投影層
        if self.input_projection is None or self.input_projection.in_features != x.shape[-1]:
            self.input_projection = nn.Linear(x.shape[-1], self.d_model).to(x.device)
        
        # 輸入投影
        x = self.input_projection(x)
        
        # 位置編碼
        x = x.transpose(0, 1)  # (L, B, D)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, L, D)
        
        # Transformer編碼
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (B, D)
        
        # 輸出投影
        x = self.output_projection(x)
        
        # 分類
        output = self.classifier(x)
        
        return output


class TransformerModule(nn.Module):
    """
    Transformer模組
    用於處理測井曲線的序列特徵和全局依賴關係
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1, 
                 max_seq_length: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 輸入投影層
        self.input_projection = nn.Linear(256, d_model)  # 假設CNN輸出256維
        
        # 輸出投影層
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)  # 6個流體類型
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向傳播
        Args:
            x: 輸入張量 (B, L, D) - B批次大小, L序列長度, D特徵維度
            mask: 注意力掩碼
        Returns:
            Transformer輸出
        """
        # 輸入投影
        x = self.input_projection(x)
        
        # 位置編碼
        x = x.transpose(0, 1)  # (L, B, D)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, L, D)
        
        # Transformer編碼
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)  # (B, D)
        
        # 輸出投影
        x = self.output_projection(x)
        
        # 分類
        output = self.classifier(x)
        
        return output


class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """前向傳播"""
        batch_size = query.size(0)
        
        # 線性變換
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # 注意力計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 應用注意力權重
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    """前饋網絡"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向傳播"""
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer塊"""
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向傳播"""
        # 自注意力
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前饋網絡
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CustomTransformerModule(nn.Module):
    """
    自定義Transformer模組
    使用自定義的Transformer塊
    """
    
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 1024, dropout: float = 0.1, 
                 max_seq_length: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer塊
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 輸入投影
        self.input_projection = nn.Linear(256, d_model)
        
        # 輸出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """前向傳播"""
        # 輸入投影
        x = self.input_projection(x)
        
        # 位置編碼
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer塊
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # 全局平均池化
        x = torch.mean(x, dim=1)
        
        # 輸出投影
        x = self.output_projection(x)
        
        # 分類
        output = self.classifier(x)
        
        return output 