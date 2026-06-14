#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
監督對比學習損失 (Supervised Contrastive Learning Loss)
用於增強特徵區分度，特別有助於少數類學習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    監督對比學習損失
    
    參考論文: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    
    核心思想：拉近同類樣本的特徵表示，推遠不同類樣本的特徵表示
    """
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        Args:
            temperature: 溫度參數，控制分布的平滑度
            contrast_mode: 對比模式 ('all' 或 'one')
            base_temperature: 基礎溫度參數
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels):
        """
        計算SupCon損失
        
        Args:
            features: 特徵張量 (batch_size, feature_dim) 或 (batch_size, n_views, feature_dim)
            labels: 標籤張量 (batch_size)
        
        Returns:
            loss: SupCon損失值
        """
        device = features.device
        
        # 處理特徵維度
        if len(features.shape) < 3:
            features = features.unsqueeze(1)  # (B, 1, D)
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # 確保labels是正確的形狀
        if labels.shape[0] != batch_size:
            raise ValueError(f'labels的batch_size ({labels.shape[0]}) '
                           f'與features的batch_size ({batch_size}) 不匹配')
        
        # 標準化特徵（L2歸一化）
        features = F.normalize(features, dim=2)  # (B, n_views, D)
        
        # 展平特徵: (B, n_views, D) -> (B*n_views, D)
        features = features.view(batch_size * n_views, -1)
        
        # 擴展labels: (B,) -> (B*n_views,)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('標籤數量錯誤')
        labels = labels.repeat(n_views, 1)
        
        # 計算相似度矩陣: (B*n_views, B*n_views)
        similarity_matrix = torch.matmul(features, features.T)
        
        # 創建mask：哪些樣本屬於同一類
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 移除對角線（自己與自己的相似度）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 計算對比損失
        # exp_logits = exp(sim / temp)
        exp_logits = torch.exp(similarity_matrix / self.temperature) * logits_mask
        log_prob = similarity_matrix / self.temperature - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 對每個錨點，計算正樣本對的平均log概率
        mask_sum = mask.sum(1)
        # 避免除以0
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # 計算損失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size)
        
        # 過濾NaN值
        valid_loss = loss[~torch.isnan(loss)]
        if len(valid_loss) > 0:
            loss = valid_loss.mean()
        else:
            loss = torch.tensor(0.0, device=loss.device)
        
        return loss


class HybridLoss(nn.Module):
    """
    混合損失函數：交叉熵 + SupCon + 少數類召回率懲罰
    
    這是方案A的核心損失函數
    """
    
    def __init__(self, 
                 class_weights=None, 
                 ce_weight=0.7, 
                 supcon_weight=0.2, 
                 minority_weight=0.1,
                 temperature=0.07,
                 minority_classes=[3, 4]):  # 差油層、油水層
        """
        Args:
            class_weights: 類別權重（用於交叉熵）
            ce_weight: 交叉熵損失權重
            supcon_weight: SupCon損失權重
            minority_weight: 少數類召回率懲罰權重
            temperature: SupCon溫度參數
            minority_classes: 少數類索引列表
        """
        super().__init__()
        
        self.ce_weight = ce_weight
        self.supcon_weight = supcon_weight
        self.minority_weight = minority_weight
        self.minority_classes = minority_classes
        
        # 交叉熵損失
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # SupCon損失
        self.supcon_loss = SupConLoss(temperature=temperature)
        
    def forward(self, logits, features, labels):
        """
        計算混合損失
        
        Args:
            logits: 模型輸出logits (B, num_classes)
            features: 特徵向量 (B, feature_dim) - 來自分類器前一層
            labels: 真實標籤 (B,)
        
        Returns:
            total_loss: 總損失
            loss_dict: 各組件損失的字典
        """
        # 1. 交叉熵損失
        ce_loss = self.ce_loss(logits, labels)
        
        # 2. SupCon損失
        supcon_loss = self.supcon_loss(features, labels)
        
        # 3. 少數類召回率懲罰
        # 計算預測
        preds = torch.argmax(logits, dim=1)
        
        # 對每個少數類，計算召回率
        minority_penalty = 0.0
        for cls in self.minority_classes:
            # 找到該類的真實樣本
            cls_mask = (labels == cls)
            if cls_mask.sum() > 0:
                # 計算該類的召回率
                cls_correct = ((preds == cls) & cls_mask).sum().float()
                cls_total = cls_mask.sum().float()
                cls_recall = cls_correct / cls_total
                
                # 懲罰：召回率越低，懲罰越高
                minority_penalty += (1.0 - cls_recall)
        
        # 平均懲罰
        if len(self.minority_classes) > 0:
            minority_penalty /= len(self.minority_classes)
        
        # 4. 總損失
        total_loss = (self.ce_weight * ce_loss + 
                     self.supcon_weight * supcon_loss + 
                     self.minority_weight * minority_penalty)
        
        # 返回損失字典以便監控
        loss_dict = {
            'total': total_loss.item(),
            'ce': ce_loss.item(),
            'supcon': supcon_loss.item(),
            'minority_penalty': minority_penalty if isinstance(minority_penalty, float) 
                               else minority_penalty.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # 測試代碼
    print("=" * 70)
    print("測試監督對比學習損失")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建測試數據
    batch_size = 8
    feature_dim = 128
    num_classes = 5
    
    features = torch.randn(batch_size, feature_dim).to(device)
    logits = torch.randn(batch_size, num_classes).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 測試SupConLoss
    print("\n1. 測試SupConLoss")
    supcon = SupConLoss(temperature=0.07).to(device)
    loss = supcon(features, labels)
    print(f"   特徵形狀: {features.shape}")
    print(f"   標籤形狀: {labels.shape}")
    print(f"   SupCon損失: {loss.item():.4f}")
    
    # 測試HybridLoss
    print("\n2. 測試HybridLoss")
    class_weights = torch.tensor([1.0, 3.0, 1.0, 5.0, 5.0]).to(device)
    hybrid_loss = HybridLoss(
        class_weights=class_weights,
        ce_weight=0.7,
        supcon_weight=0.2,
        minority_weight=0.1,
        minority_classes=[3, 4]
    ).to(device)
    
    total_loss, loss_dict = hybrid_loss(logits, features, labels)
    print(f"   總損失: {total_loss.item():.4f}")
    print(f"   損失組件:")
    for key, val in loss_dict.items():
        print(f"     - {key}: {val:.4f}")
    
    print("\n✅ 所有測試通過！")

