#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Focal Loss实现
专门解决类别不平衡和难分类样本问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    论文: "Focal Loss for Dense Object Detection"
    适用于解决类别不平衡和难分类样本问题
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing: float = 0.0):
        """
        初始化Focal Loss
        
        Args:
            alpha: 类别权重，可以是float（二分类）或tensor（多分类）
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 损失聚合方式，'mean', 'sum' 或 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
    
    def forward(self, inputs, targets):
        """
        计算Focal Loss
        
        Args:
            inputs: 模型预测logits，形状 (N, C)
            targets: 真实标签，形状 (N,)
        
        Returns:
            focal_loss: Focal Loss值
        """
        # 计算交叉熵损失
        # cross entropy with optional label smoothing
        try:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        except TypeError:
            # 兼容旧版本PyTorch: 手动实现平滑
            if self.label_smoothing > 0:
                num_classes = inputs.size(1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(inputs)
                    true_dist.fill_(self.label_smoothing / (num_classes - 1))
                    true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
                log_probs = F.log_softmax(inputs, dim=1)
                ce_loss = -(true_dist * log_probs).sum(dim=1)
            else:
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 应用alpha权重（如果提供）
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t * ce_loss
        
        # 计算Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss
    根据类别难度动态调整gamma参数
    """
    
    def __init__(self, alpha=None, gamma_range=(1.0, 3.0), reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_min, self.gamma_max = gamma_range
        self.reduction = reduction
        self.class_accuracy = None
    
    def update_class_accuracy(self, class_acc_dict):
        """更新类别准确率，用于自适应gamma"""
        self.class_accuracy = class_acc_dict
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 自适应gamma：准确率越低，gamma越大
        if self.class_accuracy is not None:
            gamma_tensor = torch.ones_like(targets, dtype=torch.float)
            for class_id, acc in self.class_accuracy.items():
                mask = targets == class_id
                # 准确率越低，gamma越大（更关注难样本）
                gamma_val = self.gamma_max - (acc / 100.0) * (self.gamma_max - self.gamma_min)
                gamma_tensor[mask] = gamma_val
        else:
            gamma_tensor = torch.full_like(targets, 2.0, dtype=torch.float)
        
        # 应用alpha权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, targets)
            ce_loss = alpha_t * ce_loss
        
        # 计算自适应Focal Loss
        focal_loss = (1 - pt) ** gamma_tensor * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_class_weighted_focal_loss(class_counts, gamma=2.0, device='cuda'):
    """
    创建类别加权的Focal Loss
    
    Args:
        class_counts: 各类别样本数量字典
        gamma: Focal Loss聚焦参数
        device: 设备
    
    Returns:
        focal_loss: 配置好的Focal Loss函数
    """
    # 计算类别权重（反比例）
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # 使用平滑的反频率权重
    alpha_weights = []
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 1)
        # 平滑权重：避免权重过大
        weight = (total_samples / (num_classes * count)) ** 0.5
        alpha_weights.append(weight)
    
    # 归一化权重
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float, device=device)
    alpha_tensor = alpha_tensor / alpha_tensor.sum() * num_classes
    
    print(f"🏷️ Focal Loss类别权重: {alpha_tensor.cpu().numpy()}")
    
    return FocalLoss(alpha=alpha_tensor, gamma=gamma)

# 储层流体识别专用配置
def create_fluid_focal_loss(class_distribution, device='cuda'):
    """
    为储层流体识别创建专门的Focal Loss
    
    Args:
        class_distribution: 类别分布字典
        device: 设备
    
    Returns:
        focal_loss: 配置好的Focal Loss
    """
    # 储层流体识别的特殊权重策略
    class_names = {0: "油层", 1: "水层", 2: "干层", 3: "差油层", 4: "油水层"}
    
    # 为储层特征设计权重：关键流体类型权重更高
    special_weights = {
        0: 1.2,  # 油层：重要，但样本较多
        1: 1.8,  # 水层：重要，样本中等
        2: 0.8,  # 干层：普通，样本最多
        3: 2.2,  # 差油层：关键，样本较少，难区分
        4: 3.2   # 油水层：最关键，样本最少，最难区分（强化）
    }
    
    # 结合频率权重和专业权重
    alpha_weights = []
    total_samples = sum(class_distribution.values())
    
    for class_id in range(5):
        count = class_distribution.get(class_id, 1)
        freq_weight = total_samples / (5 * count)
        special_weight = special_weights[class_id]
        # 组合权重
        combined_weight = (freq_weight ** 0.3) * (special_weight ** 0.7)
        alpha_weights.append(combined_weight)
    
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float, device=device)
    
    # 打印权重信息
    print(f"🧪 储层流体Focal Loss权重:")
    for i, (name, weight) in enumerate(zip(class_names.values(), alpha_tensor)):
        print(f"   {name}: {weight:.3f}")

    # 使用更高的gamma来更关注难样本（2.5~3.0）
    gamma_value = 2.8
    print(f"   gamma: {gamma_value:.2f}")
    return FocalLoss(alpha=alpha_tensor, gamma=gamma_value)
