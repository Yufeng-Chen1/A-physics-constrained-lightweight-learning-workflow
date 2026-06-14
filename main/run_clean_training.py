#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清洁训练程序 - 基于 run_ultimate_awpd_training.py 修改
==========================================
修改内容：
1. ✅ 适配 well_split_dataset_jiyuan 数据集（.npz格式）
2. ✅ 输入：7通道声谱图 + 44维辅助特征
3. ✅ 清晰的配置管理（无外部config依赖）
4. ✅ 显示每个流体类别的F1分数
==========================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import io

# UTF-8输出
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
except:
    pass

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.focal_loss import FocalLoss

# ==================== 清洁配置类 ====================
class CleanTrainingConfig:
    """清洁训练配置 - 所有参数一目了然"""
    
    # ⭐ 数据路径
    DATA_DIR = Path('well_split_dataset_jiyuan')
    
    # ⭐ 训练超参数（参考方案M++的成功配置）
    BATCH_SIZE = 64
    NUM_EPOCHS = 700
    LEARNING_RATE = 1.2e-3       # 明确的学习率
    WEIGHT_DECAY = 4.0e-5
    DROPOUT = 0.010              # 降低dropout，释放学习能力
    LABEL_SMOOTHING = 0.005
    EARLY_STOPPING_PATIENCE = 200
    WARMUP_EPOCHS = 40           # Warmup期
    WARMUP_START_FACTOR = 0.3    # 从30%开始
    
    # ⭐ 早停配置
    WARMUP_NO_EARLY_STOP = 60    # 前60轮不触发早停
    EARLY_STOP_METRIC = 'acc'    # 使用准确率作为早停指标（比Macro-F1稳定）
    
    # ⭐ 类别权重（温和的权重）
    CLASS_WEIGHTS = {
        0: 1.0,   # 油层
        1: 1.5,   # 水层
        2: 1.0,   # 干层
        3: 1.5,   # 差油层
        4: 2.0,   # 油水层
    }
    
    # ⭐ Focal Loss参数
    FOCAL_GAMMA = 1.5
    
    CLASS_NAMES = ['油层', '水层', '干层', '差油层', '油水层']
    NUM_CLASSES = 5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    LOG_DIR = Path('logs/clean_training')
    CHECKPOINT_DIR = Path('checkpoints/clean_training')

# ==================== well_split_dataset_jiyuan 数据集 ====================
class WellSplitDataset(Dataset):
    """加载 well_split_dataset_jiyuan 的 .npz 格式数据"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 加载 .npz 文件
        data_file = self.data_dir / f'{split}.npz'
        print(f"  加载数据文件: {data_file}")
        
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_file}")
        
        data = np.load(data_file, allow_pickle=True)
        
        # 提取数据
        self.spectrograms = data['spectrograms']    # (N, 7, 64, 64)
        self.aux_features = data['aux_features']    # (N, 44)
        self.labels = data['labels']                # (N,)
        
        print(f"  {split}集样本数: {len(self.labels)}")
        print(f"  声谱图形状: {self.spectrograms.shape}")
        print(f"  辅助特征形状: {self.aux_features.shape}")
        
        # 显示类别分布
        if split == 'train':
            from collections import Counter
            label_counts = Counter(self.labels)
            print(f"  训练集类别分布:")
            for label_id, count in sorted(label_counts.items()):
                fluid_name = ['油层', '水层', '干层', '差油层', '油水层'][label_id]
                percentage = count / len(self.labels) * 100
                print(f"    {fluid_name}: {count} 样本 ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spec = torch.from_numpy(self.spectrograms[idx]).float()  # (7, 64, 64)
        aux = torch.from_numpy(self.aux_features[idx]).float()   # (44,)
        label = int(self.labels[idx])
        
        return spec, aux, label

# ==================== MDSC+TAM模型 ====================
class MDSC_TAM_Model(nn.Module):
    """
    MDSC+TAM模型（适配7通道声谱图 + 44维辅助特征）
    - MDSC: 多尺度深度可分离卷积
    - TAM: 跨模态注意力机制
    """
    
    def __init__(self, num_classes=5, dropout=0.010):
        super().__init__()
        
        # MDSC分支：多尺度特征提取（7通道输入）
        self.mdsc_backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(7, 64, kernel_size=3, padding=1),  # ⭐ 7通道输入
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.3),
            
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.4),
            
            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.4),
            
            # Stage 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 辅助特征分支（44维输入）
        self.aux_fc = nn.Sequential(
            nn.Linear(44, 256),  # ⭐ 44维输入
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        
        # TAM：跨模态注意力机制
        self.cross_attention = nn.Sequential(
            nn.Linear(512 + 256, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768),  # 512+256
            nn.Sigmoid()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + 256, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, spec, aux_feat):
        B = spec.size(0)
        
        # MDSC特征提取
        mdsc_feat = self.mdsc_backbone(spec)
        mdsc_feat = mdsc_feat.view(B, -1)  # (B, 512)
        
        # 辅助特征处理
        aux_feat_processed = self.aux_fc(aux_feat)  # (B, 256)
        
        # TAM：跨模态注意力融合
        combined = torch.cat([mdsc_feat, aux_feat_processed], dim=1)  # (B, 768)
        attention_weight = self.cross_attention(combined)  # (B, 768)
        
        # 加权融合
        fused_feat = combined * attention_weight
        
        # 分类
        logits = self.classifier(fused_feat)
        
        return logits

# ==================== 清洁训练器 ====================
class CleanTrainer:
    """清洁训练器 - 简洁、清晰、可靠"""
    
    def __init__(self, config: CleanTrainingConfig):
        self.config = config
        self.device = config.DEVICE
        
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("🚀 清洁训练程序 - MDSC+TAM模型")
        print("="*80)
        print(f"✅ 数据集: {config.DATA_DIR}")
        print(f"✅ 批次大小: {config.BATCH_SIZE}")
        print(f"✅ 学习率: {config.LEARNING_RATE}")
        print(f"✅ Dropout: {config.DROPOUT}")
        print(f"✅ 权重衰减: {config.WEIGHT_DECAY}")
        print(f"✅ Warmup轮数: {config.WARMUP_EPOCHS} (起始{config.WARMUP_START_FACTOR*100:.0f}%)")
        print(f"✅ Warmup保护期: {config.WARMUP_NO_EARLY_STOP}轮（不触发早停）")
        print(f"✅ 早停指标: {config.EARLY_STOP_METRIC}")
        print(f"✅ 早停patience: {config.EARLY_STOPPING_PATIENCE}")
        print("="*80)
        
        # 加载数据
        print("\n📂 加载数据...")
        train_dataset = WellSplitDataset(config.DATA_DIR, split='train')
        val_dataset = WellSplitDataset(config.DATA_DIR, split='val')
        test_dataset = WellSplitDataset(config.DATA_DIR, split='test')
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0,  # Windows兼容
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )
        
        # 计算类别权重
        self.class_weights = self._compute_class_weights(train_dataset)
        
        print("\n🏗️  创建MDSC+TAM模型...")
        self.model = MDSC_TAM_Model(
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  模型参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        # Focal Loss
        self.criterion = FocalLoss(
            alpha=self.class_weights,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.86, 0.999)
        )
        
        # ⭐ 验证学习率
        actual_lr = self.optimizer.param_groups[0]['lr']
        print(f"\n✅ 优化器创建完成:")
        print(f"   设置的学习率: {config.LEARNING_RATE:.6f}")
        print(f"   实际的学习率: {actual_lr:.6f}")
        assert abs(actual_lr - config.LEARNING_RATE) < 1e-9, "❌ 学习率不匹配！"
        
        # 学习率调度器（三阶段）
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=config.WARMUP_START_FACTOR,
            total_iters=config.WARMUP_EPOCHS
        )
        self.plateau_scheduler = torch.optim.lr_scheduler.ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=80  # Plateau期
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS - 80,
            eta_min=8.0e-5
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.plateau_scheduler, self.cosine_scheduler],
            milestones=[config.WARMUP_EPOCHS, config.WARMUP_EPOCHS + 80]
        )
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.history = []
    
    def _compute_class_weights(self, dataset):
        print("\n📊 计算类别权重...")
        
        class_counts = {i: 0 for i in range(self.config.NUM_CLASSES)}
        for label in dataset.labels:
            class_counts[int(label)] += 1
        
        total = len(dataset)
        
        print("训练集类别权重:")
        for i in range(self.config.NUM_CLASSES):
            count = class_counts[i]
            percentage = count / total * 100
            weight = self.config.CLASS_WEIGHTS[i]
            print(f"  {self.config.CLASS_NAMES[i]}: {count} ({percentage:.2f}%) - 权重: {weight}")
        
        weights = torch.tensor([self.config.CLASS_WEIGHTS[i] for i in range(self.config.NUM_CLASSES)], 
                               dtype=torch.float32).to(self.device)
        return weights
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]', ncols=100)
        
        for batch_idx, (spec, aux_feat, target) in enumerate(pbar):
            spec = spec.to(self.device)
            aux_feat = aux_feat.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(spec, aux_feat)
            loss = self.criterion(output, target)
            
            if torch.isnan(loss):
                print(f"\n⚠️  警告：第{batch_idx}个batch出现NaN loss，跳过...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.2)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, loader, desc='Valid', show_report=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'[{desc}]', ncols=100)
            for spec, aux_feat, target in pbar:
                spec = spec.to(self.device)
                aux_feat = aux_feat.to(self.device)
                target = target.to(self.device)
                
                output = self.model(spec, aux_feat)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                acc = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
                pbar.set_postfix({
                    'loss': f'{total_loss/(len(all_preds)/self.config.BATCH_SIZE):.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        accuracy = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # ⭐ 计算每个类别的F1分数
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        
        if show_report:
            print("\n分类报告:")
            print(classification_report(
                all_targets, 
                all_preds, 
                target_names=self.config.CLASS_NAMES,
                zero_division=0,
                digits=4
            ))
        
        return total_loss / len(loader), accuracy * 100, macro_f1, per_class_f1
    
    def train(self):
        print("\n🎯 开始训练...")
        print("="*80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, per_class_f1 = self.validate(self.val_loader, desc='Valid')
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
                'per_class_f1': per_class_f1.tolist(),
                'lr': current_lr
            }
            self.history.append(epoch_info)
            
            print("-"*80)
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Macro-F1={val_f1:.4f}")
            print(f"  学习率: {current_lr:.6f}")
            
            # ⭐ Warmup保护期逻辑
            if epoch + 1 <= self.config.WARMUP_NO_EARLY_STOP:
                # Warmup期：只记录最佳
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_f1 = val_f1
                    self._save_best_model(epoch, val_acc, val_f1, per_class_f1)
                    print(f"  [WARMUP-BEST] Epoch {epoch+1}/{self.config.WARMUP_NO_EARLY_STOP}")
                else:
                    print(f"  [WARMUP] Epoch {epoch+1}/{self.config.WARMUP_NO_EARLY_STOP}")
            else:
                # 正式训练期：使用准确率做早停
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    self._save_best_model(epoch, val_acc, val_f1, per_class_f1)
                else:
                    self.patience_counter += 1
                    print(f"  [WAIT] 等待改进: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
                    
                    if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print(f"\n🛑 早停触发！最佳验证准确率: {self.best_val_acc:.2f}%")
                        break
            
            print("-"*80)
            
            # 保存历史记录
            with open(self.config.CHECKPOINT_DIR / 'training_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        print("\n✅ 训练完成！")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"🏆 最佳验证F1: {self.best_val_f1:.4f}")
        
        # 测试集评估
        print("\n" + "="*80)
        print("📊 测试集评估（使用最佳模型）")
        print("="*80)
        self.model.load_state_dict(torch.load(self.config.CHECKPOINT_DIR / 'best_model.pth')['model_state_dict'])
        test_loss, test_acc, test_f1, test_per_class_f1 = self.validate(self.test_loader, desc='Test', show_report=True)
        print(f"\n🎯 测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  Macro F1: {test_f1:.4f}")
        print(f"\n  各类别F1分数:")
        for i, (name, f1) in enumerate(zip(self.config.CLASS_NAMES, test_per_class_f1)):
            print(f"    {name}: {f1:.4f}")
        print("="*80)
    
    def _save_best_model(self, epoch, val_acc, val_f1, per_class_f1):
        """保存最佳模型并显示各类别F1"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'per_class_f1': per_class_f1.tolist(),
            'history': self.history
        }
        torch.save(checkpoint, self.config.CHECKPOINT_DIR / 'best_model.pth')
        
        # ⭐ 显示各类别F1分数
        print(f"  ✅ [BEST] 新的最佳模型！准确率: {val_acc:.2f}%, Macro-F1: {val_f1:.4f}")
        print(f"      各流体F1分数:")
        for i, (name, f1) in enumerate(zip(self.config.CLASS_NAMES, per_class_f1)):
            print(f"        {name}: {f1:.4f}")

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 清洁训练程序启动")
    print("="*80)
    print("📋 特性:")
    print("  ✅ 数据集: well_split_dataset_jiyuan (7870训练 + 2013验证 + 883测试)")
    print("  ✅ 输入: 7通道声谱图(64×64) + 44维辅助特征")
    print("  ✅ 模型: MDSC+TAM (多尺度深度可分离卷积 + 跨模态注意力)")
    print("  ✅ 配置管理: 清晰的硬编码配置（无外部依赖）")
    print("  ✅ 学习率策略: Warmup(40轮) + Plateau(80轮) + CosineAnnealing")
    print("  ✅ 早停策略: 验证准确率 + 60轮Warmup保护期")
    print("  ✅ 显示各流体F1: 保存最佳模型时显示5个类别的F1分数")
    print("="*80 + "\n")
    
    config = CleanTrainingConfig()
    trainer = CleanTrainer(config)
    trainer.train()
    
    print("\n" + "="*80)
    print("🎉 训练流程全部完成！")
    print("="*80)


