#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
储层流体识别优化训练 - 适配W32_S1数据
使用optimized_one_to_one_data_W32_S1的完整8通道时频图谱 + 57维辅助特征
"""

import os
import sys
import time
import numpy as np
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from collections import Counter

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from models.lightweight_mdsc_tam import create_lightweight_model
    from models.focal_loss import FocalLoss
except ImportError as e:
    print(f"导入失败: {e}")
    print("尝试从当前目录导入...")

# ==================== 数据集类 ====================
class OptimizedDataset(Dataset):
    """优化数据集：加载8通道时频图谱 + 57维辅助特征"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 加载NPZ数据
        npz_file = self.data_dir / f'{split}.npz'
        if not npz_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {npz_file}")
        
        data = np.load(npz_file, allow_pickle=True)
        
        # 8通道时频图谱 (N, 8, 64, 64)
        self.spectrograms = torch.from_numpy(data['spectrograms']).float()
        
        # 57维辅助特征 (N, 57)
        self.aux_features = torch.from_numpy(data['aux_features']).float()
        
        # 标签 (N,)
        self.labels = torch.from_numpy(data['labels']).long()
        
        # 井名和深度索引（用于分析）
        self.wells = data['wells']
        self.depth_indices = data['depth_indices']
        
        # 训练集特有：增强标注
        if split == 'train':
            self.is_augmented = data['is_augmented']
            self.aug_types = data['aug_types']
            self.source_indices = data['source_indices']
        else:
            self.is_augmented = np.zeros(len(self.labels), dtype=bool)
        
        print(f"  {split}: 加载了 {len(self.labels)} 个样本")
        print(f"    时频图谱: {self.spectrograms.shape}")
        print(f"    辅助特征: {self.aux_features.shape}")
        
        # 统计类别分布
        label_counts = Counter(self.labels.numpy())
        fluid_names = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}
        print(f"    类别分布:")
        for label_id in sorted(label_counts.keys()):
            count = label_counts[label_id]
            pct = count / len(self.labels) * 100
            is_aug = np.sum(self.is_augmented[self.labels.numpy() == label_id]) if hasattr(self, 'is_augmented') else 0
            print(f"      {fluid_names[label_id]}: {count} ({pct:.1f}%)" + 
                  (f", 增强: {is_aug}" if is_aug > 0 else ""))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.spectrograms[idx],
            self.aux_features[idx],
            self.labels[idx]
        )

# ==================== 训练器类 ====================
class OptimizedTrainer:
    """优化训练器 - 适配新数据格式"""
    
    def __init__(self, data_dir='optimized_one_to_one_data_W32_S1', num_epochs=100):
        self.data_dir = Path(data_dir)
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 固定随机种子
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # ✅ 优化后的训练配置
        self.patience = 25  # 早停耐心
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_model_path = "best_model_w32_s1.pth"
        
        # 混合精度训练
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 日志
        self.log_dir = Path("logs") / "optimized_w32_s1"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"优化训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  最大轮次: {self.num_epochs}")
        print(f"  早停耐心: {self.patience}")
        print(f"  数据目录: {self.data_dir}")
    
    def create_model(self):
        """创建MDSC+TAM模型 - ✅ 适配新数据配置"""
        model_config = {
            'num_classes': 5,
            'input_channels': 8,  # ✅ 8通道时频图谱
            'use_multiscale': False,  # ✅ 不使用多尺度（无patch输入）
            'conv_channels': [64, 128, 256],  # ✅ 适中容量
            'fusion_channels': 512,  # ✅ 融合层
            'dropout_rate': 0.30,  # ✅ 适度dropout
            'aux_vec_dim': 57,  # ✅ 57维辅助特征（关键修正！）
            'wavelet_vec_dim': 0,  # ✅ 不使用wavelet_vec
            'stats_dim': 0  # ✅ 不使用stats
        }
        
        print(f"\n创建模型配置:")
        print(f"  输入通道: {model_config['input_channels']}")
        print(f"  辅助特征维度: {model_config['aux_vec_dim']}")
        print(f"  卷积通道: {model_config['conv_channels']}")
        print(f"  Dropout率: {model_config['dropout_rate']}")
        
        model = create_lightweight_model(model_config, model_config['input_channels'])
        model = model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        return model
    
    def create_data_loaders(self, batch_size=64):
        """创建数据加载器"""
        print("\n创建数据加载器...")
        
        # 创建数据集
        train_dataset = OptimizedDataset(self.data_dir, 'train')
        val_dataset = OptimizedDataset(self.data_dir, 'val')
        
        # 保存类别分布用于损失函数
        self.train_class_counts = Counter(train_dataset.labels.numpy())
        
        # 创建加载器（Windows兼容：num_workers=0）
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"  训练batches: {len(train_loader)}")
        print(f"  验证batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_optimizer_and_scheduler(self, model):
        """创建优化器和学习率调度器 - ✅ 优化版"""
        
        # ✅ 优化后的超参数
        initial_lr = 1.5e-3  # 初始学习率
        weight_decay = 1e-4  # 权重衰减
        warmup_epochs = 10   # Warmup轮数
        
        print(f"\n优化器配置:")
        print(f"  初始学习率: {initial_lr:.6f}")
        print(f"  Weight Decay: {weight_decay:.6f}")
        print(f"  Warmup轮数: {warmup_epochs}")
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Warmup + CosineAnnealing调度
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.num_epochs - warmup_epochs),
            eta_min=1e-6
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
        
        print(f"  调度策略: Warmup({warmup_epochs}轮) → CosineAnnealing")
        
        return optimizer, scheduler
    
    def create_loss_function(self):
        """创建损失函数 - ✅ 优化的Focal Loss + 类别权重"""
        
        # ✅ 优化的类别权重策略（基于数据分析）
        class_weights = torch.tensor([
            1.0,   # 油层 (31.1%) - 基准
            1.8,   # 水层 (7.9%) - 少数类，高权重
            0.9,   # 干层 (41.7%) - 最大类，略低
            1.3,   # 差油层 (14.0%) - 适度提高
            2.0    # 油水层 (5.3%) - 最少类，最高权重
        ], dtype=torch.float32, device=self.device)
        
        print(f"\n损失函数配置:")
        print(f"  类别权重:")
        fluid_names = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}
        for i, w in enumerate(class_weights):
            count = self.train_class_counts.get(i, 0)
            pct = count / sum(self.train_class_counts.values()) * 100 if self.train_class_counts else 0
            print(f"    {fluid_names[i]}: {w:.1f} ({count}样本, {pct:.1f}%)")
        
        # ✅ Focal Loss参数
        gamma = 1.0  # focusing参数（标准值）
        label_smoothing = 0.05  # 标签平滑
        
        print(f"  Focal Loss gamma: {gamma}")
        print(f"  标签平滑: {label_smoothing}")
        
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
        
        return criterion
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """训练一个轮次"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (spectrograms, aux_features, labels) in enumerate(train_loader):
            spectrograms = spectrograms.to(self.device)
            aux_features = aux_features.to(self.device)
            labels = labels.to(self.device)
            
            # 数值稳定化
            spectrograms = torch.nan_to_num(spectrograms, nan=0.0, posinf=0.0, neginf=0.0)
            spectrograms = torch.clamp(spectrograms, min=-8.0, max=8.0)
            
            # 混合精度训练
            if self.use_amp:
                with autocast('cuda'):
                    # 模型前向传播
                    out_tuple = model(
                        spectrograms, 
                        stats=None, 
                        patch_x=None, 
                        wavelet_vec=None, 
                        aux_vec=aux_features
                    )
                    
                    if isinstance(out_tuple, (list, tuple)) and len(out_tuple) >= 1:
                        outputs = out_tuple[0]
                    else:
                        outputs = out_tuple
                    
                    loss = criterion(outputs, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
            else:
                # 标准训练
                out_tuple = model(
                    spectrograms, 
                    stats=None, 
                    patch_x=None, 
                    wavelet_vec=None, 
                    aux_vec=aux_features
                )
                
                if isinstance(out_tuple, (list, tuple)) and len(out_tuple) >= 1:
                    outputs = out_tuple[0]
                else:
                    outputs = out_tuple
                
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 定期输出
            if batch_idx % 50 == 0 and batch_idx > 0:
                batch_acc = 100. * correct / total
                current_lr = optimizer.param_groups[0]['lr']
                print(f'   Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, LR: {current_lr:.6f}')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个轮次"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        y_true_all = []
        y_pred_all = []
        
        with torch.no_grad():
            for spectrograms, aux_features, labels in val_loader:
                spectrograms = spectrograms.to(self.device)
                aux_features = aux_features.to(self.device)
                labels = labels.to(self.device)
                
                # 数值稳定化
                spectrograms = torch.nan_to_num(spectrograms, nan=0.0, posinf=0.0, neginf=0.0)
                spectrograms = torch.clamp(spectrograms, min=-8.0, max=8.0)
                
                # 前向传播
                if self.use_amp:
                    with autocast('cuda'):
                        out_tuple = model(
                            spectrograms, 
                            stats=None, 
                            patch_x=None, 
                            wavelet_vec=None, 
                            aux_vec=aux_features
                        )
                        if isinstance(out_tuple, (list, tuple)) and len(out_tuple) >= 1:
                            outputs = out_tuple[0]
                        else:
                            outputs = out_tuple
                        loss = criterion(outputs, labels)
                else:
                    out_tuple = model(
                        spectrograms, 
                        stats=None, 
                        patch_x=None, 
                        wavelet_vec=None, 
                        aux_vec=aux_features
                    )
                    if isinstance(out_tuple, (list, tuple)) and len(out_tuple) >= 1:
                        outputs = out_tuple[0]
                    else:
                        outputs = out_tuple
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 收集用于F1计算
                y_true_all.append(labels.cpu())
                y_pred_all.append(predicted.cpu())
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        # 计算Macro-F1和各类F1
        macro_f1 = 0.0
        try:
            from sklearn.metrics import f1_score, classification_report
            y_true = torch.cat(y_true_all).numpy()
            y_pred = torch.cat(y_pred_all).numpy()
            
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
            
            print(f"   验证 Macro-F1: {macro_f1:.4f}")
            
            # 各类F1
            fluid_names = {0: '油层', 1: '水层', 2: '干层', 3: '差油层', 4: '油水层'}
            for cid in range(5):
                key = str(cid)
                if key in report:
                    f1c = report[key].get('f1-score', 0.0)
                    precision = report[key].get('precision', 0.0)
                    recall = report[key].get('recall', 0.0)
                    print(f"     {fluid_names[cid]}: F1={f1c:.4f}, P={precision:.4f}, R={recall:.4f}")
        except Exception as e:
            print(f"   计算F1失败: {e}")
            macro_f1 = 0.0
        
        return epoch_loss, epoch_acc, macro_f1
    
    def save_model(self, model, optimizer, epoch, val_acc, val_f1, is_best=False):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'config': {
                'num_classes': 5,
                'input_channels': 8,
                'aux_vec_dim': 57
            }
        }
        
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"  💾 保存最佳模型: {self.best_model_path}")
    
    def train(self):
        """主训练流程"""
        print("\n" + "="*80)
        print("开始训练 - 优化W32_S1数据")
        print("="*80)
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(batch_size=64)
        
        # 创建模型
        model = self.create_model()
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 创建损失函数
        criterion = self.create_loss_function()
        
        # 训练历史
        train_history = {'loss': [], 'acc': []}
        val_history = {'loss': [], 'acc': [], 'f1': []}
        
        patience_counter = 0
        
        # 训练循环
        print(f"\n开始训练 (共 {self.num_epochs} 轮)")
        print("="*80)
        
        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()
            
            # 训练
            print(f"\nEpoch {epoch}/{self.num_epochs} - 训练阶段")
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # 更新学习率
            scheduler.step()
            
            # 验证
            print(f"Epoch {epoch}/{self.num_epochs} - 验证阶段")
            val_loss, val_acc, val_f1 = self.validate_epoch(model, val_loader, criterion)
            
            # 记录历史
            train_history['loss'].append(train_loss)
            train_history['acc'].append(train_acc)
            val_history['loss'].append(val_loss)
            val_history['acc'].append(val_acc)
            val_history['f1'].append(val_f1)
            
            # 显示结果
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch} 结果:")
            print(f"   训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"   验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Macro-F1: {val_f1:.4f}")
            print(f"   学习率: {current_lr:.6f}")
            print(f"   耗时: {epoch_time:.1f}s")
            
            # 早停和保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                patience_counter = 0
                self.save_model(model, optimizer, epoch, val_acc, val_f1, is_best=True)
                print(f"  🎉 新的最佳 Macro-F1: {val_f1:.4f}")
            else:
                patience_counter += 1
                print(f"  ⏳ 等待改进: {patience_counter}/{self.patience}")
            
            # 早停检查
            if patience_counter >= self.patience:
                print(f"\n🛑 早停触发! 最佳 Macro-F1: {self.best_val_f1:.4f}, "
                      f"最佳验证准确率: {self.best_val_acc:.2f}%")
                break
        
        # 训练完成
        print("\n" + "="*80)
        print("🎉 训练完成!")
        print(f"   最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"   最佳 Macro-F1: {self.best_val_f1:.4f}")
        print(f"   最佳模型: {self.best_model_path}")
        
        # 保存训练历史
        history_file = self.log_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump({
                'train': train_history,
                'val': val_history,
                'best_val_acc': float(self.best_val_acc),
                'best_val_f1': float(self.best_val_f1)
            }, f, indent=2)
        
        print(f"   训练历史: {history_file}")
        print("="*80)

def main():
    """主函数"""
    print("🎯 储层流体识别优化训练 - W32_S1数据")
    print("📊 8通道时频图谱 + 57维辅助特征")
    print("="*80)
    
    try:
        # 创建训练器
        trainer = OptimizedTrainer(
            data_dir='optimized_one_to_one_data_W32_S1',
            num_epochs=100
        )
        
        # 开始训练
        trainer.train()
        
        print("\n📋 训练总结:")
        print("   ✅ 8通道时频图谱 (6曲线 + 2增强)")
        print("   ✅ 57维辅助特征 (48曲线 + 9全局)")
        print("   ✅ MDSC+TAM模型架构")
        print("   ✅ 优化Focal Loss + 类别权重")
        print("   ✅ Warmup + CosineAnnealing调度")
        print("   ✅ 早停防止过拟合")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

