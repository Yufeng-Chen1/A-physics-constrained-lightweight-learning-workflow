#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的流体识别训练器 - 修复验证集性能问题版本
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ImprovedFluidIdentificationTrainer:
    """优化的流体识别训练器 - 修复验证集性能问题"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_batch_data_loaders()
        
        # 获取类别信息
        self.num_classes = self._get_num_classes()
        self.class_weights = self._calculate_class_weights()
        
        # 创建模型
        self.model = self._create_enhanced_model()
        
        # 创建优化器和学习率调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 创建损失函数
        self.criterion = self._create_loss_function()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': [], 'epoch_time': []
        }
        
        # 创建保存目录
        self._create_directories()
        
        print(f"✅ 训练器初始化完成")
        print(f"   模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   类别数量: {self.num_classes}")
        print(f"   类别权重: {self.class_weights.cpu().numpy()}")
    
    def _create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config.SAVE_CONFIG['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config.SAVE_CONFIG['log_dir'], exist_ok=True)
    
    def _create_batch_data_loaders(self):
        """创建批量数据加载器"""
        print("📊 创建批量数据加载器...")
        
        try:
            from data.batch_data_loader import create_batch_data_loaders
            
            data_loaders = create_batch_data_loaders(
                welldata_dir=self.config.DATA_PATHS['welldata_dir'],
                curve_names=self.config.DATA_CONFIG['curve_names'],
                sequence_length=self.config.DATA_CONFIG['sequence_length'],
                batch_size=self.config.DATA_CONFIG['batch_size'],
                train_ratio=self.config.DATA_CONFIG['train_ratio'],
                val_ratio=self.config.DATA_CONFIG['val_ratio'],
                test_ratio=self.config.DATA_CONFIG['test_ratio'],
                enable_wavelet=self.config.DATA_CONFIG['enable_wavelet'],
                image_size=self.config.DATA_CONFIG['image_size'],
                enable_cleaning=self.config.CLEANING_CONFIG['enable_cleaning'],
                num_workers=self.config.DATA_CONFIG['num_workers'],
                test_mode=False
            )
            
            train_loader = data_loaders['train']
            val_loader = data_loaders['val']
            
            print(f"✅ 批量数据加载器创建成功")
            print(f"   训练集: {len(train_loader.dataset)} 样本")
            print(f"   验证集: {len(val_loader.dataset)} 样本")
            print(f"   训练集批次数: {len(train_loader)}")
            print(f"   验证集批次数: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ 创建批量数据加载器失败: {e}")
            raise
    
    def _get_num_classes(self):
        """获取类别数量"""
        try:
            # 从数据集中获取类别数
            if hasattr(self.train_loader.dataset, 'labels_encoded'):
                labels = self.train_loader.dataset.labels_encoded
                return len(np.unique(labels))
            else:
                # 手动统计
                all_labels = []
                for _, labels in self.train_loader:
                    all_labels.extend(labels.tolist())
                return len(np.unique(all_labels))
        except:
            return 4  # 默认值
    
    def _calculate_class_weights(self):
        """计算类别权重"""
        try:
            all_labels = []
            for _, labels in self.train_loader:
                all_labels.extend(labels.tolist())
            
            labels_array = np.array(all_labels)
            counts = np.bincount(labels_array, minlength=self.num_classes)
            counts[counts == 0] = 1.0  # 避免除零
            
            # 计算反频率权重
            inv_freq = 1.0 / counts
            inv_freq = inv_freq / inv_freq.sum() * self.num_classes
            
            weights = torch.tensor(inv_freq, dtype=torch.float32, device=self.device)
            print(f"📊 类别分布: {counts}")
            print(f"📊 类别权重: {weights.cpu().numpy()}")
            
            return weights
        except Exception as e:
            print(f"⚠️ 计算类别权重失败: {e}")
            return torch.ones(self.num_classes, device=self.device)
    
    def _create_enhanced_model(self):
        """创建增强模型"""
        try:
            from models.fluid_identification_model import FluidIdentificationModel
            
            # 创建模型配置
            model_config = type('ModelConfig', (), {
                'DATA_CONFIG': {
                    'num_curves': self.config.DATA_CONFIG['num_curves'],
                    'num_classes': self.num_classes,
                    'sequence_length': self.config.DATA_CONFIG['sequence_length']
                },
                'TRANSFORMER_CONFIG': self.config.TRANSFORMER_CONFIG
            })()
            
            model = FluidIdentificationModel(model_config)
            
            model = model.to(self.device)
            return model
            
        except Exception as e:
            print(f"❌ 创建模型失败: {e}")
            raise
    
    def _create_optimizer(self):
        """创建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.TRAIN_CONFIG['learning_rate'],
            weight_decay=self.config.TRAIN_CONFIG['weight_decay'],
            betas=(self.config.TRAIN_CONFIG['adam_beta1'], self.config.TRAIN_CONFIG['adam_beta2']),
            eps=self.config.TRAIN_CONFIG['adam_eps']
        )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.TRAIN_CONFIG.get('scheduler_type', 'cosine_annealing')
        
        if scheduler_type == 'cosine_annealing':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.TRAIN_CONFIG['scheduler_t_max'],
                eta_min=self.config.TRAIN_CONFIG['scheduler_eta_min']
            )
        else:
            # 默认使用ReduceLROnPlateau
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                min_lr=1e-6,
                verbose=True
            )
    
    def _create_loss_function(self):
        """创建损失函数"""
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.config.TRAIN_CONFIG.get('label_smoothing', 0.0)
            )
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.TRAIN_CONFIG.get('label_smoothing', 0.0)
            )
        return criterion
    
    def _calculate_f1_score(self, outputs, labels):
        """计算F1分数"""
        try:
            from sklearn.metrics import f1_score
            _, predicted = outputs.max(1)
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)
            return f1
        except ImportError:
            print("⚠️  sklearn未安装，无法计算F1分数")
            return 0.0
    
    def _apply_data_augmentation(self, data, labels):
        """应用数据增强"""
        batch_size = data.size(0)
        if batch_size < 2:
            return data, labels
        
        # MixUp增强
        if np.random.rand() < self.config.TRAIN_CONFIG.get('mixup_alpha', 0.0):
            lam = np.random.beta(0.4, 0.4)
            index = torch.randperm(batch_size, device=self.device)
            mixed_data = lam * data + (1 - lam) * data[index, :]
            return mixed_data, labels, labels[index], lam
        
        return data, labels, None, None
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.TRAIN_CONFIG["epochs"]}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            try:
                data = data.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # 数据验证
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print(f"⚠️  检测到输入数据包含NaN/Inf，跳过此批次")
                    continue
                
                # 数据增强
                data, labels, labels_b, lam = self._apply_data_augmentation(data, labels)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(data)
                
                # 计算损失
                if labels_b is not None and lam is not None:
                    # MixUp损失
                    loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)
                
                # 数值检查
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1000:
                    print(f"⚠️  检测到异常损失值: {loss.item()}, 跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TRAIN_CONFIG['max_grad_norm'])
                
                # 优化器步进
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 计算F1分数
                f1_score = self._calculate_f1_score(outputs, labels)
                total_f1 += f1_score
                num_batches += 1
                
                # 更新进度条
                acc = 100. * correct / total
                avg_f1 = total_f1 / num_batches
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.2f}%',
                    'F1': f'{avg_f1:.4f}'
                })
                
            except Exception as e:
                print(f"⚠️  训练批次 {batch_idx} 出错: {e}")
                continue
        
        # 计算平均值
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        avg_f1 = total_f1 / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_acc, avg_f1
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                try:
                    data = data.to(self.device)
                    labels = labels.squeeze().to(self.device)
                    
                    # 数据验证
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        continue
                    
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                    
                    # 数值检查
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    # 计算F1分数
                    f1_score = self._calculate_f1_score(outputs, labels)
                    total_f1 += f1_score
                    num_batches += 1
                    
                except Exception as e:
                    continue
        
        # 计算平均值
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_acc = 100. * correct / total if total > 0 else 0
        avg_f1 = total_f1 / num_batches if num_batches > 0 else 0
        
        return avg_loss, avg_acc, avg_f1
    
    def save_checkpoint(self, epoch, val_loss, val_acc, val_f1, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.SAVE_CONFIG['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.SAVE_CONFIG['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"🏆 保存最佳模型: {best_path}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始优化训练...")
        start_time = time.time()
        
        for epoch in range(1, self.config.TRAIN_CONFIG['epochs'] + 1):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch}/{self.config.TRAIN_CONFIG['epochs']} 开始")
            
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate_epoch(epoch)
            
            # 更新学习率调度器
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['train_f1'].append(train_f1)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            
            # 检查是否是最佳模型
            is_best = False
            monitor = self.config.TRAIN_CONFIG.get('monitor', 'val_loss')
            min_delta = self.config.TRAIN_CONFIG.get('min_delta', 0.0001)
            
            if monitor == 'val_loss':
                improved = (self.best_val_loss == float('inf')) or (val_loss < self.best_val_loss - min_delta)
                if improved:
                    self.best_val_loss = val_loss
                    self.best_val_acc = val_acc
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    is_best = True
                    print(f"🏆 新的最佳验证损失: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
            else:
                improved = (val_f1 > self.best_val_f1 + min_delta)
                if improved:
                    self.best_val_f1 = val_f1
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    is_best = True
                    print(f"🏆 新的最佳F1分数: {val_f1:.4f}")
                else:
                    self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, val_acc, val_f1, is_best)
            
            # 记录训练日志
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.4f}")
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")
            print(f"Epoch {epoch}: LR={current_lr:.6f}, Time={epoch_time:.2f}s")
            print(f"Epoch {epoch}: Patience={self.patience_counter}/{self.config.TRAIN_CONFIG['patience']}")
            
            # 早停检查
            if self.patience_counter >= self.config.TRAIN_CONFIG['patience']:
                print(f"🛑 早停触发，在epoch {epoch}停止训练")
                print(f"📊 耐心计数器: {self.patience_counter}/{self.config.TRAIN_CONFIG['patience']}")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        print(f"\n🎉 优化训练完成！")
        print(f"⏱️  总训练时间: {hours:.2f} 小时")
        print(f"🏆 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"🏆 最佳验证F1分数: {self.best_val_f1:.4f}")
        
        # 生成可视化图表
        self._create_visualizations()
    
    def _create_visualizations(self):
        """创建可视化图表"""
        try:
            plt.figure(figsize=(15, 10))
            
            # 损失曲线
            plt.subplot(2, 3, 1)
            plt.plot(self.training_history['train_loss'], label='Train Loss', color='blue')
            plt.plot(self.training_history['val_loss'], label='Val Loss', color='red')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 准确率曲线
            plt.subplot(2, 3, 2)
            plt.plot(self.training_history['train_acc'], label='Train Acc', color='blue')
            plt.plot(self.training_history['val_acc'], label='Val Acc', color='red')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            # F1分数曲线
            plt.subplot(2, 3, 3)
            plt.plot(self.training_history['train_f1'], label='Train F1', color='blue')
            plt.plot(self.training_history['val_f1'], label='Val F1', color='red')
            plt.title('Training and Validation F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.grid(True)
            
            # 学习率曲线
            plt.subplot(2, 3, 4)
            plt.plot(self.training_history['learning_rate'], color='green')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            # 训练时间曲线
            plt.subplot(2, 3, 5)
            plt.plot(self.training_history['epoch_time'], color='orange')
            plt.title('Epoch Training Time')
            plt.xlabel('Epoch')
            plt.ylabel('Time (s)')
            plt.grid(True)
            
            # 验证集性能对比
            plt.subplot(2, 3, 6)
            epochs = range(1, len(self.training_history['val_acc']) + 1)
            plt.plot(epochs, self.training_history['val_acc'], label='Accuracy', color='blue', marker='o')
            plt.plot(epochs, [x * 100 for x in self.training_history['val_f1']], label='F1 Score (×100)', color='red', marker='s')
            plt.title('Validation Performance')
            plt.xlabel('Epoch')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(self.config.SAVE_CONFIG['log_dir'], 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  可视化图表生成失败: {e}")


def main():
    """主函数"""
    try:
        from configs.improved_config import ImprovedConfig
        
        config = ImprovedConfig()
        trainer = ImprovedFluidIdentificationTrainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
