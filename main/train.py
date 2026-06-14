#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强GPU训练脚本
使用批量井数据加载器，处理welldata目录内的所有井数据
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.simple_fluid_model import SimpleFluidIdentificationModel
from data.batch_data_loader import create_batch_data_loaders_with_config


class EnhancedGPUTrainer:
    """增强GPU训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建数据加载器
        self.train_loader, self.val_loader = self._create_batch_data_loaders()
        
        # 创建优化器和学习率调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建梯度缩放器（用于混合精度训练）
        self.scaler = GradScaler()
        
        # 创建TensorBoard写入器
        self.writer = self._create_tensorboard_writer()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [], 'epoch_time': []
        }
        
        # 创建可视化目录
        self.viz_dir = f"./visualizations/training_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def _create_model(self):
        """创建模型"""
        print("🏗️  创建模型...")
        
        model = SimpleFluidIdentificationModel(self.config)
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   模型参数总数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        return model
    
    def _create_batch_data_loaders(self):
        """创建批量数据加载器"""
        print("📊 创建批量数据加载器...")
        
        try:
            train_loader, val_loader = create_batch_data_loaders_with_config(
                config=self.config,
                welldata_dir=self.config.DATA_PATHS['welldata_dir'],
                batch_size=self.config.DATA_CONFIG['batch_size'],
                test_mode=False  # 正式训练模式
            )
            
            if train_loader is None or val_loader is None:
                raise ValueError("数据加载器创建失败")
            
            print(f"✅ 批量数据加载器创建成功")
            print(f"   训练集: {len(train_loader.dataset)} 样本")
            print(f"   验证集: {len(val_loader.dataset)} 样本")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ 创建批量数据加载器失败: {e}")
            raise
    
    def _create_optimizer(self):
        """创建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.TRAIN_CONFIG['learning_rate'],
            weight_decay=self.config.TRAIN_CONFIG['weight_decay']
        )
        
    def _create_scheduler(self):
        """创建学习率调度器"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
    
    def _create_tensorboard_writer(self):
        """创建TensorBoard写入器"""
        try:
            # 使用更安全的路径处理
            log_dir = f"./logs/batch_training_{time.strftime('%Y%m%d_%H%M%S')}"
            
            # 确保父目录存在
            os.makedirs("./logs", exist_ok=True)
            
            # 创建日志目录
            os.makedirs(log_dir, exist_ok=True)
            
            # 验证目录是否创建成功
            if not os.path.isdir(log_dir):
                raise RuntimeError(f"无法创建日志目录: {log_dir}")
                
            return SummaryWriter(log_dir)
        except Exception as e:
            print(f"⚠️  TensorBoard 日志目录创建失败: {e}")
            # 使用临时目录作为备选方案
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="tensorboard_logs_")
            print(f"   使用临时目录: {temp_dir}")
            return SummaryWriter(temp_dir)
    
    def _calculate_f1_score(self, outputs, labels):
        """计算F1分数"""
        try:
            from sklearn.metrics import f1_score
            _, predicted = outputs.max(1)
            f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
            return f1
        except Exception as e:
            print(f"⚠️  计算F1分数失败: {e}")
            return 0.0
    
    def _get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3  # GB
            return 0.0
        except:
            return 0.0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.TRAIN_CONFIG["epochs"]}')
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data = data.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算F1分数
            f1_score = self._calculate_f1_score(outputs, labels)
            total_f1 += f1_score
            
            # 更新进度条
            acc = 100. * correct / total
            avg_f1 = total_f1 / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%',
                'F1': f'{f1_score:.4f}'
            })
            
            # 记录到TensorBoard
            if self.writer:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', acc, step)
                self.writer.add_scalar('Train/F1_Score', f1_score, step)
                self.writer.add_scalar('Train/GPU_Memory', self._get_gpu_memory_usage(), step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        avg_f1 = total_f1 / len(self.train_loader)
        
        return avg_loss, avg_acc, avg_f1
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                # 检查批次是否为空
                if data.size(0) == 0 or labels.size(0) == 0:
                    print("⚠️  跳过空批次")
                    continue
                    
                data = data.to(self.device)
                
                # 安全地处理标签维度
                if labels.dim() > 1:
                    labels = labels.squeeze()
                labels = labels.to(self.device)
                
                # 确保标签不为空且维度正确
                if labels.numel() == 0:
                    print("⚠️  跳过空标签批次")
                    continue
                
                # 确保标签是1D张量
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 计算F1分数
                f1_score = self._calculate_f1_score(outputs, labels)
                total_f1 += f1_score
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        avg_f1 = total_f1 / len(self.val_loader)
        
        # 记录到TensorBoard
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', avg_acc, epoch)
            self.writer.add_scalar('Val/F1_Score', avg_f1, epoch)
        
        return avg_loss, avg_acc, avg_f1
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
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
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['train_loss'], label='训练损失', color='blue')
        plt.plot(self.training_history['val_loss'], label='验证损失', color='red')
        plt.title('训练和验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['train_acc'], label='训练准确率', color='blue')
        plt.plot(self.training_history['val_acc'], label='验证准确率', color='red')
        plt.title('训练和验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config.SAVE_CONFIG['checkpoint_dir'], 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练曲线已保存: {save_path}")
    
    def train(self):
        """开始训练"""
        print("🚀 开始批量井数据训练...")
        start_time = time.time()
        
        for epoch in range(1, self.config.TRAIN_CONFIG['epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.TRAIN_CONFIG['epochs']}")
            print(f"{'='*60}")
            
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_time'].append(epoch_time)
            
            # 打印结果
            print(f"Epoch {epoch:3d}/{self.config.TRAIN_CONFIG['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                print(f"🏆 新的最佳验证损失: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 生成实时可视化
            self._generate_realtime_visualizations(epoch)
            
            # 早停检查
            if self.patience_counter >= self.config.TRAIN_CONFIG['patience']:
                print(f"🛑 早停触发，在epoch {epoch}停止训练")
                break
        
        # 训练完成
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        print(f"\n🎉 批量训练完成！")
        print(f"⏱️  总训练时间: {hours:.2f} 小时")
        print(f"🏆 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 生成最终可视化报告
        self._generate_final_visualization_report()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
            print(f"📊 TensorBoard日志已保存: {self.writer.log_dir}")
        
        print(f"📊 可视化报告已保存: {self.viz_dir}")
    
    def _generate_realtime_visualizations(self, epoch):
        """生成实时可视化"""
        try:
            # 学习率变化图
            plt.figure(figsize=(8, 6))
            plt.plot(self.training_history['learning_rate'])
            plt.title('学习率变化曲线')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(os.path.join(self.viz_dir, 'learning_rate_curve.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 训练时间分析
            if len(self.training_history['epoch_time']) > 1:
                plt.figure(figsize=(8, 6))
                plt.plot(self.training_history['epoch_time'])
                plt.title('每个Epoch训练时间')
                plt.xlabel('Epoch')
                plt.ylabel('Time (seconds)')
                plt.grid(True)
                plt.savefig(os.path.join(self.viz_dir, 'epoch_time_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"⚠️  生成实时可视化失败: {e}")
    
    def _generate_final_visualization_report(self):
        """生成最终可视化报告"""
        try:
            # 创建综合可视化报告
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('训练结果综合分析报告', fontsize=16, fontweight='bold')
            
            # 1. 损失曲线
            axes[0, 0].plot(self.training_history['train_loss'], label='训练损失', color='blue', linewidth=2)
            axes[0, 0].plot(self.training_history['val_loss'], label='验证损失', color='red', linewidth=2)
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 2. 准确率曲线
            axes[0, 1].plot(self.training_history['train_acc'], label='训练准确率', color='blue', linewidth=2)
            axes[0, 1].plot(self.training_history['val_acc'], label='验证准确率', color='red', linewidth=2)
            axes[0, 1].set_title('准确率曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 3. 学习率变化
            axes[0, 2].plot(self.training_history['learning_rate'], color='green', linewidth=2)
            axes[0, 2].set_title('学习率变化')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].grid(True)
            
            # 4. 训练时间分析
            axes[1, 0].plot(self.training_history['epoch_time'], color='orange', linewidth=2)
            axes[1, 0].set_title('每个Epoch训练时间')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True)
            
            # 5. 训练效率分析（准确率/时间）
            if len(self.training_history['epoch_time']) > 0:
                efficiency = [acc/time if time > 0 else 0 for acc, time in zip(self.training_history['train_acc'], self.training_history['epoch_time'])]
                axes[1, 1].plot(efficiency, color='purple', linewidth=2)
                axes[1, 1].set_title('训练效率 (准确率/时间)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Efficiency')
                axes[1, 1].grid(True)
            
            # 6. 损失与准确率关系
            axes[1, 2].scatter(self.training_history['train_loss'], self.training_history['train_acc'], 
                               alpha=0.6, color='blue', label='训练')
            axes[1, 2].scatter(self.training_history['val_loss'], self.training_history['val_acc'], 
                               alpha=0.6, color='red', label='验证')
            axes[1, 2].set_title('损失与准确率关系')
            axes[1, 2].set_xlabel('Loss')
            axes[1, 2].set_ylabel('Accuracy (%)')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'comprehensive_training_report.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成训练统计报告
            self._generate_training_statistics_report()
            
        except Exception as e:
            print(f"⚠️  生成最终可视化报告失败: {e}")
    
    def _generate_training_statistics_report(self):
        """生成训练统计报告"""
        try:
            report_path = os.path.join(self.viz_dir, 'training_statistics_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("训练统计报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"训练配置:\n")
                f.write(f"  总轮数: {self.config.TRAIN_CONFIG['epochs']}\n")
                f.write(f"  学习率: {self.config.TRAIN_CONFIG['learning_rate']}\n")
                f.write(f"  批次大小: {self.config.DATA_CONFIG['batch_size']}\n")
                f.write(f"  序列长度: {self.config.DATA_CONFIG['sequence_length']}\n")
                f.write(f"  图像尺寸: {self.config.DATA_CONFIG['image_size']}\n\n")
                
                f.write(f"数据集信息:\n")
                f.write(f"  训练集大小: {len(self.train_loader.dataset)}\n")
                f.write(f"  验证集大小: {len(self.val_loader.dataset)}\n")
                f.write(f"  类别数量: {self.config.DATA_CONFIG['num_classes']}\n\n")
                
                f.write(f"训练结果:\n")
                f.write(f"  最佳验证损失: {self.best_val_loss:.4f}\n")
                f.write(f"  最佳验证准确率: {self.best_val_acc:.2f}%\n")
                f.write(f"  总训练时间: {(sum(self.training_history['epoch_time'])/3600):.2f} 小时\n")
                f.write(f"  平均每轮时间: {np.mean(self.training_history['epoch_time']):.2f} 秒\n")
                f.write(f"  最快轮次: {np.min(self.training_history['epoch_time']):.2f} 秒\n")
                f.write(f"  最慢轮次: {np.max(self.training_history['epoch_time']):.2f} 秒\n\n")
                
                f.write(f"模型信息:\n")
                f.write(f"  模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}\n")
                f.write(f"  可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")
                
            print(f"📊 训练统计报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️  生成训练统计报告失败: {e}")


def main():
    """主函数"""
    print("🚀 批量井数据训练系统")
    print("=" * 60)
    
    # 加载配置
    config = Config()
    
    # 打印配置信息
    print("📋 当前配置:")
    print(f"   数据目录: {config.DATA_PATHS['welldata_dir']}")
    print(f"   测井曲线: {config.DATA_CONFIG['curve_names']}")
    print(f"   序列长度: {config.DATA_CONFIG['sequence_length']}")
    print(f"   批次大小: {config.DATA_CONFIG['batch_size']}")
    print(f"   图像尺寸: {config.DATA_CONFIG['image_size']}")
    print(f"   启用小波: {config.DATA_CONFIG['enable_wavelet']}")
    print(f"   启用清洗: {config.DATA_CONFIG['enable_cleaning']}")
    print(f"   训练轮数: {config.TRAIN_CONFIG['epochs']}")
    print(f"   学习率: {config.TRAIN_CONFIG['learning_rate']}")
    
    # 创建训练器
    trainer = EnhancedGPUTrainer(config)
    
    # 开始训练
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 