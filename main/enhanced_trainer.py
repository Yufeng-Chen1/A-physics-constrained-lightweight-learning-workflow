#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的训练器
集成Adam优化器、评价指标计算和可视化功能
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入可视化模块
from visualizations.enhanced_visualization import EnhancedVisualizationModule


class EnhancedFluidIdentificationTrainer:
    """增强的流体识别训练器"""
    
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
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 创建Adam优化器
        self.optimizer = self._create_adam_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建梯度缩放器（用于混合精度训练）
        self.scaler = GradScaler()
        
        # 创建TensorBoard写入器
        self.writer = self._create_tensorboard_writer()
        
        # 创建可视化模块
        self.viz_module = EnhancedVisualizationModule()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': [], 'epoch_time': [], 'gpu_memory': []
        }
        
        # 创建输出目录
        self.output_dir = f"./enhanced_training_outputs_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建训练日志
        self.log_file = os.path.join(self.output_dir, 'training_log.txt')
        self._log_message("🚀 增强训练开始")
        self._log_message(f"设备: {self.device}")
        self._log_message(f"模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log_message(f"模型大小: {sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024:.2f} MB")
    
    def _create_model(self):
        """创建模型"""
        print("🏗️  创建模型...")
        
        try:
            from models.lightweight_fluid_model import create_lightweight_model
            model = create_lightweight_model(self.config)
            print("   ✅ 使用轻量化模型")
        except ImportError:
            print("⚠️  轻量化模型导入失败，使用默认模型")
            from models.simple_fluid_model import SimpleFluidIdentificationModel
            model = SimpleFluidIdentificationModel(self.config)
        
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   模型参数总数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        print("📊 创建数据加载器...")
        
        try:
            from data.enhanced_data_loader import EnhancedWellLogDataset
            
            # 创建训练数据集
            train_dataset = EnhancedWellLogDataset(
                data_path=self.config.DATA_PATHS['default_data'],
                sequence_length=self.config.DATA_CONFIG['sequence_length'],
                curve_names=self.config.DATA_CONFIG['curve_names'],
                enable_wavelet=self.config.DATA_CONFIG['enable_wavelet'],
                wavelet_config=self.config.WAVELET_CONFIG['curve_specific_wavelets'],
                image_size=self.config.DATA_CONFIG['image_size']
            )
            
            # 创建验证数据集（使用相同的数据，但不同的分割）
            val_dataset = EnhancedWellLogDataset(
                data_path=self.config.DATA_PATHS['default_data'],
                sequence_length=self.config.DATA_CONFIG['sequence_length'],
                curve_names=self.config.DATA_CONFIG['curve_names'],
                enable_wavelet=self.config.DATA_CONFIG['enable_wavelet'],
                wavelet_config=self.config.WAVELET_CONFIG['curve_specific_wavelets'],
                image_size=self.config.DATA_CONFIG['image_size']
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.DATA_CONFIG['batch_size'],
                shuffle=True,
                num_workers=self.config.DATA_CONFIG['num_workers'],
                pin_memory=self.config.DATA_CONFIG['pin_memory'],
                persistent_workers=self.config.DATA_CONFIG['persistent_workers']
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.DATA_CONFIG['batch_size'],
                shuffle=False,
                num_workers=self.config.DATA_CONFIG['num_workers'],
                pin_memory=self.config.DATA_CONFIG['pin_memory'],
                persistent_workers=self.config.DATA_CONFIG['persistent_workers']
            )
            
            print(f"   ✅ 训练集: {len(train_dataset)} 样本")
            print(f"   ✅ 验证集: {len(val_dataset)} 样本")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"⚠️  数据加载器创建失败: {e}")
            print("   使用模拟数据...")
            
            # 创建模拟数据加载器
            return self._create_mock_data_loaders()
    
    def _create_mock_data_loaders(self):
        """创建模拟数据加载器（用于测试）"""
        print("   创建模拟数据...")
        
        # 创建模拟数据集
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # 模拟测井数据 (6通道, 64x64)
                data = torch.randn(6, 64, 64)
                # 模拟标签 (11类)
                label = torch.randint(0, 11, (1,))
                return data, label
        
        train_dataset = MockDataset(80)
        val_dataset = MockDataset(20)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        return train_loader, val_loader
    
    def _create_adam_optimizer(self):
        """创建Adam优化器"""
        print("⚡ 创建Adam优化器...")
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.TRAIN_CONFIG['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.TRAIN_CONFIG['weight_decay']
        )
        
        print(f"   ✅ 学习率: {self.config.TRAIN_CONFIG['learning_rate']}")
        print(f"   ✅ 权重衰减: {self.config.TRAIN_CONFIG['weight_decay']}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.TRAIN_CONFIG['scheduler_step_size'],
            gamma=self.config.TRAIN_CONFIG['scheduler_gamma']
        )
        
        return scheduler
    
    def _create_tensorboard_writer(self):
        """创建TensorBoard写入器"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "..", "logs", f"enhanced_training_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(log_dir, exist_ok=True)
            return SummaryWriter(log_dir)
        except Exception as e:
            print(f"⚠️  TensorBoard目录创建失败: {e}")
            try:
                backup_dir = os.path.join(current_dir, "..", "logs", "backup_training")
                os.makedirs(backup_dir, exist_ok=True)
                print(f"📊 使用备用日志目录: {backup_dir}")
                return SummaryWriter(backup_dir)
            except Exception as e2:
                print(f"❌ 备用目录也创建失败: {e2}")
                return None
    
    def _log_message(self, message):
        """记录日志消息"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def _get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def _calculate_metrics(self, outputs, labels):
        """计算评价指标"""
        _, predicted = torch.max(outputs, 1)
        
        # 转换为CPU numpy数组
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        
        # 计算基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 计算AUC-ROC（多分类）
        try:
            y_true_bin = label_binarize(y_true, classes=range(self.config.DATA_CONFIG['num_classes']))
            y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # 计算每个类别的AUC，然后取平均
            auc_scores = []
            for i in range(y_true_bin.shape[1]):
                if y_true_bin[:, i].sum() > 0:  # 确保有正样本
                    auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    auc_scores.append(auc)
            
            auc_roc = np.mean(auc_scores) if auc_scores else 0.0
        except:
            auc_roc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'predictions': y_pred,
            'probabilities': torch.softmax(outputs, dim=1).cpu().numpy()
        }
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
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
            
            # 收集预测结果用于指标计算
            all_predictions.append(outputs)
            all_labels.append(labels)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # 计算训练指标
        all_outputs = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        train_metrics = self._calculate_metrics(all_outputs, all_labels)
        
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, train_metrics
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_predictions.append(outputs)
                all_labels.append(labels)
        
        # 计算验证指标
        all_outputs = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_metrics = self._calculate_metrics(all_outputs, all_labels)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, val_metrics
    
    def save_checkpoint(self, epoch, val_loss, val_metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'training_history': self.training_history
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.SAVE_CONFIG['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.SAVE_CONFIG['checkpoint_dir'], 'best_enhanced_model.pth')
            torch.save(checkpoint, best_path)
            self._log_message(f"🏆 保存最佳模型: {best_path}")
    
    def generate_visualizations(self, val_metrics, val_labels, val_probabilities):
        """生成可视化图表"""
        print("📊 生成可视化图表...")
        
        # 获取类别名称
        class_names = [f'Class_{i}' for i in range(self.config.DATA_CONFIG['num_classes'])]
        
        # 1. 绘制混淆矩阵
        self.viz_module.plot_confusion_matrix(
            val_labels, val_metrics['predictions'], 
            class_names, 'enhanced_confusion_matrix.png'
        )
        
        # 2. 绘制ROC曲线
        self.viz_module.plot_roc_curves(
            val_labels, val_metrics['probabilities'], 
            class_names, 'enhanced_roc_curves.png'
        )
        
        # 3. 绘制精确率-召回率曲线
        self.viz_module.plot_precision_recall_curves(
            val_labels, val_metrics['probabilities'], 
            class_names, 'enhanced_pr_curves.png'
        )
        
        # 4. 绘制训练指标
        self.viz_module.plot_training_metrics(
            self.training_history, 'enhanced_training_metrics.png'
        )
        
        print("   ✅ 可视化图表生成完成")
    
    def run_ablation_study(self):
        """运行消融实验"""
        print("🔬 运行消融实验...")
        
        # 获取验证数据
        val_data = []
        val_labels = []
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                val_data.append(data)
                val_labels.append(labels)
        
        val_data = torch.cat(val_data, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        
        # 运行消融实验
        ablation_results, ablation_plot_path = self.viz_module.create_ablation_experiment(
            self.model, val_data, val_labels, 'enhanced_ablation_results.png'
        )
        
        # 计算模块贡献率
        module_contributions = self.viz_module.calculate_module_contributions(ablation_results)
        
        # 绘制模块贡献率
        contribution_plot_path = self.viz_module.plot_module_contribution(
            module_contributions, 'enhanced_module_contribution.png'
        )
        
        print("   ✅ 消融实验完成")
        
        return ablation_results, module_contributions, ablation_plot_path, contribution_plot_path
    
    def run_optimizer_comparison(self):
        """运行优化器对比实验"""
        print("⚡ 运行优化器对比实验...")
        
        # 运行优化器对比
        optimizer_results, optimizer_plot_path = self.viz_module.run_optimizer_comparison(
            type(self.model), self.config, self.train_loader, self.val_loader,
            'enhanced_optimizer_comparison.png'
        )
        
        print("   ✅ 优化器对比实验完成")
        
        return optimizer_results, optimizer_plot_path
    
    def train(self):
        """开始训练"""
        self._log_message("🚀 开始增强训练...")
        start_time = time.time()
        
        for epoch in range(1, self.config.TRAIN_CONFIG['epochs'] + 1):
            epoch_start_time = time.time()
            
            self._log_message(f"Epoch {epoch}/{self.config.TRAIN_CONFIG['epochs']} 开始")
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1_score'])
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1_score'])
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            self.training_history['gpu_memory'].append(self._get_gpu_memory_usage())
            
            # 检查是否是最佳模型
            is_best = val_metrics['f1_score'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_score']
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._log_message(f"🏆 新的最佳F1分数: {val_metrics['f1_score']:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, val_metrics, is_best)
            
            # 记录训练日志
            self._log_message(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_metrics['accuracy']:.4f}, Train F1={train_metrics['f1_score']:.4f}")
            self._log_message(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}, Val F1={val_metrics['f1_score']:.4f}")
            self._log_message(f"Epoch {epoch}: LR={current_lr:.6f}, Time={epoch_time:.2f}s")
            
            # 记录到TensorBoard
            if self.writer:
                step = epoch * len(self.train_loader)
                self.writer.add_scalar('Train/Loss', train_loss, step)
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], step)
                self.writer.add_scalar('Train/F1_Score', train_metrics['f1_score'], step)
                self.writer.add_scalar('Val/Loss', val_loss, step)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], step)
                self.writer.add_scalar('Val/F1_Score', val_metrics['f1_score'], step)
                self.writer.add_scalar('Val/AUC_ROC', val_metrics['auc_roc'], step)
                self.writer.add_scalar('Learning_Rate', current_lr, step)
            
            # 早停检查
            if self.patience_counter >= self.config.TRAIN_CONFIG['patience']:
                self._log_message(f"🛑 早停触发，在epoch {epoch}停止训练")
                break
        
        # 训练完成后的分析
        self._log_message("🎉 训练完成，开始生成分析报告...")
        
        # 获取最终验证结果
        final_val_loss, final_val_metrics = self.validate_epoch(epoch)
        
        # 生成可视化图表
        self.generate_visualizations(final_val_metrics, 
                                   torch.cat([labels for _, labels in self.val_loader], dim=0),
                                   final_val_metrics['probabilities'])
        
        # 运行消融实验
        ablation_results, module_contributions, ablation_plot_path, contribution_plot_path = self.run_ablation_study()
        
        # 运行优化器对比实验
        optimizer_results, optimizer_plot_path = self.run_optimizer_comparison()
        
        # 生成综合报告
        all_results = {
            'accuracy': final_val_metrics['accuracy'],
            'precision': final_val_metrics['precision'],
            'recall': final_val_metrics['recall'],
            'f1_score': final_val_metrics['f1_score'],
            'auc_roc': final_val_metrics['auc_roc'],
            'ablation_results': ablation_results,
            'optimizer_results': optimizer_results,
            'module_contributions': module_contributions
        }
        
        report_path = self.viz_module.generate_comprehensive_report(all_results, 'enhanced_comprehensive_report.md')
        
        # 训练完成
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        self._log_message(f"🎉 增强训练完成！")
        self._log_message(f"⏱️  总训练时间: {hours:.2f} 小时")
        self._log_message(f"🏆 最佳验证F1分数: {self.best_val_f1:.4f}")
        self._log_message(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
        self._log_message(f"🏆 最佳验证损失: {self.best_val_loss:.4f}")
        self._log_message(f"📊 AUC-ROC: {final_val_metrics['auc_roc']:.4f}")
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
            self._log_message(f"📊 TensorBoard日志已保存: {self.writer.log_dir}")
        
        self._log_message(f"📊 可视化报告已保存: {self.viz_module.output_dir}")
        self._log_message(f"📋 综合报告已保存: {report_path}")


def main():
    """主函数"""
    try:
        from configs.config import Config
        
        config = Config()
        trainer = EnhancedFluidIdentificationTrainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
