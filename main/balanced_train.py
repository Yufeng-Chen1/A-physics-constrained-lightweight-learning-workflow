#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用类别平衡方法的训练脚本
集成BSMOTE和其他平衡技术来提升模型训练精度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.spectrogram_data_loader import create_spectrogram_data_loaders
from data.class_balance_handler import create_balanced_data_loader, ClassBalanceHandler
from models.fluid_identification_model import FluidIdentificationModel
from configs.optimized_config import OptimizedConfig
from visualization.training_analysis import IntegratedTrainingAnalyzer

class BalancedTrainer:
    """使用类别平衡的训练器"""
    
    def __init__(self, config, balance_method='bsmote', target_ratio=0.5):
        """
        初始化平衡训练器
        
        Args:
            config: 配置对象
            balance_method: 平衡方法 ('bsmote', 'smote', 'class_weight', 'none')
            target_ratio: 目标平衡比例
        """
        self.config = config
        self.balance_method = balance_method
        self.target_ratio = target_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = None
        
        # 训练日志
        self.training_logs = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        print(f"🚀 初始化平衡训练器")
        print(f"   设备: {self.device}")
        print(f"   平衡方法: {balance_method.upper()}")
        print(f"   目标比例: {target_ratio}")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        print(f"\n📊 创建数据加载器...")
        
        # 19口井的分割策略
        custom_split = {
            'train': ['di199-48', 'di200-47', 'geng120', 'geng166', 'geng181', 
                     'geng203', 'geng207', 'geng217', 'geng219', 'geng220', 
                     'geng221', 'ji117', 'ji121', 'luo160'],
            'val': ['geng343', 'geng60', 'jian39'],
            'test': ['geng86', 'yuan3']
        }
        
        # 创建原始数据加载器
        data_loaders = create_spectrogram_data_loaders(
            welldata_dir=self.config.DATA_PATHS['welldata_dir'],  # 修正为DATA_PATHS
            batch_size=self.config.DATA_CONFIG['batch_size'],
            enable_cleaning=True,
            test_mode=False,
            use_fluid_mapping=True,
            custom_split=custom_split
        )
        
        if self.balance_method == 'none':
            return data_loaders
        
        # 应用BSMOTE + 类别权重平衡
        print("\n🔄 应用BSMOTE + 类别权重平衡...")
        balanced_data_loaders = create_balanced_data_loader(
            data_loaders, 
            method=self.balance_method,
            target_ratio=self.target_ratio
        )
        
        if 'class_weights' in balanced_data_loaders:
            self.class_weights = balanced_data_loaders['class_weights']
        
        return balanced_data_loaders
    
    def create_model(self):
        """创建模型"""
        print(f"\n🏗️  创建模型...")
        
        model = FluidIdentificationModel(self.config)
        model = model.to(self.device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建完成")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        return model
    
    def create_optimizer_and_scheduler(self, model):
        """创建优化器和学习率调度器"""
        print(f"\n⚙️  创建优化器和调度器...")
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.TRAIN_CONFIG['learning_rate'],
            weight_decay=self.config.TRAIN_CONFIG['weight_decay']
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.TRAIN_CONFIG['epochs'],
            eta_min=self.config.TRAIN_CONFIG['learning_rate'] * 0.01
        )
        
        print(f"✅ 优化器和调度器创建完成")
        print(f"   优化器: AdamW")
        print(f"   调度器: CosineAnnealingLR")
        
        return optimizer, scheduler
    
    def create_loss_function(self):
        """创建损失函数"""
        print(f"\n📉 创建损失函数...")
        
        if self.class_weights is not None:
            # 使用类别权重
            class_weights = self.class_weights.to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"✅ 使用加权交叉熵损失")
            print(f"   类别权重: {class_weights.cpu().numpy()}")
        else:
            # 标准交叉熵损失
            criterion = nn.CrossEntropyLoss()
            print(f"✅ 使用标准交叉熵损失")
        
        return criterion
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def train(self, epochs=None):
        """训练模型"""
        if epochs is None:
            epochs = self.config.TRAIN_CONFIG['epochs']
        
        print(f"\n🎯 开始训练 (共 {epochs} 个epoch)")
        print("=" * 60)
        
        # 创建组件
        data_loaders = self.create_data_loaders()
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val']
        model = self.create_model()
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        criterion = self.create_loss_function()
        
        # 训练循环
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(
                model, self.train_loader, optimizer, criterion
            )
            
            # 验证
            val_loss, val_acc, val_f1 = self.validate_epoch(
                model, self.val_loader, criterion
            )
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录日志
            self.training_logs['train_loss'].append(train_loss)
            self.training_logs['val_loss'].append(val_loss)
            self.training_logs['train_acc'].append(train_acc)
            self.training_logs['val_acc'].append(val_acc)
            self.training_logs['train_f1'].append(train_f1)
            self.training_logs['val_f1'].append(val_f1)
            self.training_logs['learning_rate'].append(current_lr)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✅ 训练完成，最佳验证准确率: {best_val_acc:.2f}%")
        
        return model, self.training_logs
    
    def generate_analysis(self, model, training_logs):
        """生成训练分析"""
        print(f"\n📊 生成训练分析...")
        
        # 创建分析器
        analyzer = IntegratedTrainingAnalyzer()
        
        # 生成综合分析
        report_path = analyzer.generate_comprehensive_analysis(
            training_logs=training_logs,
            ablation_results=None
        )
        
        print(f"✅ 分析报告已生成: {report_path}")
        return report_path

def main():
    """主函数"""
    print("🚀 储层流体识别 - 类别平衡训练")
    print("=" * 60)
    
    # 配置
    config = OptimizedConfig()
    
    # 测试不同平衡方法
    balance_methods = ['none', 'class_weight', 'bsmote']
    
    results = {}
    
    for method in balance_methods:
        print(f"\n{'='*60}")
        print(f"🔧 测试平衡方法: {method.upper()}")
        print(f"{'='*60}")
        
        try:
            # 创建训练器
            trainer = BalancedTrainer(
                config=config,
                balance_method=method,
                target_ratio=0.5
            )
            
            # 训练模型
            model, training_logs = trainer.train(epochs=10)  # 使用较少epoch进行测试
            
            # 记录结果
            best_val_acc = max(training_logs['val_acc'])
            best_val_f1 = max(training_logs['val_f1'])
            
            results[method] = {
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'training_logs': training_logs
            }
            
            print(f"✅ {method.upper()} 完成")
            print(f"   最佳验证准确率: {best_val_acc:.2f}%")
            print(f"   最佳验证F1分数: {best_val_f1:.4f}")
            
        except Exception as e:
            print(f"❌ {method.upper()} 失败: {e}")
            results[method] = None
    
    # 比较结果
    print(f"\n📊 结果比较")
    print("=" * 60)
    
    for method, result in results.items():
        if result is not None:
            print(f"{method.upper():12s}: 验证准确率 {result['best_val_acc']:6.2f}% | F1分数 {result['best_val_f1']:.4f}")
        else:
            print(f"{method.upper():12s}: 失败")
    
    # 找出最佳方法
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_method = max(valid_results.keys(), key=lambda k: valid_results[k]['best_val_acc'])
        print(f"\n🏆 最佳方法: {best_method.upper()}")
        print(f"   验证准确率: {valid_results[best_method]['best_val_acc']:.2f}%")
        print(f"   F1分数: {valid_results[best_method]['best_val_f1']:.4f}")

if __name__ == "__main__":
    main()
