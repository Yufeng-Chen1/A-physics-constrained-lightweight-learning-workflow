#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用增强特征提取器的训练脚本
专门针对增强时频特征优化的训练流程
"""

import os
import sys
import time
import torch
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量（降低majority threshold以获得更多数据）
os.environ['MAJ_THRESH'] = '0.3'

from enhanced_training_config import create_enhanced_config
from data.spectrogram_data_loader import create_spectrogram_data_loaders
from models.fluid_identification_model import FluidIdentificationModel
from data.fluid_types import FluidTypes

def setup_training_environment():
    """设置训练环境"""
    print("🔧 设置训练环境...")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 优化GPU设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    return device

def create_enhanced_data_loaders(config):
    """创建增强特征数据加载器"""
    print("📊 创建增强特征数据加载器...")
    
    try:
        # 创建数据加载器，明确启用增强特征提取
        data_loaders = create_spectrogram_data_loaders(
            welldata_dir=config.DATA_PATHS['welldata_dir'],
            curve_names=config.DATA_CONFIG['curve_names'],
            scale_range=(5, 36),
            time_window=32,
            time_step=1,
            feature_size=config.DATA_CONFIG['image_size'],
            batch_size=config.DATA_CONFIG['batch_size'],
            train_ratio=config.DATA_CONFIG['train_ratio'],
            val_ratio=config.DATA_CONFIG['val_ratio'],
            test_ratio=config.DATA_CONFIG['test_ratio'],
            enable_cleaning=config.DATA_CONFIG['enable_cleaning'],
            num_workers=config.DATA_CONFIG['num_workers'],
            test_mode=False,
            use_fluid_mapping=True,
            use_stratified_split=True,
            random_state=42,
            wavelet_config=config.WAVELET_CONFIG  # 传递增强配置
        )
        
        print(f"   ✅ 数据加载器创建成功")
        print(f"   训练批次: {len(data_loaders['train'])}")
        print(f"   验证批次: {len(data_loaders['val'])}")
        
        return data_loaders
        
    except Exception as e:
        print(f"   ❌ 数据加载器创建失败: {e}")
        raise

def create_model(config, device):
    """创建模型"""
    print("🧠 创建模型...")
    
    try:
        model = FluidIdentificationModel(config)
        model = model.to(device)
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✅ 模型创建成功")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        return model
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        raise

def create_optimizer_and_scheduler(model, config):
    """创建优化器和学习率调度器"""
    print("⚙️  创建优化器和调度器...")
    
    # 创建优化器
    if config.OPTIMIZER_CONFIG['type'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.OPTIMIZER_CONFIG['lr'],
            weight_decay=config.OPTIMIZER_CONFIG['weight_decay'],
            betas=config.OPTIMIZER_CONFIG['betas'],
            eps=config.OPTIMIZER_CONFIG['eps']
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.TRAIN_CONFIG['learning_rate'],
            weight_decay=config.TRAIN_CONFIG['weight_decay']
        )
    
    # 创建学习率调度器
    if config.SCHEDULER_CONFIG['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.SCHEDULER_CONFIG['T_max'],
            eta_min=config.SCHEDULER_CONFIG['eta_min']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
    
    print(f"   ✅ 优化器: {config.OPTIMIZER_CONFIG['type']}")
    print(f"   ✅ 调度器: {config.SCHEDULER_CONFIG['type']}")
    
    return optimizer, scheduler

def create_loss_function(config):
    """创建损失函数"""
    print("🎯 创建损失函数...")
    
    # 使用Focal Loss
    from torch.nn import CrossEntropyLoss
    
    # 如果有类别权重，创建加权损失
    if 'alpha' in config.LOSS_CONFIG['focal_loss']:
        alpha = torch.tensor(config.LOSS_CONFIG['focal_loss']['alpha'])
        criterion = CrossEntropyLoss(weight=alpha)
    else:
        criterion = CrossEntropyLoss()
    
    print(f"   ✅ 损失函数创建成功")
    
    return criterion

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    """主训练函数"""
    print("🚀 增强特征提取器训练开始")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 设置环境
        device = setup_training_environment()
        
        # 加载配置
        config = create_enhanced_config()
        
        # 打印流体类型信息
        print("\n🏷️  流体类型信息:")
        FluidTypes.print_fluid_types()
        
        # 创建数据加载器
        data_loaders = create_enhanced_data_loaders(config)
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        
        # 检查数据集大小
        if len(train_loader) == 0:
            print("❌ 训练集为空，请检查数据和配置")
            return
        
        # 创建模型
        model = create_model(config, device)
        
        # 创建优化器和调度器
        optimizer, scheduler = create_optimizer_and_scheduler(model, config)
        
        # 创建损失函数
        criterion = create_loss_function(config)
        criterion = criterion.to(device)
        
        # 训练循环
        print(f"\n🎯 开始训练 (总轮数: {config.TRAIN_CONFIG['epochs']})")
        print("=" * 60)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(config.TRAIN_CONFIG['epochs']):
            print(f"\n📊 Epoch {epoch+1}/{config.TRAIN_CONFIG['epochs']}")
            
            # 训练
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # 验证
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # 学习率调度
            if config.SCHEDULER_CONFIG['type'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            # 输出结果
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"   LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': config
                }, 'enhanced_feature_best_model.pth')
                
                print(f"   🏆 新的最佳验证准确率: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= config.TRAIN_CONFIG['patience']:
                print(f"\n⏹️  早停触发 (耐心值: {config.TRAIN_CONFIG['patience']})")
                break
        
        # 训练完成
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print(f"   总训练时间: {total_time/60:.1f}分钟")
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   模型保存: enhanced_feature_best_model.pth")
        
        print("\n📊 增强特征提取器训练总结:")
        print(f"   ✅ 使用了多尺度小波包分解")
        print(f"   ✅ 使用了连续小波变换(CWT)")
        print(f"   ✅ 使用了75%重叠时间窗口")
        print(f"   ✅ 提取了丰富的时频联合特征")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()