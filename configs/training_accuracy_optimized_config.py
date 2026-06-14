#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集准确率优化配置
专门用于提升训练集准确率，减少训练集和验证集之间的性能差距
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainingAccuracyOptimizedConfig:
    """训练集准确率优化配置"""
    
    def __init__(self):
        # ==================== 数据配置 ====================
        self.DATA_CONFIG = {
            'data_dir': 'welldata',
            'train_ratio': 0.8,
            'val_ratio': 0.2,
            'batch_size': 32,  # 增加批次大小，提高训练稳定性
            'num_workers': 4,
            'sequence_length': 100,
            'spectrogram_size': (64, 100),
            'scale_range': (1, 32),
            'use_wavelet': True,
            'normalize': True,
            'shuffle': True,
        }
        
        # ==================== 模型配置 ====================
        self.MODEL_CONFIG = {
            'model_type': 'enhanced',  # 使用增强模型
            'input_channels': 1,
            'num_classes': 6,  # 6个类别：油层、水层、干层、差油层、油水同层、含油水层
            'dropout_rate': 0.1,  # 大幅降低dropout，从0.3降到0.1
            'use_batch_norm': True,
            'use_attention': True,
        }
        
        # ==================== CNN配置 ====================
        self.CNN_CONFIG = {
            'conv_channels': [32, 64, 128, 256],  # 保持足够的特征提取能力
            'kernel_sizes': [3, 3, 3, 3],
            'pool_sizes': [2, 2, 2, 2],
            'dropout_rate': 0.1,  # 降低CNN的dropout
            'use_batch_norm': True,
            'use_attention': True,
        }
        
        # ==================== Transformer配置 ====================
        self.TRANSFORMER_CONFIG = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 1024,
            'dropout': 0.1,  # 降低Transformer的dropout
            'max_seq_len': 100,
            'use_positional_encoding': True,
        }
        
        # ==================== 训练配置 ====================
        self.TRAINING_CONFIG = {
            'epochs': 200,  # 增加训练轮数
            'learning_rate': 0.002,  # 提高学习率，加快收敛
            'weight_decay': 1e-4,  # 降低权重衰减，减少正则化
            'optimizer': 'adamw',
            'scheduler': 'cosine',  # 使用余弦退火调度器
            'warmup_epochs': 10,  # 添加学习率预热
            'min_lr': 1e-6,
            'gradient_clip': 1.0,
            'early_stopping_patience': 50,  # 增加耐心值
            'save_best_model': True,
            'save_checkpoint': True,
        }
        
        # ==================== 数据增强配置 ====================
        self.AUGMENTATION_CONFIG = {
            'use_mixup': False,  # 完全禁用MixUp，这是主要影响因素
            'mixup_alpha': 0.2,
            'mixup_prob': 0.0,  # 设置为0，完全禁用
            'use_cutmix': False,  # 禁用CutMix
            'cutmix_alpha': 1.0,
            'cutmix_prob': 0.0,
            'use_random_crop': False,  # 禁用随机裁剪
            'use_random_flip': False,  # 禁用随机翻转
            'use_noise': False,  # 禁用噪声添加
            'noise_std': 0.01,
        }
        
        # ==================== 损失函数配置 ====================
        self.LOSS_CONFIG = {
            'loss_type': 'cross_entropy',
            'label_smoothing': 0.0,  # 禁用标签平滑，这是影响训练准确率的重要因素
            'use_focal_loss': False,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'use_class_weights': True,  # 保持类别权重平衡
        }
        
        # ==================== 验证配置 ====================
        self.VALIDATION_CONFIG = {
            'val_frequency': 1,  # 每个epoch都验证
            'val_metrics': ['accuracy', 'f1', 'precision', 'recall'],
            'save_predictions': True,
            'save_confusion_matrix': True,
        }
        
        # ==================== 输出配置 ====================
        self.OUTPUT_CONFIG = {
            'output_dir': 'training_accuracy_optimized_outputs',
            'log_frequency': 10,  # 更频繁的日志记录
            'save_frequency': 10,  # 更频繁的模型保存
            'tensorboard': True,
            'plot_training_curves': True,
            'save_model_architecture': True,
        }
        
        # ==================== 设备配置 ====================
        self.DEVICE_CONFIG = {
            'use_cuda': True,
            'cuda_device': 0,
            'mixed_precision': False,  # 禁用混合精度，提高稳定性
            'compile_model': False,  # 禁用模型编译，避免兼容性问题
        }
        
        # ==================== 调试配置 ====================
        self.DEBUG_CONFIG = {
            'debug_mode': False,
            'verbose': True,
            'save_gradients': False,
            'save_activations': False,
            'profile_memory': False,
        }
    
    def get_config(self):
        """获取完整配置"""
        return {
            'data': self.DATA_CONFIG,
            'model': self.MODEL_CONFIG,
            'cnn': self.CNN_CONFIG,
            'transformer': self.TRANSFORMER_CONFIG,
            'training': self.TRAINING_CONFIG,
            'augmentation': self.AUGMENTATION_CONFIG,
            'loss': self.LOSS_CONFIG,
            'validation': self.VALIDATION_CONFIG,
            'output': self.OUTPUT_CONFIG,
            'device': self.DEVICE_CONFIG,
            'debug': self.DEBUG_CONFIG,
        }
    
    def print_config(self):
        """打印配置信息"""
        print("🚀 训练集准确率优化配置")
        print("=" * 50)
        print(f"📊 数据配置:")
        print(f"  批次大小: {self.DATA_CONFIG['batch_size']}")
        print(f"  训练比例: {self.DATA_CONFIG['train_ratio']}")
        print(f"  验证比例: {self.DATA_CONFIG['val_ratio']}")
        
        print(f"\n🧠 模型配置:")
        print(f"  Dropout率: {self.MODEL_CONFIG['dropout_rate']}")
        print(f"  CNN Dropout: {self.CNN_CONFIG['dropout_rate']}")
        print(f"  Transformer Dropout: {self.TRANSFORMER_CONFIG['dropout']}")
        
        print(f"\n🎯 训练配置:")
        print(f"  学习率: {self.TRAINING_CONFIG['learning_rate']}")
        print(f"  权重衰减: {self.TRAINING_CONFIG['weight_decay']}")
        print(f"  训练轮数: {self.TRAINING_CONFIG['epochs']}")
        
        print(f"\n🔄 数据增强配置:")
        print(f"  MixUp: {'禁用' if not self.AUGMENTATION_CONFIG['use_mixup'] else '启用'}")
        print(f"  CutMix: {'禁用' if not self.AUGMENTATION_CONFIG['use_cutmix'] else '启用'}")
        print(f"  随机裁剪: {'禁用' if not self.AUGMENTATION_CONFIG['use_random_crop'] else '启用'}")
        
        print(f"\n📉 损失函数配置:")
        print(f"  标签平滑: {'禁用' if self.LOSS_CONFIG['label_smoothing'] == 0 else f'{self.LOSS_CONFIG['label_smoothing']}'}")
        print(f"  类别权重: {'启用' if self.LOSS_CONFIG['use_class_weights'] else '禁用'}")
        
        print("=" * 50)

if __name__ == "__main__":
    config = TrainingAccuracyOptimizedConfig()
    config.print_config()
