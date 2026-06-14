import torch

class ImprovedConfig:
    """优化的配置类 - 修复验证集性能问题"""
    
    # 数据配置 - 优化数据分布和增强
    DATA_CONFIG = {
        'num_curves': 6,  # 六条测井曲线：GR、SP、AC、DEN、CNL、RT
        'curve_names': ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT'],
        'sequence_length': 100,  # 保持原有长度
        'num_classes': 6,  # 流体类型数量：油层、水层、干层、差油层、油水同层、含油水层
        'train_ratio': 0.7,  # 减少训练集比例，增加验证集
        'val_ratio': 0.3,   # 增加验证集比例，提高代表性
        'test_ratio': 0.0,
        'batch_size': 16,   # 增加批次大小，提高稳定性
        'num_workers': 0,   # 减少worker数量，避免数据加载问题
        'enable_wavelet': True,
        'enable_cleaning': True,
        'image_size': (64, 64),
        'pin_memory': False,  # 禁用pin_memory，提高稳定性
        'persistent_workers': False,
        'shuffle': True,     # 确保数据打乱
        'drop_last': False   # 保留不完整的批次
    }
    
    # 小波配置 - 保持原有设置
    WAVELET_CONFIG = {
        'enable_wavelet_packet': True,
        'curve_specific_wavelets': {
            'AC': {'wavelet': 'sym8', 'level': 4},
            'GR': {'wavelet': 'sym8', 'level': 4},
            'SP': {'wavelet': 'db4', 'level': 3},
            'CNL': {'wavelet': 'sym8', 'level': 4},
            'DEN': {'wavelet': 'sym8', 'level': 4},
            'RT': {'wavelet': 'sym8', 'level': 4}
        },
        'wavelet_mode': 'symmetric',
        'normalize_coeffs': True,
        'log_transform': True,
        'feature_resize_method': 'interpolation'
    }
    
    # 数据清洗配置 - 保持原有设置
    CLEANING_CONFIG = {
        'enable_cleaning': True,
        'denoising_methods': {
            'GR': 'gaussian_filter',
            'SP': 'savgol_filter',
            'AC': 'gaussian_filter',
            'DEN': 'gaussian_filter',
            'CNL': 'gaussian_filter',
            'RT': 'gaussian_filter'
        },
        'normalization_methods': {
            'GR': 'robust',
            'SP': 'standard',
            'AC': 'robust',
            'DEN': 'minmax',
            'CNL': 'robust',
            'RT': 'log_robust'
        },
        'outlier_thresholds': {
            'GR': 3.0,
            'SP': 2.5,
            'AC': 3.0,
            'DEN': 2.5,
            'CNL': 3.0,
            'RT': 3.0
        }
    }
    
    # CNN配置 - 减少模型复杂度，防止过拟合
    CNN_CONFIG = {
        'input_channels': 6,
        'conv_channels': [32, 64, 128, 256],  # 减少通道数
        'kernel_sizes': [3, 3, 3, 3],
        'pool_sizes': [2, 2, 2, 2],
        'dropout_rate': 0.3,  # 增加dropout
        'use_batch_norm': True,
        'activation': 'relu'
    }
    
    # Transformer配置 - 减少模型复杂度
    TRANSFORMER_CONFIG = {
        'd_model': 256,  # 减少维度
        'nhead': 8,
        'num_layers': 4,  # 减少层数
        'dim_feedforward': 1024,  # 减少FFN大小
        'dropout': 0.3,  # 增加dropout
        'max_seq_length': 100,
        'use_positional_encoding': True
    }
    
    # 训练配置 - 关键修复
    TRAIN_CONFIG = {
        'epochs': 200,  # 增加训练轮数
        'learning_rate': 0.001,  # 提高初始学习率
        'weight_decay': 5e-4,  # 增加权重衰减，防止过拟合
        'patience': 50,  # 大幅增加早停耐心值
        'min_delta': 0.0001,  # 降低最小改进阈值
        'monitor': 'val_loss',  # 改为监控验证损失，更稳定
        'monitor_mode': 'min',  # 监控模式：越小越好
        
        # 学习率调度优化
        'scheduler_type': 'cosine_annealing',  # 使用余弦退火调度
        'scheduler_t_max': 200,  # 余弦周期
        'scheduler_eta_min': 1e-6,  # 最小学习率
        
        # 早停策略优化
        'early_stopping_mode': 'min',  # 早停模式
        'restore_best_weights': True,  # 恢复最佳权重
        
        # 正则化增强
        'label_smoothing': 0.1,  # 标签平滑
        'mixup_alpha': 0.2,  # MixUp增强
        'cutmix_prob': 0.1,  # CutMix增强概率
        
        # 梯度控制
        'max_grad_norm': 1.0,
        'gradient_accumulation_steps': 1,
        
        # 其他优化
        'mixed_precision': False,  # 暂时禁用混合精度，提高稳定性
        'benchmark_cudnn': False,  # 禁用cuDNN基准测试
        'save_best_only': True,
        'optimizer': 'adamw',
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_eps': 1e-8
    }
    
    # 模型保存配置
    SAVE_CONFIG = {
        'model_dir': './checkpoints',
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'best_model_name': 'best_fluid_identification_model.pth',
        'wavelet_images_dir': './demo_outputs/wavelet_images',
        'cleaning_report_dir': './demo_outputs/cleaning_reports'
    }
    
    # 数据路径配置
    DATA_PATHS = {
        'welldata_dir': 'welldata',
        'well221_sample': 'welldata/geng221.txt',
        'well60_sample': 'welldata/geng60.txt',
        'well166_sample': 'welldata/geng166.txt',
        'well181_sample': 'welldata/geng181.txt',
        'well203_sample': 'welldata/geng203.txt',
        'well207_sample': 'welldata/geng207.txt',
        'well217_sample': 'welldata/geng217.txt',
        'well219_sample': 'welldata/geng219.txt',
        'well220_sample': 'welldata/geng220.txt',
        'well86_sample': 'welldata/geng86.txt',
        'ji117_sample': 'welldata/ji117.txt',
        'default_data': 'welldata/geng221.txt'
    }
