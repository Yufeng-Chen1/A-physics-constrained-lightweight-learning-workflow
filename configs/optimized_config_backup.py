#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于optimized_train.py的配置文件 - 支持64×64×6时频图谱输入
"""

class OptimizedConfig:
    """优化的配置类 - 支持64×64×6时频图谱输入"""
    
    # 数据配置 - 优化数据利用，支持时频图谱
    DATA_CONFIG = {
        'curve_names': ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC'],
        'num_curves': 6,  # 6个测井曲线
        'num_classes': 6,  # 6个类别：油层、水层、干层、差油层、油水同层、含油水层
        'sequence_length': 100,  # 原始序列长度
        'batch_size': 32,  # 进一步优化批次大小，提高GPU利用率
        'train_ratio': 0.8,  # 训练集比例：80%
        'val_ratio': 0.1,   # 验证集比例：10%
        'test_ratio': 0.1,  # 测试集比例：10%
        'enable_wavelet': True,  # 启用小波变换
        'enable_cleaning': True,  # 启用数据清洗
        'image_size': (64, 64),  # 时频图谱尺寸：64×64
        'num_workers': 0,  # 禁用多进程数据加载，避免Windows上的pickle问题
        'pin_memory': True,  # 内存固定
        'shuffle': True,
        'drop_last': True,
        'overlap_ratio': 0.5,  # 序列重叠比例，增加样本数量
        'use_all_wells': True,  # 使用所有井数据
        'min_sequence_length': 32,  # 最小序列长度
        'max_sequence_length': 100,  # 最大序列长度
        # BSMOTE数据平衡配置
        'enable_bsmote': False,  # 启用BSMOTE
        'bsmote_target_ratio': 0.6,  # BSMOTE目标平衡比例
        'bsmote_k_neighbors': 5,  # BSMOTE近邻数
        'bsmote_random_state': 42,  # BSMOTE随机种子
        # 类别权重配置
        'enable_class_weights': True,  # 启用类别权重
        'class_weight_method': 'balanced',  # 类别权重计算方法
        # PCA降维配置
        'enable_pca': False,  # 启用PCA降维
        'pca_components': 1000,  # PCA组件数
        'pca_variance_ratio': 0.90,  # PCA保留方差比例
        'pca_auto_components': True,  # 自动计算PCA组件数
        # 新增：指定训练和验证井
        'train_wells': ['geng221', 'geng60', 'geng166', 'geng181', 'geng203', 'geng207', 'geng217', 'geng219'],  # 8口训练井
        'val_wells': ['geng220', 'geng86', 'ji117']  # 3口验证井
    }
    
    # 小波包分解配置 - 专门用于生成时频图谱
    WAVELET_CONFIG = {
        'enable_wavelet_packet': True,  # 启用小波包分解
        'wavelet_type': 'db8',  # 小波基类型
        'decomposition_level': 2,  # 分解层数（减少到2，适应数据长度）
        'scale_range': (5, 36),  # 尺度范围：5-36个采样点
        'time_window': 32,  # 时间窗口：32个采样点
        'time_step': 1,  # 时间步长：1个采样点
        'output_size': (64, 64),  # 输出时频图谱尺寸
        'curve_specific_wavelets': {
            'AC': {'wavelet': 'db8', 'level': 2},      # AC使用db8小波基分解2层
            'GR': {'wavelet': 'sym8', 'level': 2},     # GR使用sym8小波基分解2层
            'SP': {'wavelet': 'db4', 'level': 2},      # SP使用db4小波基分解2层
            'CNL': {'wavelet': 'sym8', 'level': 2},    # CNL使用sym8小波基分解2层
            'DEN': {'wavelet': 'sym8', 'level': 2},    # DEN使用sym8小波基分解2层
            'RT': {'wavelet': 'sym8', 'level': 2}      # RT使用sym8小波基分解2层
        },
        'wavelet_mode': 'symmetric',  # 小波变换模式
        'normalize_coefficients': True,  # 归一化系数
        'use_energy_spectrum': True,  # 使用能量谱
        'save_wavelet_images': False,  # 是否保存小波图像
        'wavelet_images_dir': './demo_outputs/wavelet_images'
    }
    
    # 数据清洗配置 - 增强数据质量
    CLEANING_CONFIG = {
        'enable_cleaning': True,
        'outlier_method': 'zscore',  # 使用Z-score方法
        'normalization_methods': {
            'GR': 'robust',
            'SP': 'standard',
            'AC': 'robust',
            'DEN': 'minmax',
            'CNL': 'robust',
            'RT': 'log_robust'
        },
        'outlier_thresholds': {
            'GR': 2.5,  # 降低阈值，更严格的数据清洗
            'SP': 2.0,
            'AC': 2.5,
            'DEN': 2.0,
            'CNL': 2.5,
            'RT': 2.5
        },
        'enable_interpolation': True,  # 启用插值填充缺失值
        'interpolation_method': 'linear',  # 线性插值
        'enable_smoothing': True,  # 启用数据平滑
        'smoothing_window': 5  # 平滑窗口大小
    }
    
    # CNN配置 - 适应64×64×6输入
    CNN_CONFIG = {
        'input_channels': 6,  # 6个通道
        'conv_channels': [32, 64, 128, 256],  # 减少通道数，适应更大的图像
        'kernel_sizes': [3, 3, 3, 3],  # 3x3卷积核
        'pool_sizes': [2, 2, 2, 2],  # 2x2池化
        'dropout_rate': 0.4,  # 增加dropout，防止过拟合
        'use_batch_norm': True,
        'activation': 'relu',
        'use_residual': True,  # 启用残差连接
        'use_se_block': True,  # 启用SE注意力机制
        'expansion_factor': 2  # 通道扩展因子
    }
    
    # Transformer配置 - 优化注意力机制
    TRANSFORMER_CONFIG = {
        'd_model': 128,  # 减少模型维度，适应更大的图像
        'nhead': 8,  # 注意力头数
        'num_layers': 2,  # 减少层数，提高训练效率
        'dim_feedforward': 512,  # 减少FFN大小
        'dropout': 0.4,  # 增加dropout
        'max_seq_length': 100,
        'use_positional_encoding': True,
        'use_layer_norm': True,  # 启用层归一化
        'use_pre_norm': True,  # 使用Pre-Norm结构
        'attention_dropout': 0.2,  # 注意力dropout
        'use_relative_position': True  # 使用相对位置编码
    }
    
    # 训练配置 - 优化训练策略
    TRAIN_CONFIG = {
        'epochs': 100,  # 适中轮数，配合OneCycle更快收敛
        'learning_rate': 0.001,  # 大幅提升学习率，促进收敛
        'weight_decay': 5e-4,  # 适度增加权重衰减，防止过拟合
        'patience': 80,  # 提高耐心，避免过早早停
        'min_delta': 0.001,  # 改进阈值
        'min_epochs': 80,  # 至少训练80轮再考虑早停
        'monitor': 'val_f1',  # 监控F1分数
        'monitor_mode': 'max',  # 监控模式：越大越好
        
        # 学习率调度优化
        'scheduler_type': 'cosine_annealing',  # 使用余弦退火重启调度器
        'scheduler_t_0': 30,  # 重启周期
        'scheduler_t_mult': 2,  # 周期倍增
        'scheduler_eta_min': 5e-5,  # 最小学习率
        'warmup_epochs': 5,  # 预热轮数
        'warmup_factor': 0.1,  # 预热因子
        
        # 早停配置
        'early_stopping': True,
        'early_stopping_mode': 'max',
        'early_stopping_patience': 50,
        'early_stopping_min_delta': 0.001,
        'early_stopping_min_epochs': 50,
        
        # 正则化配置
        'label_smoothing': 0.1,  # 启用标签平滑，提高泛化能力
        'mixup_alpha': 0.5,  # MixUp增强
        'cutmix_prob': 0.3,  # CutMix概率
        'randaugment': True,  # RandAugment
        'randaugment_m': 15,  # RandAugment强度
        'randaugment_n': 3,  # RandAugment操作数
        
        # 梯度配置
        'gradient_clip': True,
        'gradient_clip_value': 2.0,  # 增加梯度裁剪值，提高学习能力
        'gradient_clip_norm': 1.0,
        
        # 混合精度训练
        'use_amp': True,
        'amp_dtype': 'float16',

        # 损失函数配置
        'use_focal_loss': True,  # 启用Focal Loss处理类别不平衡
    }
    
    # 数据增强配置 - 增强数据多样性
    AUGMENTATION_CONFIG = {
        'enable_augmentation': True,
        'noise_factor': 0.05,  # 噪声因子
        'scaling_factor': 0.1,  # 缩放因子
        'rotation_angle': 5,  # 旋转角度
        'translation_factor': 0.05,  # 平移因子
        'elastic_deformation': True,  # 弹性变形
        'elastic_alpha': 1.0,  # 弹性变形强度
        'elastic_sigma': 50.0,  # 弹性变形标准差
        'elastic_alpha_affine': 50.0  # 仿射弹性变形强度
    }
    
    # 模型保存配置
    MODEL_CONFIG = {
        'save_best_model': True,
        'save_last_model': True,
        'model_save_dir': './checkpoints',
        'model_name_prefix': 'fluid_identification',
        'save_optimizer_state': True,
        'save_scheduler_state': True
    }
    
    # 日志配置
    LOGGING_CONFIG = {
        'enable_tensorboard': True,
        'tensorboard_log_dir': './logs/tensorboard',
        'enable_wandb': False,
        'log_interval': 10,
        'save_interval': 50
    }
    
    # 验证配置
    VALIDATION_CONFIG = {
        'validation_interval': 1,  # 每个epoch都验证
        'save_validation_predictions': True,
        'validation_predictions_dir': './validation_predictions',
        'compute_confusion_matrix': True,
        'compute_classification_report': True
    }
    
    # 数据路径配置 - 添加缺少的DATA_PATHS属性
    DATA_PATHS = {
        'welldata_dir': 'welldata',  # 井数据目录
        'well221_sample': 'welldata/geng221.txt',  # 耿221井数据
        'well60_sample': 'welldata/geng60.txt',    # 耿60井数据
        'well166_sample': 'welldata/geng166.txt',  # 耿166井数据
        'well181_sample': 'welldata/geng181.txt',  # 耿181井数据
        'well203_sample': 'welldata/geng203.txt',  # 耿203井数据
        'well207_sample': 'welldata/geng207.txt',  # 耿207井数据
        'well217_sample': 'welldata/geng217.txt',  # 耿217井数据
        'well219_sample': 'welldata/geng219.txt',  # 耿219井数据
        'well220_sample': 'welldata/geng220.txt',  # 耿220井数据
        'well86_sample': 'welldata/geng86.txt',    # 耿86井数据
        'ji117_sample': 'welldata/ji117.txt',      # 吉117井数据
        'default_data': 'welldata/geng221.txt'     # 默认使用耿221井数据
    }
    
    # 模型保存配置 - 添加缺少的SAVE_CONFIG属性
    SAVE_CONFIG = {
        'checkpoint_dir': './checkpoints',
        'model_dir': './models',
        'log_dir': './logs',
        'visualization_dir': './visualizations'
    }