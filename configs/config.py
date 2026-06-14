import torch

class Config:
    # 數據配置 - 優化大數據集和GPU訓練
    DATA_CONFIG = {
        'num_curves': 6,  # 六條測井曲線：GR、SP、AC、DEN、CNL、RT
        'curve_names': ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT'],  # 測井曲線名稱
        'sequence_length': 25,  # 修复：使用实际的数据长度，避免维度不匹配
        'num_classes': 6,  # 流体类型数量：油层、水层、干层、差油层、油水同层、含油水层
        'train_ratio': 0.8,  # 增加训练集比例
        'val_ratio': 0.2,  # 减少验证集比例
        'test_ratio': 0.0,  # 不设置测试集，只用于训练和验证
        'batch_size': 8,  # 适中的批次大小，平衡内存和稳定性
        'num_workers': 2,  # 减少worker數量，避免数据加载问题
        'enable_wavelet': True,  # 是否啟用小波包分解
        'enable_cleaning': True,  # 是否启用数据清洗
        'image_size': (16, 16),  # 修复：使用适中的图像尺寸，既不太小也不太大
        'pin_memory': True,  # 啟用pin_memory加速
        'persistent_workers': False  # 禁用持久化worker，避免内存问题
    }
    
    # 小波包分解配置 - 根據您的要求優化
    WAVELET_CONFIG = {
        'enable_wavelet_packet': True,  # 啟用小波包分解
        'curve_specific_wavelets': {
            'AC': {'wavelet': 'sym8', 'level': 4},      # AC使用sym8小波基分解4层
            'GR': {'wavelet': 'sym8', 'level': 4},      # GR使用sym8小波基分解4层
            'SP': {'wavelet': 'db4', 'level': 3},       # SP使用db4小波基分解3层
            'CNL': {'wavelet': 'sym8', 'level': 4},     # CNL使用sym8小波基分解4层
            'DEN': {'wavelet': 'sym8', 'level': 4},     # DEN使用sym8小波基分解4层
            'RT': {'wavelet': 'sym8', 'level': 4}       # RT使用sym8小波基分解4层
        },
        'wavelet_mode': 'symmetric',  # 小波變換模式
        'normalize_coeffs': True,  # 是否標準化係數
        'log_transform': True,  # 是否使用對數變換
        'feature_resize_method': 'interpolation'  # 特徵尺寸調整方法
    }
    
    # 數據清洗配置
    CLEANING_CONFIG = {
        'enable_cleaning': True,  # 啟用數據清洗
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
    
    # 高清CNN配置 - 優化版本
    CNN_CONFIG = {
        'input_channels': 6,  # 輸入通道數（六條測井曲線）
        'conv_channels': [64, 128, 256, 512],  # 提升容量：更多通道
        'kernel_sizes': [3, 3, 3, 3],  # 卷積核大小
        'pool_sizes': [2, 2, 2, 2],  # 池化層大小
        'dropout_rate': 0.2,  # 适度降低，保留表征能力
        'use_batch_norm': True,  # 使用批標準化
        'activation': 'relu'  # 激活函數
    }
    
    # 高清Transformer配置 - 優化版本
    TRANSFORMER_CONFIG = {
        'd_model': 384,  # 提升维度
        'nhead': 8,  # 注意力头数保持
        'num_layers': 6,  # 层数增加
        'dim_feedforward': 1536,  # FFN更大
        'dropout': 0.2,
        'max_seq_length': 128,  # 匹配新的序列长度
        'use_positional_encoding': True  # 使用位置編碼
    }
    
    # 訓練配置 - 優化訓練效率
    TRAIN_CONFIG = {
        'epochs': 100,  # 适度减少训练轮数
        'min_epochs': 20,  # 最少训练轮数，避免过早停止
        'learning_rate': 0.0005,  # 降低学习率提高稳定性
        'weight_decay': 1e-4,  # 适度权重衰减
        'scheduler_step_size': 15,  # 更频繁的学习率调度
        'scheduler_gamma': 0.7,  # 更强的学习率衰减
        'patience': 25,  # 适中的早停耐心值
        'min_delta': 0.001,  # 降低最小改进幅度
        'use_cosine_scheduler': True,  # 使用余弦退火调度器
        'device': torch.device('cuda'),  # 強制使用GPU
        'mixed_precision': True,  # 使用混合精度訓練
        'gradient_accumulation_steps': 1,  # 減少梯度累積
        'max_grad_norm': 1.0,  # 梯度裁剪
        'gpu_memory_fraction': 0.8,  # 減少GPU內存使用
        'benchmark_cudnn': True,  # 啟用cuDNN基準測試
        'save_best_only': True,  # 只保存最佳模型
        'monitor': 'val_f1',  # 監控F1分數
        'optimizer': 'adamw',  # 使用AdamW优化器
        'adam_beta1': 0.9,  # Adam优化器beta1参数
        'adam_beta2': 0.999,  # Adam优化器beta2参数
        'adam_eps': 1e-8  # Adam优化器epsilon参数
    }
    
    # 模型保存配置
    SAVE_CONFIG = {
        'model_dir': './checkpoints',
        'checkpoint_dir': './checkpoints',  # 检查点保存目录
        'log_dir': './logs',
        'best_model_name': 'best_fluid_identification_model.pth',
        'wavelet_images_dir': './demo_outputs/wavelet_images',  # 小波時頻圖譜保存目錄
        'cleaning_report_dir': './demo_outputs/cleaning_reports'  # 清洗報告保存目錄
    }
    
    # 數據路徑配置
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