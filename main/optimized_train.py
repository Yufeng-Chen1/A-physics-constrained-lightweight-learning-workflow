#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强优化训练脚本 - 集成所有优化策略
解决训练精度停滞问题，提高模型性能
集成BSMOTE数据平衡、模型架构优化、训练策略优化
"""

import os
import sys
import time
from datetime import datetime
import random
import torch
import os as _os
try:
    # 全局無頭渲染，避免 Qt 後端報錯
    import matplotlib
    matplotlib.use('Agg')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
except Exception:
    pass
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except Exception as _tb_e:  # noqa: F841
    SummaryWriter = None  # type: ignore
    _TB_AVAILABLE = False
    print(f"⚠️  TensorBoard不可用，将禁用日志记录: {_tb_e}")
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import warnings
import json
import copy

# 可选依赖导入，避免编译器警告
_PSUTIL_AVAILABLE = False
_PYNVML_AVAILABLE = False

# 仅在需要时导入psutil，避免编译器警告
try:
    import psutil  # type: ignore
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    print("⚠️  psutil未安装，部分系统监控功能将禁用. 请安装: pip install psutil")

# 仅在需要时导入pynvml，避免编译器警告  
try:
    import pynvml  # type: ignore
    _PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None  # type: ignore
    print("⚠️  pynvml未安装，GPU内存监控功能将禁用. 请安装: pip install pynvml")

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要模块
try:
    from data.class_balance_handler import ClassBalanceHandler
    from data.spectrogram_data_loader import create_spectrogram_data_loaders
    from data.enhanced_wavelet_processor import EnhancedWaveletProcessor
    from data.fluid_types import get_fluid_name, get_num_fluid_classes
    from models.fluid_identification_model import FluidIdentificationModel
    from configs.optimized_config import OptimizedConfig
    from visualization.training_analysis import IntegratedTrainingAnalyzer
    from visualizations.training_analysis_manager import UnifiedVisualizationManager
except ImportError as e:
    print(f"⚠️ 导入模块失败: {e}")
    print("请确保所有依赖模块都已正确安装")

def create_enhanced_config():
    """创建增强配置 - 专门解决精度停滞问题"""
    config = OptimizedConfig()
    
    # 重新设计训练配置 - AWPD-MDSC-TAM方案（稳定学习动态）
    config.TRAIN_CONFIG.update({
        'epochs': 200,
        'learning_rate': 3e-4,  # 更稳定的初始LR
        'weight_decay': 1e-4,   # 适度权重衰减
        'patience': 30,
        'scheduler_type': 'cosine',
        'max_lr': 3e-4,
        'min_lr': 5e-6,         # 最小学习率（更低尾端退火）
        'warmup_epochs': 5,     # 预热至base lr
        'min_epochs': 50,       # 至少训练N轮才允许早停
        'label_smoothing': 0.02,
        'gradient_clip': True,
        'gradient_clip_value': 1.0,
        'use_amp': True,        # 混合精度训练(FP16)
        'early_stopping_patience': 5,  # 更快响应的早停
        'min_delta': 0.0001,
        'debug_mode': False,
        'validation_interval': 3,
        'performance_monitoring': True,
        'use_focal_loss': True,  # 启用Focal Loss
        'focal_alpha': 1.0,      # α=1，依赖类权重，不单独放大
        'focal_gamma': 1.2,      # γ=1.2（稳定默认）
        'class3_boost': 1.8,     # 差油层(ID=3)权重放大因子（可通过环境变量覆盖）
        # EMA权重
        'use_ema': True,
        'ema_decay': 0.995,
        'memory_monitor_interval': 50,
        'gpu_memory_threshold': 0.92,
        'batch_size_adjust_factor': 0.5,
        'optimizer_type': 'adamw',
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        # 运行模式配置
        'run_mode': 'full',
        'use_torch_compile': False,
        'use_channels_last': True,
        'grad_accum_steps': 2,
        'use_gradient_checkpointing': False,
        # 模型缩放
        'width_mult': 0.75,
        'transformer_width_mult': 0.75,
        # 极少数类召回监控
        'monitor_rare_recall': True,
        'rare_recall_target': 0.7,    # 目标>70%
        'rare_patience': 10,          # 连续10轮无提升
    })
    
    # AWPD数据配置 - 自适应小波包分解策略
    config.DATA_CONFIG.update({
        'batch_size': 128,
        'num_curves': 6,
        'sequence_length': 64,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'enable_cleaning': True,
        'use_stratified_split': True,
                'enable_bsmote': False,
                'bsmote_target_ratio': 0.7,  # 關閉BSMOTE，僅保留加權與抽樣
        # AWPD策略
        'enable_wavelet_augmentation': False,  # 禁用复杂小波增强
        'wavelet_diversity_mode': 'awpd',      # AWPD模式
        'wavelet_overlap_ratio': 0.75,         # 75%重叠
        'wavelet_compensation_mode': 'periodic',  # 周期补偿消除边界失真
        # 数据增强（仅训练集）
        'enable_frequency_masking': True,      # 频率掩蔽
        'frequency_masking_ratio': 0.2,        # 遮挡10~15Hz，20%比例
        'enable_time_shift': True,             # 时间偏移
        'time_shift_range': 2,                 # ±2时间步
        'enable_noise_augmentation': True,     # 高斯白噪声
        'noise_snr_db': 20,                    # SNR=20dB
        # 数据预处理
        'use_simple_preprocessing': False,     # 使用完整AWPD预处理
        'enable_normalization': True,
        'normalization_method': 'z_score',
        # 多数票阈值（可被环境变量 MAJ_THRESH 覆盖）
        'majority_threshold': 0.70,
        # 轻量域稳健增强（不依赖CPU小波，直接在GPU上执行）
        'enable_light_augmentations': True,
        'light_frequency_masking_ratio': 0.10,
        'light_time_shift_range': 1,
        'light_noise_snr_db': 22,
        # 采样与平滑
        'priority_classes': [1, 3],
        'sampler_priority_boost': 1.2,
        'enable_smoothing': True,
        'smooth_window': 5,
        # 均衡批采样默认开启
        'use_balanced_batch': True,
        'use_weighted_sampler': False,
    })

    # 环境变量覆盖多数票阈值
    try:
        maj_env = os.environ.get('MAJ_THRESH', '').strip()
        if maj_env:
            config.DATA_CONFIG['majority_threshold'] = float(maj_env)
            print(f"🎚️  MAJ_THRESH 覆盖: {config.DATA_CONFIG['majority_threshold']}")
    except Exception:
        pass

    # “无”类ID（可选）：用于评估时排除“无”类并下调其权重
    try:
        none_id_env = os.environ.get('NONE_CLASS_ID', '').strip()
        if none_id_env:
            config.MODEL_CONFIG['none_class_id'] = int(none_id_env)
        else:
            # 若类别数>=6，缺省假设ID=5为“无”
            config.MODEL_CONFIG['none_class_id'] = config.MODEL_CONFIG.get('none_class_id', None)
    except Exception:
        pass
    
    # MDSC CNN配置 - 多分支深度可分离卷积（支持宽度缩放）
    width_mult = float(config.TRAIN_CONFIG.get('width_mult', 1.0))
    base_channels = [64, 128, 256]  # MDSC使用3个分支
    scaled_channels = [max(16, int(c * width_mult)) for c in base_channels]
    fusion_channels = max(256, int(512 * width_mult))  # 融合层通道数也需要缩放
    
    config.CNN_CONFIG.update({
        'input_channels': 6,  # 6个通道
        'conv_channels': scaled_channels,  # 3个分支的通道数
        'kernel_sizes': [3, 5, 7],  # 不同分支的卷积核大小
        'pool_sizes': [1, 2, 2],  # 分支的步长 (1, 2, 2)
        'fusion_channels': fusion_channels,  # 融合层输出通道数
        'dropout_rate': 0.2,  # 降低Dropout提高学习能力
        'use_batch_norm': True,
        'activation': 'relu',
        'use_residual': True,
        'use_se_block': False,  # MDSC中不使用SE
        'use_attention': False,  # 注意力在TAM中处理
        'num_conv_layers': 3,  # 3个分支
    })
    
    # TAM Transformer配置 - 三重注意力机制（支持宽度缩放）
    t_width_mult = float(config.TRAIN_CONFIG.get('transformer_width_mult', 1.0))
    d_model = fusion_channels  # 与MDSC融合层输出匹配
    ff_dim = max(256, int(d_model * 2))  # FFN通常是d_model的2倍
    
    config.TRANSFORMER_CONFIG.update({
        'd_model': d_model,  # 匹配fusion_channels
        'nhead': 4,  # 4个注意力头
        'num_layers': 1,  # 简化为1层TAM，稳定训练
        'dim_feedforward': ff_dim,  # FFN隐藏层维度
        'dropout': 0.2,  # 降低Dropout
        'max_seq_length': 256,  # 序列长度 (16x16=256)
        'use_positional_encoding': True,  # 使用二维位置编码
        'activation': 'gelu',  # GELU激活函数
        'attention_type': 'self_cross',  # Self + Cross attention
    })
    
    # 新增：精度提升专用配置
    config.BREAKTHROUGH_CONFIG = {
        'enable_breakthrough_mode': False,  # 默認禁用突破模式（避免大模型與長訓練）
        'aggressive_learning': True,  # 激进学习模式
        'disable_complex_preprocessing': True,  # 禁用复杂预处理
        'use_larger_model': False,  # 默認不使用超大模型
        'increase_batch_size': True,  # 增加批次大小
        'reduce_regularization': True,  # 减少正则化
        'extend_training': True,  # 延长训练时间
    }

    # 运行模式开关：'efficient' | 'full'（默认full）
    run_mode = (config.TRAIN_CONFIG.get('run_mode') or os.environ.get('RUN_MODE') or 'full').lower()
    config.RUN_MODE = run_mode
    if run_mode not in ['efficient', 'full']:
        run_mode = 'full'
        config.RUN_MODE = run_mode

    if run_mode == 'efficient':
        print("⚙️  运行模式: efficient（轻量稳定配置）")
        # 训练配置（更稳健）
        config.TRAIN_CONFIG.update({
            'learning_rate': 0.002,
            'scheduler_type': 'cosine',
            'min_lr': 1e-5,
            'patience': 15,
            'early_stopping_patience': 15,
            'gradient_clip': True,
            'gradient_clip_value': 1.0,
            'weight_decay': 1e-4,
        })
        # 数据配置（高吞吐）
        config.DATA_CONFIG.update({
            'batch_size': 128,
            'enable_bsmote': True,
            'bsmote_target_ratio': 0.7,
            'num_workers': 0,
        })
        # 轻量CNN/Transformer
        config.CNN_CONFIG.update({
            'conv_channels': [64, 128, 256],
            'kernel_sizes': [3, 3, 3],
            'pool_sizes': [2, 2, 2],
            'num_conv_layers': 3,
            'dropout_rate': 0.2,
            'use_batch_norm': True,
            'use_residual': True,
            'use_se_block': False,
            'use_attention': False,
            'activation': 'relu',
        })
        config.TRANSFORMER_CONFIG.update({
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'use_positional_encoding': True,
            'max_seq_length': 128,
            'activation': 'relu',
        })
    else:
        print("⚙️  运行模式: full（增强大模型配置）")
    
    return config

def create_loss_function(config, class_weights=None):
    """创建增强损失函数 - 支持类别权重和标签平滑"""
    if class_weights is not None:
        # 使用类别权重 + 标签平滑（降低“無”類權重干擾）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        class_weights = class_weights.to(device)
        try:
            # ID=5 為“無”類，適度下調其權重（不影響類別存在性）
            if class_weights.shape[0] >= 6:
                class_weights[5] = class_weights[5] * 0.6
        except Exception:
            pass
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.TRAIN_CONFIG.get('label_smoothing', 0.1)
        )
        print("✅ 使用加权交叉熵损失 + 标签平滑 (BSMOTE + 类别权重 + 标签平滑)")
    else:
        # 使用标签平滑的标准损失
        criterion = nn.CrossEntropyLoss(
            label_smoothing=config.TRAIN_CONFIG.get('label_smoothing', 0.1)
        )
        print("✅ 使用标准交叉熵损失 + 标签平滑")
    
    return criterion

def set_global_seed(seed: int = 42):
    """固定随机种子，保证可复现实验"""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"🔒 已固定随机种子: {seed}")
    except Exception as e:
        print(f"⚠️ 固定随机种子失败: {e}")

class ModelEMA:
    """简单的EMA实现：在optimizer.step()后调用update。
    - store/copy_to/restore 用于在验证/测试时临时使用EMA权重。
    """
    def __init__(self, model: nn.Module, decay: float = 0.995, device: str = None):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        self.device = device
        # 初始化shadow参数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone().detach()

    def store(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone().detach()

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

class OptimizedFluidIdentificationTrainer:
    """增强优化流体识别训练器 - 集成所有优化策略"""
    
    def __init__(self, config=None, train_loader=None, val_loader=None, class_weights=None):
        # 使用增强配置
        self.config = config if config is not None else create_enhanced_config()
        # 檢測 Tiny 模式（需在任何依賴之前）
        self._tiny_mode = str(os.environ.get('TINY_OVERFIT', '')).strip().lower() in ('1','true','yes','y')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 使用设备: {self.device}")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory_total > 0:
                    print(f"   显存: {gpu_memory_total / 1024**3:.1f}GB")
                else:
                    print("   显存: 未知")
            except Exception as e:
                print(f"   GPU信息获取失败: {e}")
                print("   显存: 未知")

        # 性能优化相关属性（必须在引用之前定义）
        self._enhanced_data_cache = {}  # 小波增强数据缓存
        self._preprocessing_cache_enabled = False  # 预处理缓存开关（默认为关，避免空缓存与额外开销）
        self._cache_max_size = 200  # 缓存最大批次数 (平衡内存使用和性能)
        self._cache_access_count = {}  # 缓存访问频率跟踪
        self._async_processing = True  # 异步处理开关
        self._pin_memory = True  # 内存钉扎优化
        self._gpu_wavelet_enabled = False  # GPU加速小波处理开关（默认关闭）
        self._bsmote_applied_once = False  # BSMOTE一次性旗标，避免重复重采样
        self.test_loader = None  # 测试集数据加载器
        self._wavelet_batch_size = self.config.DATA_CONFIG.get('batch_size', 32)  # 小波处理批次大小 - 与主批次大小保持一致
        self._precomputed_wavelets = {}  # 预计算小波变换
        self._auto_batch_size_adjustment = True  # 启用自动批次大小调整

        # 动态批次大小调整属性
        self._psutil_available = False
        if torch.cuda.is_available() and _PYNVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._psutil_available = True
                print("   ✅ pynvml已加载，启用GPU内存监控")
            except Exception as e:
                print(f"⚠️  pynvml初始化失败: {e}")
        elif not torch.cuda.is_available():
            print("   内存监控不可用: 非CUDA设备")
        else:
            print("   内存监控不可用: pynvml未安装或初始化失败")
        
        self._memory_monitor_interval = self.config.TRAIN_CONFIG.get('memory_monitor_interval', 10) # 每N个批次监控一次内存
        self._gpu_memory_threshold = self.config.TRAIN_CONFIG.get('gpu_memory_threshold', 0.8) # GPU内存使用率阈值 (80%)
        self._batch_size_adjust_factor = self.config.TRAIN_CONFIG.get('batch_size_adjust_factor', 0.5) # 批次大小调整因子
        self._current_batch_size = self.config.DATA_CONFIG.get('batch_size', 64)
        
        # GPU加速配置提示（在属性定义之后）
        if torch.cuda.is_available():
            if self._gpu_wavelet_enabled:
                print("⚡ GPU加速小波处理: 启用")
                print("   缓存大小: {} 批次".format(self._cache_max_size))
                print("   批次大小: {}".format(self._wavelet_batch_size))
            else:
                print("⚠️  GPU加速小波处理: 禁用")

        # Tiny 模式下：預先關閉 BSMOTE/增強/正則配置，確保後續數據載入與優化器一致
        if self._tiny_mode:
            try:
                self.config.DATA_CONFIG['enable_bsmote'] = False
                self.config.DATA_CONFIG['enable_frequency_masking'] = False
                self.config.DATA_CONFIG['enable_time_shift'] = False
                self.config.DATA_CONFIG['enable_noise_augmentation'] = False
                self.config.TRAIN_CONFIG['use_focal_loss'] = False
                self.config.TRAIN_CONFIG['label_smoothing'] = 0.0
                self.config.TRAIN_CONFIG['gradient_clip'] = False
                self.config.TRAIN_CONFIG['warmup_epochs'] = 0
                self.config.TRAIN_CONFIG['scheduler_type'] = 'none'
                self.config.TRAIN_CONFIG['weight_decay'] = 0.0
            except Exception:
                pass

        # 创建增强数据加载器
        if train_loader is not None and val_loader is not None:
            self.train_loader, self.val_loader = train_loader, val_loader
        else:
            self.train_loader, self.val_loader = self._create_enhanced_data_loaders()
        
        # 计算类别数和类别权重
        self.num_classes = self._detect_num_classes()
        
        # 更新模型配置中的类别数（鎖定為檢測到的數量）
        self.config.MODEL_CONFIG['num_classes'] = self.num_classes
        print(f"   ✅ 模型配置更新，类别数: {self.num_classes}")
        
        # 如果使用了优化的BSMOTE方案，类别权重已经在数据加载器中计算
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            print(f"✅ 使用优化BSMOTE计算的类别权重")
        else:
            self.class_weights = class_weights if class_weights is not None else self._calculate_class_weights()
        
        # 创建增强模型
        self.model = self._create_enhanced_model()

        # Tiny 模式下凍結 BatchNorm 統計，避免小樣本統計抖動
        if self._tiny_mode:
            try:
                self._set_batchnorm_eval(self.model)
                print("   ✅ Tiny: 已將 BatchNorm 設為 eval() 冻结统计")
            except Exception:
                pass

        # 可选：channels_last 内存格式，提高吞吐
        if self.config.TRAIN_CONFIG.get('use_channels_last', True):
            try:
                self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        
        # 创建优化器和调度器
        self.optimizer = self._create_enhanced_optimizer()
        self.scheduler = self._create_enhanced_scheduler()

        # Tiny 模式下設定更激進的學習率與0權重衰減
        if self._tiny_mode:
            try:
                tiny_lr = float(os.environ.get('TINY_LR', '0.01'))
                for g in self.optimizer.param_groups:
                    g['lr'] = tiny_lr
                    g['weight_decay'] = 0.0
                print(f"   🔧 Tiny: 強制設置LR={tiny_lr:.4f}, weight_decay=0.0")
            except Exception:
                pass
        
        # 创建损失函数
        # 优先使用weight_tensor（如果来自优化BSMOTE），否则使用class_weights
        weights_for_loss = getattr(self, 'weight_tensor', self.class_weights)
        
        # 检查权重信息
        if weights_for_loss is not None:
            if hasattr(weights_for_loss, 'cpu') and weights_for_loss.shape[0] != self.num_classes:
                print(f"⚠️  权重形状不匹配！实际: {weights_for_loss.shape[0]}, 预期: {self.num_classes}")
                weights_for_loss = None
        
        # Tiny 模式使用純CE
        if self._tiny_mode:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = create_loss_function(self.config, weights_for_loss)
        
        # 创建增强损失函数
        self.focal_loss = self._create_focal_loss()
        # 使用增强的主损失函数
        if self.config.TRAIN_CONFIG.get('use_focal_loss', False):
            self.criterion = self.focal_loss
            print("🎯 使用Focal Loss作为主损失函数")
        else:
            # 使用原始的损失函数
            pass
        # 調整 label_smoothing
        try:
            self.config.TRAIN_CONFIG['label_smoothing'] = 0.02
        except Exception:
            pass
        
        # 混合精度训练
        self.scaler = GradScaler() if self.config.TRAIN_CONFIG.get('use_amp', False) else None

        # 温度缩放（推理时可选）
        self._temperature: float = 1.0
        try:
            temp_env = os.environ.get('TEMP_SCALE', '').strip()
            if temp_env:
                self._temperature = max(0.5, float(temp_env))
        except Exception:
            pass

        # 创建EMA（可选）
        self.ema = None
        try:
            if bool(self.config.TRAIN_CONFIG.get('use_ema', False)):
                self.ema = ModelEMA(self.model, decay=float(self.config.TRAIN_CONFIG.get('ema_decay', 0.995)))
                print(f"   ✅ 已启用EMA (decay={float(self.config.TRAIN_CONFIG.get('ema_decay', 0.995)):.4f})")
        except Exception as _ema_e:
            print(f"   ⚠️ 启用EMA失败: {_ema_e}")
        
        # 训练状态
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = self.config.TRAIN_CONFIG.get('early_stopping_patience', 30)

        # 数值稳定性监控
        self.nan_counter = 0
        self.max_nan_threshold = 50  # 最大允许的NaN批次数

        # 训练监控
        self.epoch_nan_count = 0
        self.consecutive_nan_epochs = 0
        self.training_stability_score = 1.0
        
        # GradScaler状态跟踪
        self._grad_unscaled = False
        
        # 训练历史
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': [], 'epoch_time': [], 'gpu_memory': []
        }
        
        # 创建输出目录和统一可视化管理器
        self.output_dir = "./visualizations/training_analysis"
        self.viz_manager = UnifiedVisualizationManager(self.output_dir)

        # 创建TensorBoard写入器
        self.writer = self._create_tensorboard_writer()

        print("✅ 增强优化训练器初始化完成")
        print(f"   可视化管理器: {type(self.viz_manager).__name__}")
        print(f"📊 模型参数总数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 模型大小: {sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024:.2f} MB")
    
    def _create_enhanced_data_loaders(self):
        """创建增强数据加载器 - 集成小波包分解、BSMOTE和类别平衡"""
        print("🌊 创建增强数据加载器，集成小波包分解、BSMOTE和类别平衡...")
        
        try:
            # 创建小波包分解处理器
            self.wavelet_processor = None # 默认不实例化，仅在需要CPU回退时使用
            
            # 优化数据加载器参数（Windows 平台禁用多進程避免spawn卡頓）
            num_workers = self.config.DATA_CONFIG.get('num_workers', 0)
            if os.name == 'nt':
                num_workers = 0
                print("⚠️  Windows 平台關閉 DataLoader 多進程以避免啟動開銷與卡頓")
            elif self._async_processing and torch.cuda.is_available():
                # 使用异步处理优化 - 大幅增加工作进程以提升效率（僅限非Windows）
                num_workers = min(8, num_workers if num_workers > 0 else 4)
                print(f"⚡ 启用异步数据加载，工作进程数: {num_workers} (針對大批次優化)")
            else:
                num_workers = 0  # 单线程模式

            # 解析welldata路径（支持从任意工作目录运行）
            welldata_dir_cfg = self.config.DATA_PATHS.get('welldata_dir', 'welldata')
            if os.path.isabs(welldata_dir_cfg):
                welldata_dir_resolved = welldata_dir_cfg
            else:
                # 以项目根为基准解析
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                welldata_dir_resolved = os.path.normpath(os.path.join(project_root, welldata_dir_cfg))
            if not os.path.isdir(welldata_dir_resolved):
                print(f"⚠️  解析welldata目录失败: {welldata_dir_resolved}，退回相对路径 'welldata'")
                welldata_dir_resolved = 'welldata'

            # 创建基础数据加载器
            # 75% 重叠、时间窗口高低频差异通过 EnhancedWaveletProcessor 内部配置体现
            data_loaders = create_spectrogram_data_loaders(
                welldata_dir=welldata_dir_resolved,
                curve_names=self.config.DATA_CONFIG['curve_names'],
                scale_range=(5, 36),  # 扩展尺度带宽，提升判别信息
                time_window=32,       # 低频统一32采样点，满足原始64×64输出
                time_step=1,
                feature_size=(64, 64),
                batch_size=self.config.DATA_CONFIG['batch_size'],
                train_ratio=self.config.DATA_CONFIG['train_ratio'],
                val_ratio=self.config.DATA_CONFIG['val_ratio'],
                test_ratio=self.config.DATA_CONFIG['test_ratio'],
                enable_cleaning=self.config.DATA_CONFIG['enable_cleaning'],
                num_workers=num_workers,
                test_mode=False,
                # 启用文件内标签同义词统一与混合层保留（禁用井名映射）
                use_fluid_mapping=False,
                use_stratified_split=True,
                random_state=42,
                wavelet_config={
                    'curve_configs': {
                        # 高频曲线采用Morlet/CWT，分解层数等效深度5；时间窗口偏短（16），统一输出到64×64
                        'AC':  {'wavelet': 'morl', 'level': 5, 'mode': 'periodic', 'time_window': 16, 'output_size': (32, 32)},
                        'RT':  {'wavelet': 'morl', 'level': 5, 'mode': 'periodic', 'time_window': 16, 'output_size': (32, 32)},
                        # 低频曲线采用Sym8/WPD，分解层数4；时间窗口32；输出64×64
                        'GR':  {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                        'SP':  {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                        'DEN': {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                        'CNL': {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)}
                    },
                    # 75%重叠（步长=window*(1-0.75)）；若 force_time_step=True，则使用调用者传入的 time_step
                    'overlap_ratio': 0.75,
                    'force_time_step': True,
                    'label_mode': 'majority',
                    'majority_threshold': float(self.config.DATA_CONFIG.get('majority_threshold', 0.60)),
                    'threshold_discard': False,
                    'enable_rare_fallback': False
                }
            )
            
            # 修复批次大小不匹配问题：重新创建数据加载器并设置drop_last=True
            print(f"🔧 修复批次大小不匹配问题，批次大小: {self.config.DATA_CONFIG['batch_size']}")
            
            # 重新创建训练数据加载器，确保批次大小一致
            actual_batch_size = self.config.DATA_CONFIG['batch_size']
            train_kwargs = dict(
                batch_size=actual_batch_size,
                shuffle=True,
                num_workers=max(2, num_workers) if num_workers > 0 else 0,
                pin_memory=True,
                drop_last=True
            )
            if train_kwargs['num_workers'] > 0:
                train_kwargs['prefetch_factor'] = 4
                train_kwargs['persistent_workers'] = True
            # 可選：使用 BalancedBatchSampler 以批次內類別均衡；否則使用 WeightedRandomSampler/Shuffle
            use_balanced_batch = bool(self.config.DATA_CONFIG.get('use_balanced_batch', True))
            use_weighted_sampler = bool(self.config.DATA_CONFIG.get('use_weighted_sampler', False)) and not use_balanced_batch
            if use_balanced_batch:
                try:
                    base_dataset = data_loaders['train'].dataset
                    # 提取標籤
                    all_labels = []
                    for idx in range(len(base_dataset)):
                        sample = base_dataset[idx]
                        if isinstance(sample, (list, tuple)):
                            all_labels.append(int(sample[1]))
                    import numpy as _np
                    labels_np = _np.array(all_labels)
                    num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else self.num_classes

                    # 構建每類索引池
                    class_to_indices = {c: _np.where(labels_np == c)[0].tolist() for c in range(num_classes)}
                    for c in class_to_indices:
                        if len(class_to_indices[c]) == 0:
                            class_to_indices[c] = []

                    batch_size = actual_batch_size
                    per_class = max(1, batch_size // max(num_classes, 1))
                    import math as _math
                    # 調整以填滿整批
                    while per_class * num_classes > batch_size:
                        per_class -= 1
                    rest = batch_size - per_class * num_classes

                    import random as _random
                    def _balanced_batches():
                        # 隨機打亂每類索引
                        ptrs = {c: 0 for c in range(num_classes)}
                        for c in range(num_classes):
                            _random.shuffle(class_to_indices[c])
                        # 生成盡可能多的批次
                        total = len(labels_np)
                        produced = 0
                        while produced + batch_size <= total:
                            batch = []
                            for c in range(num_classes):
                                pool = class_to_indices[c]
                                if len(pool) == 0:
                                    continue
                                # 迴圈取樣
                                for _i in range(per_class):
                                    if ptrs[c] >= len(pool):
                                        _random.shuffle(pool)
                                        ptrs[c] = 0
                                    batch.append(pool[ptrs[c]])
                                    ptrs[c] += 1
                            # 分配剩餘名額
                            for _r in range(rest):
                                c = _r % num_classes
                                pool = class_to_indices[c]
                                if len(pool) == 0:
                                    continue
                                if ptrs[c] >= len(pool):
                                    _random.shuffle(pool)
                                    ptrs[c] = 0
                                batch.append(pool[ptrs[c]])
                                ptrs[c] += 1
                            if len(batch) == batch_size:
                                produced += batch_size
                                yield batch

                    class _BalancedBatchSampler(torch.utils.data.Sampler):
                        def __init__(self, make_iter):
                            self._make_iter = make_iter
                        def __iter__(self):
                            for b in self._make_iter():
                                yield b
                        def __len__(self):
                            return len(base_dataset) // batch_size

                    batch_sampler = _BalancedBatchSampler(_balanced_batches)
                    self.train_loader = DataLoader(
                        base_dataset,
                        batch_sampler=batch_sampler,
                        num_workers=train_kwargs['num_workers'],
                        pin_memory=train_kwargs['pin_memory'],
                        persistent_workers=train_kwargs.get('persistent_workers', False)
                    )
                    print("   ✅ 使用 BalancedBatchSampler 進行均衡批採樣")
                except Exception as _bb_e:
                    print(f"   ⚠️  BalancedBatchSampler 構建失敗，回退 shuffle: {_bb_e}")
                    self.train_loader = DataLoader(
                        data_loaders['train'].dataset,
                        **train_kwargs
                    )
            elif use_weighted_sampler:
                try:
                    base_dataset = data_loaders['train'].dataset
                    # 提取標籤
                    all_labels = []
                    for idx in range(len(base_dataset)):
                        sample = base_dataset[idx]
                        if isinstance(sample, (list, tuple)):
                            lbl = int(sample[1])
                        else:
                            continue
                        all_labels.append(lbl)
                    import numpy as _np
                    labels_np = _np.array(all_labels)
                    num_classes = int(labels_np.max()) + 1 if labels_np.size > 0 else self.num_classes
                    class_counts = _np.bincount(labels_np, minlength=num_classes).astype(_np.float32)
                    class_counts[class_counts == 0] = 1.0
                    class_weights_np = 1.0 / class_counts
                    sample_weights = class_weights_np[labels_np]
                    import torch as _torch
                    # 对优先类提高采样权重
                    try:
                        pri_classes = self.config.DATA_CONFIG.get('priority_classes', [1, 3])
                        pri_boost = float(self.config.DATA_CONFIG.get('sampler_priority_boost', 1.2))
                        for i, lbl in enumerate(labels_np):
                            if int(lbl) in pri_classes:
                                sample_weights[i] *= pri_boost
                    except Exception:
                        pass
                    sampler = _torch.utils.data.WeightedRandomSampler(weights=_torch.tensor(sample_weights, dtype=_torch.double), num_samples=len(base_dataset), replacement=True)
                    self.train_loader = DataLoader(
                        base_dataset,
                        batch_size=actual_batch_size,
                        sampler=sampler,
                        num_workers=train_kwargs['num_workers'],
                        pin_memory=train_kwargs['pin_memory'],
                        drop_last=train_kwargs['drop_last'],
                        persistent_workers=train_kwargs.get('persistent_workers', False)
                    )
                    print("   ✅ 使用 WeightedRandomSampler 進行批次均衡抽樣")
                except Exception as _ws_e:
                    print(f"   ⚠️  WeightedRandomSampler 構建失敗，回退 shuffle: {_ws_e}")
                    self.train_loader = DataLoader(
                        data_loaders['train'].dataset,
                        **train_kwargs
                    )
            else:
            self.train_loader = DataLoader(
                data_loaders['train'].dataset,
                **train_kwargs
            )
            print(f"   ✅ 训练数据加载器批次大小: {actual_batch_size}")
            
            # 重新创建验证数据加载器
            val_kwargs = dict(
                batch_size=actual_batch_size,
                shuffle=False,
                num_workers=max(2, num_workers) if num_workers > 0 else 0,
                pin_memory=True,
                drop_last=False
            )
            if val_kwargs['num_workers'] > 0:
                val_kwargs['prefetch_factor'] = 4
                val_kwargs['persistent_workers'] = True
            self.val_loader = DataLoader(
                data_loaders['val'].dataset,
                **val_kwargs
            )
            print(f"   ✅ 验证数据加载器批次大小: {actual_batch_size}")

            # 若存在测试集，创建测试数据加载器（與驗證一致參數）
            if 'test' in data_loaders and data_loaders['test'] is not None:
                test_kwargs = dict(
                batch_size=actual_batch_size,
                shuffle=False,
                    num_workers=max(2, num_workers) if num_workers > 0 else 0,
                pin_memory=True,
                    drop_last=False
                )
                if test_kwargs['num_workers'] > 0:
                    test_kwargs['prefetch_factor'] = 4
                    test_kwargs['persistent_workers'] = True
                self.test_loader = DataLoader(
                    data_loaders['test'].dataset,
                    **test_kwargs
                )
                print(f"   ✅ 测试数据加载器批次大小: {actual_batch_size}")
            
            # 验证批次大小一致性
            print(f"🔍 验证数据加载器批次大小:")
            print(f"   训练数据加载器批次大小: {self.train_loader.batch_size}")
            print(f"   验证数据加载器批次大小: {self.val_loader.batch_size}")
            print(f"   配置批次大小: {self.config.DATA_CONFIG['batch_size']}")
            
            # 检查第一个批次的实际大小
            try:
                first_batch = next(iter(self.train_loader))
                if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 2:
                    data, labels = first_batch[0], first_batch[1]
                    print(f"   第一个训练批次数据形状: {data.shape}")
                    print(f"   第一个训练批次标签形状: {labels.shape}")
            except Exception as e:
                print(f"⚠️  无法检查第一个批次: {e}")
            
            # 计算正确的批次数
            train_batches = len(self.train_loader)
            val_batches = len(self.val_loader)
            expected_train_batches = (len(self.train_loader.dataset) + actual_batch_size - 1) // actual_batch_size
            expected_val_batches = (len(self.val_loader.dataset) + actual_batch_size - 1) // actual_batch_size
            
            print(f"✅ 数据加载器修复完成")
            print(f"   训练批次数: {train_batches} (预期: {expected_train_batches})")
            print(f"   验证批次数: {val_batches} (预期: {expected_val_batches})")
            print(f"   批次大小: {actual_batch_size}")
            print(f"   训练样本数: {len(self.train_loader.dataset)}")
            print(f"   验证样本数: {len(self.val_loader.dataset)}")

            # 应用内存钉扎优化
            if self._pin_memory and torch.cuda.is_available():
                print("📌 启用内存钉扎优化")
                self.train_loader = self._optimize_data_loader(self.train_loader)
                self.val_loader = self._optimize_data_loader(self.val_loader)
            
            print(f"✅ 基础数据加载器创建成功")
            print(f"   训练集: {len(self.train_loader.dataset)} 样本")
            print(f"   验证集: {len(self.val_loader.dataset)} 样本")
            if 'test' in data_loaders:
                print(f"   测试集: {len(data_loaders['test'].dataset)} 样本")
            
            # 应用小波包分解数据增强 + BSMOTE + 类别权重方案
            if self.config.DATA_CONFIG.get('enable_bsmote', True) and not self._bsmote_applied_once:
                enable_pca = self.config.DATA_CONFIG.get('enable_pca', True)
                pca_status = "启用PCA降维" if enable_pca else "无PCA降维"
                print(f"🔄 应用小波包分解数据增强 + BSMOTE + 类别权重方案 ({pca_status})...")
                try:
                    from data.class_balance_handler import create_balanced_data_loader
                    
                    # 添加内存监控
                    memory_before = 0
                    try:
                        import psutil  # type: ignore
                        memory_before = psutil.virtual_memory().percent
                        print(f"   内存使用: {memory_before:.1f}%")
                    except Exception as e:
                        print(f"   内存监控不可用: {e}")
                    
                    # 应用小波包分解数据增强
                    if self.wavelet_processor is not None:
                        print("🌊 应用小波包分解数据增强...")
                        # 这里可以添加小波包分解数据增强逻辑
                        # 暂时保持原有BSMOTE逻辑
                    
                    balanced_loaders = create_balanced_data_loader(
                        data_loaders, 
                        method='bsmote', 
                        target_ratio=self.config.DATA_CONFIG.get('bsmote_target_ratio', 0.7),
                        config=self.config.DATA_CONFIG
                    )
                    
                    try:
                        import psutil  # type: ignore
                        memory_after = psutil.virtual_memory().percent
                        print(f"   处理后内存使用: {memory_after:.1f}%")
                    except Exception as e:
                        print(f"   内存监控不可用: {e}")
                    
                    print(f"✅ BSMOTE + 类别权重方案完成 ({pca_status})")
                    print(f"   平衡后训练集: {len(balanced_loaders['train'].dataset)} 样本")
                    print(f"   平衡后验证集: {len(balanced_loaders['val'].dataset)} 样本")
                    if 'test' in balanced_loaders:
                        print(f"   平衡后测试集: {len(balanced_loaders['test'].dataset)} 样本")
                    
                    # 存储类别权重用于训练
                    self.class_weights = balanced_loaders['class_weights']
                    self.weight_tensor = balanced_loaders['weight_tensor']
                    
                    # 重新创建数据加载器以使用平衡后的数据
                    balanced_train_loader = DataLoader(
                        balanced_loaders['train'].dataset,
                        batch_size=actual_batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=True,
                        persistent_workers=True if num_workers > 0 else False
                    )
                    
                    balanced_val_loader = DataLoader(
                        balanced_loaders['val'].dataset,
                        batch_size=actual_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=False,
                        persistent_workers=True if num_workers > 0 else False
                    )
                    
                    # 更新批次数量信息
                    print(f"✅ BSMOTE后数据加载器更新完成")
                    print(f"   平衡后训练批次数: {len(balanced_train_loader)}")
                    print(f"   平衡后验证批次数: {len(balanced_val_loader)}")
                    print(f"   平衡后训练样本数: {len(balanced_loaders['train'].dataset)}")
                    print(f"   平衡后验证样本数: {len(balanced_loaders['val'].dataset)}")
                    
                    # 一次性標記，避免後續重覆重採樣
                    self._bsmote_applied_once = True
                    return balanced_train_loader, balanced_val_loader
                    
                except KeyboardInterrupt:
                    print("⚠️  用户中断BSMOTE处理")
                    print("🔄 回退到基础数据加载器...")
                    return data_loaders['train'], data_loaders['val']
                except Exception as e:
                    print(f"⚠️  优化BSMOTE失败: {e}")
                    print("🔄 回退到基础数据加载器...")
                    import traceback
                    traceback.print_exc()
                    return data_loaders['train'], data_loaders['val']
            else:
                return data_loaders['train'], data_loaders['val']
            
        except Exception as e:
            print(f"❌ 数据加载器创建失败: {e}")
            raise
    
    def _detect_num_classes(self):
        """检测类别数"""
        try:
            # 優先從底層 dataset 屬性獲取標籤
            base_ds = None
            if hasattr(self.train_loader, 'dataset'):
                ds = self.train_loader.dataset
                base_ds = getattr(ds, 'dataset', ds)

            if base_ds is not None and hasattr(base_ds, 'labels_encoded'):
                labels_encoded = np.array(base_ds.labels_encoded)
                if labels_encoded.size > 0:
                    num_classes = int(labels_encoded.max()) + 1
                    print(f"   检测到类别数: {num_classes}")
                    return num_classes
            
            # 回退方案：從 dataset 逐樣本掃描一小部分獲取類別數
            try:
                uniq = set()
                ds_iter = self.train_loader.dataset
                sample_limit = min(len(ds_iter), 4096) if hasattr(ds_iter, '__len__') else 2048
                for idx in range(sample_limit):
                    try:
                        sample = ds_iter[idx]
                        if isinstance(sample, (list, tuple)):
                            lbl = int(sample[1])
                            uniq.add(lbl)
                        if len(uniq) >= 32:  # 安全上限
                            break
                    except Exception:
                        break
                if len(uniq) > 0:
                    num_classes = max(uniq) + 1
                    print(f"   检测到类别数(掃描): {num_classes}")
                    return num_classes
            except Exception:
                pass
            
            # 最後回退到默認（僅在無法讀取時）
            num_classes = get_num_fluid_classes()
            print(f"   使用默认类别数: {num_classes}")
            return num_classes
            
        except Exception as e:
            print(f"⚠️  类别数检测失败: {e}")
            num_classes = get_num_fluid_classes()
            print(f"   使用默认类别数: {num_classes}")
            return num_classes
    
    def _calculate_class_weights(self):
        """计算类别权重"""
        try:
            if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'dataset'):
                dataset_ref = self.train_loader.dataset.dataset
            else:
                dataset_ref = self.train_loader.dataset
            
            if dataset_ref is not None and hasattr(dataset_ref, 'labels_encoded'):
                labels_encoded = np.array(dataset_ref.labels_encoded)
                if labels_encoded.size > 0:
                    num_classes = int(labels_encoded.max()) + 1
                    counts = np.bincount(labels_encoded, minlength=num_classes).astype(np.float32)
                    # 基於有效樣本數的CB權重: w_c = (1 - beta) / (1 - beta^{n_c})
                    beta = float(self.config.DATA_CONFIG.get('cb_beta', 0.999))
                    counts_safe = np.clip(counts, 1.0, None)
                    cb_weights = (1.0 - beta) / (1.0 - np.power(beta, counts_safe))
                    # 規範化到平均為1，避免全局縮放影響LR
                    cb_weights = cb_weights / (cb_weights.mean() + 1e-12)
                    class_weights = torch.tensor(cb_weights, dtype=torch.float32, device=self.device)
                    
                    print(f"   ✅ 计算类别权重完成")
                    for i, weight in enumerate(class_weights):
                        try:
                            fluid_name = get_fluid_name(i)
                            print(f"     {fluid_name}: {weight:.3f}")
                        except:
                            print(f"     类别 {i}: {weight:.3f}")
                    
                    return class_weights
        except Exception as e:
            print(f"⚠️  类别权重计算失败: {e}")
        
        return None
    
    def _create_enhanced_model(self):
        """创建增强模型"""
        print("🏗️  创建增强模型...")
        
        try:
            # 模型選擇：支持簡化模型baseline（在Tiny或顯式指定時）
            use_simple = bool(self.config.MODEL_CONFIG.get('use_simple_model', False))
            if self._tiny_mode:
                use_simple = True

            if use_simple:
                from models.simple_fluid_model import create_simple_model
                model = create_simple_model(self.config).to(self.device)
                print("   ✅ 使用簡化模型 SimpleFluidIdentificationModel")
                # Tiny 模式下將所有 Dropout 關閉
                if self._tiny_mode:
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = 0.0
            else:
                # 确保模型配置包含所有必要参数
                from models.fluid_identification_model import FluidIdentificationModel
            model = FluidIdentificationModel(
                config=self.config
            ).to(self.device)
            
            # 增强权重初始化以提高数值稳定性
            # self._initialize_model_weights(model) # 权重初始化现在在模型内部处理

            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"   模型参数总数: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

            return model
            
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            raise

    def _initialize_model_weights(self, model):
        """增强模型权重初始化以提高数值稳定性和学习能力"""
        print("🔧 初始化模型权重...")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，增加gain值提高学习能力
                nn.init.xavier_uniform_(module.weight, gain=2.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.05)  # 增加偏差初始值

            elif isinstance(module, nn.Conv2d):
                # 使用He初始化，增加gain值
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.05)

            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm初始化，使用更积极的初始化
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm初始化
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, nn.MultiheadAttention):
                # 多头注意力初始化
                for param in module.parameters():
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param, gain=1.0)

        print("✅ 模型权重初始化完成")

    def _set_batchnorm_eval(self, model: nn.Module) -> None:
        """將模型中的 BatchNorm 層設為 eval 模式，穩定 Tiny 小樣本訓練。"""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                # 可選：減小動量，基本不更新統計
                try:
                    m.momentum = 0.0
                except Exception:
                    pass

    def _analyze_training_stability(self, loss, acc, f1, epoch):
        """分析训练稳定性"""
        # 检查当前epoch是否有NaN问题
        has_nan = torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))

        if has_nan:
            self.epoch_nan_count += 1
            self.consecutive_nan_epochs += 1
        else:
            self.consecutive_nan_epochs = 0

        # 计算稳定性分数
        if self.consecutive_nan_epochs > 0:
            self.training_stability_score = max(0.1, self.training_stability_score * 0.8)
        else:
            self.training_stability_score = min(1.0, self.training_stability_score * 1.02)

        # 稳定性报告
        if epoch % 10 == 0 or self.consecutive_nan_epochs > 2:
            print(f"📊 训练稳定性分析 (Epoch {epoch}):")
            print(f"   NaN计数: {self.epoch_nan_count}")
            print(f"   连续NaN轮数: {self.consecutive_nan_epochs}")
            print(f"   稳定性分数: {self.training_stability_score:.3f}")

            if self.consecutive_nan_epochs > 5:
                print("⚠️  检测到持续的数值不稳定，可能需要启用紧急模式")
            elif self.training_stability_score < 0.5:
                print("⚠️  训练稳定性较低，建议检查数据预处理")

    def _pre_training_stability_check(self):
        """训练前数值稳定性自检"""
        print("   🔍 检查数据加载器...")
        try:
            # 检查数据加载器
            sample_batch = next(iter(self.train_loader))
            if isinstance(sample_batch, (list, tuple)):
                data = sample_batch[0]
            else:
                data = sample_batch

            print(f"   ✅ 数据加载器正常，批次形状: {data.shape}")

            # 检查数据数值范围
            data_min, data_max = data.min(), data.max()
            print(f"   📊 数据范围: [{data_min:.6f}, {data_max:.6f}]")

            if torch.isnan(data).any() or torch.isinf(data).any():
                print("   ⚠️  检测到数据中存在NaN或Inf值")
                nan_count = torch.isnan(data).sum().item()
                inf_count = torch.isinf(data).sum().item()
                print(f"      NaN数量: {nan_count}, Inf数量: {inf_count}")
            else:
                print("   ✅ 数据数值正常")

        except Exception as e:
            print(f"   ❌ 数据加载器检查失败: {e}")
            import traceback
            traceback.print_exc()

        print("   🔍 检查模型前向传播...")
        try:
            # 创建测试输入
            test_input = torch.randn(2, self.config.DATA_CONFIG['num_curves'], 64, 64).to(self.device)
            self.model.eval()
            with torch.no_grad():
                test_output = self.model(test_input)
            print(f"   ✅ 模型前向传播正常，输出形状: {test_output.shape}")
            print(f"   📊 模型输出统计: mean={test_output.mean():.6f}, std={test_output.std():.6f}")

            if torch.isnan(test_output).any() or torch.isinf(test_output).any():
                print("   ⚠️  模型输出中存在NaN或Inf值")
                nan_count = torch.isnan(test_output).sum().item()
                inf_count = torch.isinf(test_output).sum().item()
                print(f"      NaN数量: {nan_count}, Inf数量: {inf_count}")
            else:
                print("   ✅ 模型输出数值正常")

        except Exception as e:
            print(f"   ❌ 模型前向传播检查失败: {e}")
            import traceback
            traceback.print_exc()

        print("   🔍 检查损失函数...")
        try:
            # 使用訓練器檢測到的實際類別數，避免 KeyError
            test_output = torch.randn(2, self.num_classes).to(self.device)
            test_labels = torch.randint(0, self.num_classes, (2,)).to(self.device)

            print(f"   测试输出形状: {test_output.shape}")
            print(f"   测试标签: {test_labels}")
            print(f"   测试输出范围: [{test_output.min():.6f}, {test_output.max():.6f}]")

            test_loss = self.criterion(test_output, test_labels)
            print(f"   ✅ 损失函数正常，测试损失: {test_loss:.6f}")

            if torch.isnan(test_loss) or torch.isinf(test_loss):
                print("   ⚠️  损失函数输出NaN或Inf值")
                print(f"      损失值详情: {test_loss.item()}")
            else:
                print("   ✅ 损失函数数值正常")

        except Exception as e:
            print(f"   ❌ 损失函数检查失败: {e}")
            import traceback
            traceback.print_exc()

        print("   🔍 检查优化器...")
        try:
            # 检查优化器参数
            print(f"   ✅ 优化器正常，参数组数量: {len(self.optimizer.param_groups)}")
            for i, group in enumerate(self.optimizer.param_groups):
                print(f"      组{i}: lr={group['lr']:.8f}, weight_decay={group.get('weight_decay', 0)}")
        except Exception as e:
            print(f"   ❌ 优化器检查失败: {e}")
            import traceback
            traceback.print_exc()

        print("   🔍 检查配置参数...")
        try:
            print(f"   类别数量: {self.num_classes}")
            print(f"   曲线数量: {self.config.DATA_CONFIG['num_curves']}")
            print(f"   批次大小: {self.config.DATA_CONFIG['batch_size']}")
            print(f"   学习率: {self.config.TRAIN_CONFIG['learning_rate']}")
            print(f"   权重衰减: {self.config.TRAIN_CONFIG['weight_decay']}")
            print(f"   使用Focal Loss: {self.config.TRAIN_CONFIG.get('use_focal_loss', False)}")
        except Exception as e:
            print(f"   ❌ 配置检查失败: {e}")

        print("   🔍 深度诊断小波处理器...")
        try:
            if hasattr(self, 'wavelet_processor') and self.wavelet_processor is not None:
                print("   小波处理器存在，进行深度检查...")
                # 创建简单的测试数据
                test_data = torch.randn(1, self.config.DATA_CONFIG['num_curves'], 64, 64).to(self.device)
                curve_names = self.config.DATA_CONFIG['curve_names']

                print(f"   测试数据形状: {test_data.shape}")
                print(f"   曲线名称: {curve_names}")

                # 测试CPU小波变换
                try:
                    result = self._cpu_wavelet_transform(test_data, curve_names)
                    print(f"   ✅ CPU小波变换正常，输出形状: {result.shape}")
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        print("   ⚠️  CPU小波变换产生NaN或Inf值")
                except Exception as wave_e:
                    print(f"   ❌ CPU小波变换失败: {wave_e}")
            else:
                print("   小波处理器未启用或不存在")
        except Exception as e:
            print(f"   ❌ 小波处理器诊断失败: {e}")

    def _create_enhanced_optimizer(self):
        """创建增强优化器"""
        print("⚙️  创建增强优化器...")
        
        # 获取优化器配置
        optimizer_type = self.config.TRAIN_CONFIG.get('optimizer_type', 'adamw')
        learning_rate = self.config.TRAIN_CONFIG['learning_rate']
        weight_decay = self.config.TRAIN_CONFIG['weight_decay']
        
        if optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(self.config.TRAIN_CONFIG.get('beta1', 0.9),
                       self.config.TRAIN_CONFIG.get('beta2', 0.999)),
                eps=self.config.TRAIN_CONFIG.get('eps', 1e-8)
            )
            print(f"   ✅ 使用AdamW优化器")
        elif optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(self.config.TRAIN_CONFIG.get('beta1', 0.9), 
                       self.config.TRAIN_CONFIG.get('beta2', 0.999)),
                eps=self.config.TRAIN_CONFIG.get('eps', 1e-8)
            )
            print(f"   ✅ 使用Adam优化器")
        else:
            # 默认使用AdamW
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"   ✅ 使用AdamW优化器（默认）")
    
        print(f"   学习率: {learning_rate}")
        print(f"   权重衰减: {weight_decay}")
        
        return optimizer
    
    def _create_enhanced_scheduler(self):
        """创建增强学习率调度器"""
        print("📈 创建增强学习率调度器...")
        
        scheduler_type = self.config.TRAIN_CONFIG.get('scheduler_type', 'cosine')
        
        if scheduler_type == 'cosine':
            # 余弦退火调度器 - 更稳定的学习率调度
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.TRAIN_CONFIG.get('epochs', 100),
                eta_min=self.config.TRAIN_CONFIG.get('min_lr', 1e-6)
            )
            print(f"   ✅ 使用余弦退火调度器")
            print(f"   📊 最小学习率: {self.config.TRAIN_CONFIG.get('min_lr', 1e-6)}")
        elif scheduler_type == 'onecycle':
            # OneCycle调度器 - 激进学习率变化
            try:
                steps = max(1, len(self.train_loader))
                scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config.TRAIN_CONFIG.get('max_lr', 0.003),
                    epochs=self.config.TRAIN_CONFIG.get('epochs', 100),
                    steps_per_epoch=steps,
                    pct_start=self.config.TRAIN_CONFIG.get('pct_start', 0.1),
                    div_factor=self.config.TRAIN_CONFIG.get('div_factor', 25),
                    final_div_factor=self.config.TRAIN_CONFIG.get('final_div_factor', 1000),
                    anneal_strategy='cos'
                )
                print(f"   ✅ 使用OneCycle调度器")
                print(f"   📊 最大学习率: {self.config.TRAIN_CONFIG.get('max_lr', 0.003)}")
                print(f"   📊 上升阶段比例: {self.config.TRAIN_CONFIG.get('pct_start', 0.1)}")
            except Exception as e:
                print(f"⚠️  OneCycle调度器创建失败: {e}")
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.TRAIN_CONFIG['epochs'],
                    eta_min=1e-6
                )
                print(f"   ✅ 回退到余弦退火调度器")
        else:
            # 支持 'none'：固定LR不做调度（用于Tiny）
            if scheduler_type == 'none':
                class _NoScheduler:
                    def step(self):
                        return
                scheduler = _NoScheduler()
                print("   ✅ 调度器禁用（固定LR）")
            else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.TRAIN_CONFIG['epochs'],
                eta_min=1e-6
            )
            print(f"   ✅ 使用余弦退火调度器")
        
        return scheduler
    
    def _create_focal_loss(self):
        """创建Focal Loss - 处理类别不平衡，优化参数"""
        import torch  # 在方法内部导入torch
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', class_weights=None):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                self.class_weights = class_weights
            
            def forward(self, inputs, targets):
                ce_loss = nn.functional.cross_entropy(
                    inputs, targets, reduction='none', weight=self.class_weights
                ) if self.class_weights is not None else nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        # 使用适度的类别权重（来自分布/BSMOTE），并对“无”类进行温和下调
        class_w = None
        try:
            if hasattr(self, 'weight_tensor') and self.weight_tensor is not None:
                cw = self.weight_tensor.detach().clone()
            else:
                # 若无BSMOTE权重，基于训练集标签分布估计权重
                cw = self._calculate_class_weights()
            # 校驗權重長度與 num_classes 一致
            if cw is not None and int(getattr(cw, 'shape', [0])[0]) == int(self.config.MODEL_CONFIG.get('num_classes', self.num_classes)):
                # 額外提升差油层(ID=3)權重以改善召回（支持环境变量覆盖）
                try:
                    if cw.shape[0] > 3:
                        cw = cw.clone()
                        boost_env = os.environ.get('CLASS3_BOOST', '').strip()
                        boost = float(boost_env) if boost_env else float(self.config.TRAIN_CONFIG.get('class3_boost', 1.5))
                        cw[3] = cw[3] * max(1.0, boost)
                        # 額外提升水层(ID=1)權重改善召回（A方案）
                        boost1_env = os.environ.get('CLASS1_BOOST', '').strip()
                        if boost1_env:
                            boost1 = float(boost1_env)
                            cw[1] = cw[1] * max(1.0, boost1)
                        # 若存在“无”类ID，则温和下调其权重，避免其主导梯度
                        none_id = self.config.MODEL_CONFIG.get('none_class_id', None)
                        try:
                            if none_id is not None and 0 <= int(none_id) < cw.shape[0]:
                                cw[int(none_id)] = cw[int(none_id)] * 0.6
                        except Exception:
                            pass
                        cw = cw / cw.mean()
                except Exception:
                    pass
                class_w = cw
            else:
                if cw is not None:
                    print(f"⚠️  類別權重維度不匹配，忽略: got={int(getattr(cw,'shape',[0])[0])}, expected={int(self.config.MODEL_CONFIG.get('num_classes', self.num_classes))}")
                class_w = None
        except Exception:
            class_w = None

        # 調低 gamma 提升對易錯類的梯度（避免過度抑制易樣本）
        gamma_value = float(self.config.TRAIN_CONFIG.get('focal_gamma', 1.5))
        focal_loss = FocalLoss(alpha=float(self.config.TRAIN_CONFIG.get('focal_alpha', 1.0)), gamma=gamma_value, class_weights=class_w)
        print(f"✅ 创建Focal Loss (α={self.config.TRAIN_CONFIG.get('focal_alpha', 1.0)}, γ={gamma_value})，使用适度类别权重")
        return focal_loss
    
    def _create_tensorboard_writer(self):
        """创建TensorBoard写入器"""
        try:
            if not _TB_AVAILABLE:
                return None
            # 修复路径拼写错误：tensorboard而不是tennsorboard
            log_dir = os.path.join(self.output_dir, 'tensorboard')
            # 确保目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 如果目录创建失败，使用备用路径
            if not os.path.exists(log_dir):
                safe_dir = os.path.join('./visualizations', 'tensorboard')
                os.makedirs(safe_dir, exist_ok=True)
                log_dir = safe_dir
                
            writer = SummaryWriter(log_dir=log_dir)
            print(f"📊 TensorBoard日志目录: {log_dir}")
            return writer
        except Exception as e:
            print(f"⚠️  TensorBoard创建失败: {e}")
            # 即使失败也返回None，不影响训练
            return None
    
    def train_epoch(self, epoch):
        """训练一个epoch - 优化版本"""
        self.model.train()
        if self.config.TRAIN_CONFIG.get('use_channels_last', True):
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        labels_buffer = []
        preds_buffer = []
        num_batches = 0
        accum_steps = max(1, int(self.config.TRAIN_CONFIG.get('grad_accum_steps', 1)))
        # 梯度累積初始化
        self.optimizer.zero_grad(set_to_none=True)

        # 性能监控
        import time
        start_time = time.time()
        data_loading_time = 0
        wavelet_processing_time = 0
        forward_time = 0
        backward_time = 0

        # 预处理整个训练数据集并缓存（如果启用）
        if self._preprocessing_cache_enabled and epoch == 1 and len(self._enhanced_data_cache) == 0:
            print("🔄 预处理整个训练数据集并缓存...")
            self._preprocess_all_data_cache()
            print(f"✅ 预处理缓存完成，共缓存 {len(self._enhanced_data_cache)} 个批次")

        # 计算正确的总批次数用于进度条
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.TRAIN_CONFIG["epochs"]}', total=total_batches)

        for batch_idx, batch in enumerate(pbar):
            try:
                batch_start_time = time.time()
                
                enhanced_data_from_cache = None
                if self._preprocessing_cache_enabled and batch_idx in self._enhanced_data_cache:
                    enhanced_data_from_cache = self._enhanced_data_cache[batch_idx]
                    self._cache_access_count[batch_idx] = self._cache_access_count.get(batch_idx, 0) + 1

                if enhanced_data_from_cache is not None:
                    # 如果数据已缓存，直接使用缓存数据
                    enhanced_data = enhanced_data_from_cache
                    data, labels = self._get_original_data_from_batch(batch)
                    # 规范化标签形状为一维 [B]
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                    elif labels.dim() > 1:
                        labels = labels.view(-1)
                    if torch.cuda.is_available():
                        labels = labels.to(self.device, non_blocking=True)
                    else:
                        labels = labels.to(self.device)
                else:
                    # 如果没有缓存，或者缓存被禁用，进行实时处理
                    try:
                        # 解包批次
                        data, labels = self._get_original_data_from_batch(batch)
                        # 规范化标签形状为一维 [B]
                        if labels.dim() == 0:
                            labels = labels.unsqueeze(0)
                        elif labels.dim() > 1:
                            labels = labels.view(-1)
                        
                        # 强制设备检查 - 确保数据在GPU上
                        if torch.cuda.is_available():
                            data = data.to(self.device, non_blocking=True)
                            labels = labels.to(self.device, non_blocking=True)
                        else:
                            data = data.to(self.device)
                            labels = labels.to(self.device)
                        
                        # 验证数据形状
                        if data.dim() != 4:
                            print(f"⚠️  数据维度不正确: {data.dim()}, 期望: 4")
                            continue
                        
                        if labels.dim() != 1:
                            print(f"⚠️  标签维度不正确: {labels.dim()}, 期望: 1")
                            continue
                        
                        if data.shape[0] != labels.shape[0]:
                            print(f"⚠️  批次大小不匹配: 数据 {data.shape[0]}, 标签 {labels.shape[0]}")
                            continue

                        # 实时小波包分解处理 - 每个epoch使用不同的变换
                        if self.wavelet_processor is not None:
                            # 获取曲线名称
                            curve_names = self.config.DATA_CONFIG['curve_names']
                            
                            # 为每个epoch添加随机种子，确保不同的变换
                            epoch_seed = epoch * 1000 + batch_idx
                            np.random.seed(epoch_seed)
                            
                            if self._gpu_wavelet_enabled:
                                enhanced_data = self._optimized_gpu_wavelet_transform(data, curve_names)
                            else:
                                # 回退到CPU处理
                                enhanced_data = self._cpu_wavelet_transform(data, curve_names)
                        else:
                            # 如果没有小波处理器，Tiny 下直接使用原始數據；否則使用簡單增強
                            if self._tiny_mode:
                                enhanced_data = data
                            else:
                            enhanced_data = self._simple_data_augmentation(data)
                        
                        # 确保数据在GPU上
                        if enhanced_data.device != self.device:
                            enhanced_data = enhanced_data.to(self.device)

                        # 如果启用了缓存，将实时处理后的数据存入缓存
                        if self._preprocessing_cache_enabled:
                            self._manage_smart_cache(batch_idx, enhanced_data)

                    except Exception as e:
                        import traceback
                        print(f"⚠️  批次 {batch_idx} 数据处理失败: {e}")
                        try:
                            print(f"   data.type={type(data)}, labels.type={type(labels)}")
                            if isinstance(data, torch.Tensor):
                                print(f"   data.shape={tuple(data.shape)}, device={data.device}, dtype={data.dtype}")
                            if isinstance(labels, torch.Tensor):
                                print(f"   labels.shape={tuple(labels.shape)}, device={labels.device}, dtype={labels.dtype}")
                        except Exception:
                            pass
                        traceback.print_exc()
                        continue

                # 数据加载时间统计
                data_loading_time += time.time() - batch_start_time
                
                # 快速防禦：若上一階段未產生 enhanced_data，回退為簡單增強
                if 'enhanced_data' not in locals() or enhanced_data is None:
                    try:
                        enhanced_data = data if self._tiny_mode else self._simple_data_augmentation(data)
                        if enhanced_data.device != self.device:
                            enhanced_data = enhanced_data.to(self.device)
                    except Exception as e:
                        print(f"⚠️  回退簡單增強也失敗: {e}")
                        continue

                # 动态批次大小调整
                if self._auto_batch_size_adjustment and self._psutil_available and batch_idx % self._memory_monitor_interval == 0:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 获取第一个GPU的句柄
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_used = info.used / (1024**3) # GB
                        gpu_memory_total = info.total / (1024**3) # GB
                        gpu_memory_utilization = gpu_memory_used / gpu_memory_total

                        # 显存使用率过高，减小批次大小
                        if gpu_memory_utilization > self._gpu_memory_threshold:
                            new_batch_size = max(1, int(self._current_batch_size * self._batch_size_adjust_factor))
                            if new_batch_size < self._current_batch_size:
                                print(f"⚠️  GPU内存使用率 {gpu_memory_utilization:.2f} 超过阈值 {self._gpu_memory_threshold:.2f}，将批次大小从 {self._current_batch_size} 调整为 {new_batch_size}")
                                self._current_batch_size = new_batch_size
                                self._update_data_loaders(self._current_batch_size)
                        # 显存使用率过低，尝试增大批次大小 (但不超过配置的原始批次大小)
                        elif gpu_memory_utilization < (self._gpu_memory_threshold * 0.75) and self._current_batch_size < self.config.DATA_CONFIG.get('batch_size', 64):
                            new_batch_size = min(self.config.DATA_CONFIG.get('batch_size', 64), int(self._current_batch_size / self._batch_size_adjust_factor))
                            if new_batch_size > self._current_batch_size:
                                print(f"💡 GPU内存使用率 {gpu_memory_utilization:.2f} 较低，将批次大小从 {self._current_batch_size} 调整为 {new_batch_size}")
                                self._current_batch_size = new_batch_size
                                self._update_data_loaders(self._current_batch_size)

                    except Exception as e:
                        print(f"⚠️  动态批次大小调整失败: {e}")
                        self._psutil_available = False # 禁用监控以避免重复错误

                # 已在上方完成小波處理與搬運，移除重複流程
                
                # 增强数据验证和数值稳定性处理
                if torch.isnan(enhanced_data).any() or torch.isinf(enhanced_data).any():
                    print(f"⚠️  检测到无效数据: NaN={torch.isnan(enhanced_data).any()}, Inf={torch.isinf(enhanced_data).any()}")
                    # 替换无效值为0
                    enhanced_data = torch.where(torch.isnan(enhanced_data), torch.zeros_like(enhanced_data), enhanced_data)
                    enhanced_data = torch.where(torch.isinf(enhanced_data), torch.zeros_like(enhanced_data), enhanced_data)

                # 添加数值范围检查
                data_min, data_max = enhanced_data.min(), enhanced_data.max()
                if data_max > 1e6 or data_min < -1e6:
                    print(f"⚠️  数据值范围异常: min={data_min:.2f}, max={data_max:.2f}，进行标准化")
                    enhanced_data = torch.clamp(enhanced_data, -1e6, 1e6)  # 限制数值范围

                # 前向传播时间统计
                forward_start_time = time.time()
                # 梯度在累積步內不清零，只在步進邊界清零

                # 前向传播 - 高级GPU利用率优化
                try:
                    if self.scaler is not None:
                        with autocast():
                            outputs = self.model(enhanced_data)
                            loss = self.criterion(outputs, labels)
                    else:
                        # 稳定优化：移除不稳定的梯度累积策略
                        outputs = self.model(enhanced_data)
                        loss = self.criterion(outputs, labels)
                    
                except Exception as forward_e:
                    print(f"⚠️  前向传播失败: {forward_e}")
                    continue

                forward_time += time.time() - forward_start_time

                # 增强NaN和数值稳定性检查
                loss_is_invalid = torch.isnan(loss) or torch.isinf(loss)
                output_is_invalid = torch.isnan(outputs).any() or torch.isinf(outputs).any()

                if loss_is_invalid or output_is_invalid:
                    self.nan_counter += 1
                    print(f"⚠️  检测到无效值 (第{self.nan_counter}次)")
                    print(f"   无效损失: NaN={torch.isnan(loss)}, Inf={torch.isinf(loss)}")
                    print(f"   无效输出: NaN={torch.isnan(outputs).any()}, Inf={torch.isinf(outputs).any()}")

                    # 如果NaN次数过多，启用应急模式
                    if self.nan_counter >= self.max_nan_threshold:
                        print(f"🚨  NaN次数过多 ({self.nan_counter})，启用应急模式")
                        self._activate_emergency_mode()

                    continue

                # 反向传播时间统计
                backward_start_time = time.time()

                # 反向传播與梯度累積
                raw_loss = loss
                loss_to_backward = loss / accum_steps
                do_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == total_batches)

                if self.scaler is not None:
                    self.scaler.scale(loss_to_backward).backward()
                    if do_step:
                        if self.config.TRAIN_CONFIG.get('gradient_clip', False):
                            if not getattr(self, '_grad_unscaled', False):
                                self.scaler.unscale_(self.optimizer)
                                self._grad_unscaled = True
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.config.TRAIN_CONFIG.get('gradient_clip_value', 5.0)
                            )
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                self.optimizer.zero_grad(set_to_none=True)
                                if self.config.TRAIN_CONFIG.get('debug_mode', False):
                                    print(f"⚠️  梯度裁剪后检测到无效梯度，重置梯度继续")
                                backward_time += time.time() - backward_start_time
                                continue
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        # EMA更新
                        if self.ema is not None:
                            try:
                                self.ema.update(self.model)
                            except Exception as _e:
                                pass
                        self.optimizer.zero_grad(set_to_none=True)
                        self._grad_unscaled = False
                else:
                    loss_to_backward.backward()
                    if do_step:
                        if self.config.TRAIN_CONFIG.get('gradient_clip', False):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config.TRAIN_CONFIG.get('gradient_clip_value', 1.0)
                            )
                        self.optimizer.step()
                        # EMA更新
                        if self.ema is not None:
                            try:
                                self.ema.update(self.model)
                            except Exception:
                                pass
                        self.optimizer.zero_grad(set_to_none=True)

                backward_time += time.time() - backward_start_time

                # 统计
                total_loss += raw_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # 收集樣本以計算macro F1（節流成本）
                if num_batches % 10 == 0:
                    labels_buffer.clear()
                    preds_buffer.clear()
                labels_buffer.append(labels.detach().cpu())
                preds_buffer.append(predicted.detach().cpu())
                # 近似即時F1（顯示用，不作為學習判據）
                f1 = (predicted == labels).float().mean().item()
                total_f1 += f1
                num_batches += 1

                # 优化进度条更新 - 只在关键批次更新以提升性能
                if num_batches % 50 == 0:  # 每50个批次更新一次，减少开销
                    pbar.set_postfix({
                        'Loss': f'{raw_loss.item():.4f}',
                        'Acc': f'{100. * correct / total:.2f}%',
                        'F1': f'{f1:.4f}'
                    })
                
            except Exception as e:
                print(f"⚠️  训练批次 {batch_idx} 失败: {e}")
                continue

        # 计算平均指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        # 計算macro F1（用累積子樣本降低CPU成本）
        try:
            if labels_buffer and preds_buffer:
                import numpy as np
                from sklearn.metrics import f1_score
                y_true = torch.cat(labels_buffer).numpy()
                y_pred = torch.cat(preds_buffer).numpy()
                macro_f1 = f1_score(y_true, y_pred, average='macro')
            else:
                macro_f1 = total_f1 / num_batches if num_batches > 0 else 0
        except Exception:
            macro_f1 = total_f1 / num_batches if num_batches > 0 else 0
        avg_f1 = macro_f1

        # 训练稳定性分析
        self._analyze_training_stability(avg_loss, avg_acc, avg_f1, epoch)

        # 性能报告
        epoch_time = time.time() - start_time
        # 优化训练日志，显示关键性能指标
        if epoch % 5 == 0 or epoch <= 5:  # 每5个epoch或前5个epoch显示统计
            cache_hit_rate, cache_hits = self._get_cache_hit_stats()
            print(f"   💾 缓存命中率: {cache_hit_rate:.1f}% ({cache_hits}/{len(self._enhanced_data_cache) if self._enhanced_data_cache else 0})")
            print(f"   ⏱️  Epoch时间: {epoch_time:.1f}s")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                try:
                    # 正确的GPU利用率计算
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_usage_percent = (gpu_memory / gpu_total) * 100 if gpu_total > 0 else 0
                except Exception as e:
                    gpu_memory = 0
                    gpu_total = 1  # 避免除零错误
                    gpu_usage_percent = 0
                    print(f"⚠️  GPU信息计算失败: {e}")
                
                # 更准确的GPU利用率估算（基于实际批次大小和批次数）
                actual_batch_size = self.config.DATA_CONFIG.get('batch_size', 32)
                actual_total_batches = len(self.train_loader)
                
                # 基于实际批次数和批次大小计算GPU利用率
                if actual_batch_size >= 256:
                    gpu_util_estimate = min(95, gpu_usage_percent + 25)  # 大批次通常有高GPU利用率
                elif actual_batch_size >= 128:
                    gpu_util_estimate = min(85, gpu_usage_percent + 15)  # 128批次平衡利用率
                elif actual_batch_size >= 64:
                    gpu_util_estimate = min(75, gpu_usage_percent + 10)  # 中等批次
                else:
                    gpu_util_estimate = min(75, gpu_usage_percent + 15)
                
                # 显示正确的批次信息
                print(f"   📊 批次信息: 大小={actual_batch_size}, 总数={actual_total_batches}")
                
                print(f"   🚀 GPU利用率: ~{gpu_util_estimate:.1f}% (内存: {gpu_memory:.1f}GB / {gpu_total:.1f}GB, {gpu_usage_percent:.1f}%)")
                
                # RTX A2000专用监控
                if gpu_usage_percent > 90:
                    print(f"   ⚠️  显存使用率过高: {gpu_usage_percent:.1f}%, 建议减少批次大小")
                elif gpu_usage_percent < 60:
                    print(f"   💡 显存使用率较低: {gpu_usage_percent:.1f}%, 可考虑增加批次大小")
                
                # 显示GPU性能统计
                if hasattr(torch.cuda, 'memory_stats'):
                    stats = torch.cuda.memory_stats()
                    print(f"   📊 GPU统计: 分配次数={stats.get('num_alloc_retries', 0)}, OOM次数={stats.get('num_ooms', 0)}")
                
                # 定期清理GPU内存
                if epoch % 10 == 0:
                    torch.cuda.empty_cache()
                    print(f"   🧹 已清理GPU内存缓存")

        return avg_loss, avg_acc, avg_f1

    def _get_cache_hit_stats(self):
        """获取缓存命中率统计 - 修复计算逻辑"""
        if not hasattr(self, '_enhanced_data_cache') or not self._enhanced_data_cache:
            return 0.0, 0

        # 计算实际缓存命中率
        total_batches = len(self._enhanced_data_cache)
        if total_batches == 0:
            return 0.0, 0

        # 检查有多少批次被实际使用（访问次数 > 0）
        if hasattr(self, '_cache_access_count') and self._cache_access_count:
            cache_hits = sum(1 for count in self._cache_access_count.values() if count > 0)
            hit_rate = cache_hits / total_batches if total_batches > 0 else 0.0
        else:
            # 如果没有访问计数，假设所有缓存的批次都被使用
            cache_hits = total_batches
            hit_rate = 1.0

        return hit_rate, cache_hits

    def _manage_smart_cache(self, batch_idx, data):
        """优化的智能缓存管理 - 修复缓存命中率问题"""
        if not self._preprocessing_cache_enabled:
            return

        # 记录访问频率
        if batch_idx not in self._cache_access_count:
            self._cache_access_count[batch_idx] = 0
        self._cache_access_count[batch_idx] += 1

        # 如果缓存已满，使用LRU策略移除最少使用的项
        if len(self._enhanced_data_cache) >= self._cache_max_size:
            # 找到访问次数最少的批次
            if self._cache_access_count:
                least_used_batch = min(self._cache_access_count.keys(),
                                     key=lambda x: self._cache_access_count.get(x, 0))
                if least_used_batch in self._enhanced_data_cache:
                    del self._enhanced_data_cache[least_used_batch]
                    del self._cache_access_count[least_used_batch]

        # 优化：存储到GPU而不是CPU，避免重复传输
        if data.device.type == 'cuda':
            # 数据已在GPU上，直接存储
            self._enhanced_data_cache[batch_idx] = data.clone().detach()
        else:
            # 数据在CPU上，传输到GPU后存储
            self._enhanced_data_cache[batch_idx] = data.to(self.device, non_blocking=True)

    def _print_performance_report(self, epoch, epoch_time, data_loading_time,
                                wavelet_processing_time, forward_time, backward_time, num_batches):
        """简化的性能报告 - 只在必要时显示关键信息"""
        # 只在第一个epoch或性能严重问题时显示
        if epoch == 1 and wavelet_processing_time > epoch_time * 0.5:
            print(f"\n⚠️  性能警告: 小波处理时间过长 ({wavelet_processing_time:.1f}s)")
            print("   正在应用自动优化...")
            self._apply_critical_optimization()

    def _auto_optimize_performance(self, epoch: int, issue_type: str):
        """自动性能优化 - 简化的执行"""
        if epoch == 1:  # 只在第一个epoch进行自动优化
            if issue_type == 'wavelet_bottleneck':
                # 禁用小波数据增强
                if hasattr(self.config.DATA_CONFIG, 'enable_wavelet_augmentation'):
                    self.config.DATA_CONFIG['enable_wavelet_augmentation'] = False
                # 增大缓存大小
                if hasattr(self, '_cache_max_size'):
                    self._cache_max_size = min(300, self._cache_max_size * 2)

            elif issue_type == 'small_batch_size':
                # 尝试增大批次大小
                if hasattr(self.config.DATA_CONFIG, 'batch_size'):
                    old_batch_size = self.config.DATA_CONFIG['batch_size']
                    new_batch_size = min(16, old_batch_size * 2)
                    if new_batch_size != old_batch_size:
                        self.config.DATA_CONFIG['batch_size'] = new_batch_size

            elif issue_type == 'high_cpu_overhead':
                # 禁用GPU加速小波处理，减少传输开销
                if hasattr(self, '_gpu_wavelet_enabled'):
                    self._gpu_wavelet_enabled = False

    def _activate_emergency_mode(self):
        """激活紧急数值稳定性优化模式"""
        print("🚨 激活紧急数值稳定性优化模式...")

        # 方案1: 完全禁用小波处理
        if hasattr(self, 'wavelet_processor'):
            self.wavelet_processor = None
            print("   ❌ 已禁用小波处理")

        # 方案2: 简化数据增强
        if hasattr(self.config.DATA_CONFIG, 'enable_augmentation'):
            self.config.DATA_CONFIG['enable_augmentation'] = False
            print("   ❌ 已禁用数据增强")

        # 方案3: 大幅降低学习率
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.01  # 降低100倍
                print(f"   📉 学习率降低至: {param_group['lr']}")

        # 方案4: 禁用缓存
        if hasattr(self, '_preprocessing_cache_enabled'):
            self._preprocessing_cache_enabled = False
            print("   ❌ 已禁用预处理缓存")

        # 方案5: 启用更严格的梯度裁剪
        if hasattr(self.config.TRAIN_CONFIG, 'gradient_clip_value'):
            self.config.TRAIN_CONFIG['gradient_clip_value'] = 0.5  # 更严格的梯度裁剪
            print("   ✂️  启用严格梯度裁剪 (0.5)")

        # 方案6: 重置NaN计数器
        self.nan_counter = 0
        print("✅ 紧急数值稳定性模式激活完成")

    def _apply_critical_optimization(self):
        """应用紧急性能优化 - 简化的执行"""
        # 方案1: 完全禁用小波处理
        if hasattr(self, 'wavelet_processor'):
            self.wavelet_processor = None

        # 方案2: 简化数据增强
        if hasattr(self.config.DATA_CONFIG, 'enable_augmentation'):
            self.config.DATA_CONFIG['enable_augmentation'] = False

        # 方案3: 增大批次大小
        if hasattr(self.config.DATA_CONFIG, 'batch_size'):
            old_batch_size = self.config.DATA_CONFIG['batch_size']
            self.config.DATA_CONFIG['batch_size'] = min(16, old_batch_size * 2)

        # 方案4: 禁用缓存
        if hasattr(self, '_preprocessing_cache_enabled'):
            self._preprocessing_cache_enabled = False

    def _preprocess_all_data_cache(self):
        """预处理整个训练数据集 - GPU加速全量缓存"""
        if not self._preprocessing_cache_enabled or self.wavelet_processor is None:
            return

        print("🔄 GPU加速全量预处理小波增强数据缓存...")

        # 不清空缓存，如果已有缓存则跳过
        if len(self._enhanced_data_cache) > 0:
            print(f"✅ 缓存已存在，跳过预处理 (已有 {len(self._enhanced_data_cache)} 个批次)")
            return

        # 获取训练数据集的总大小
        try:
            total_samples = len(self.train_loader.dataset)
            batch_size = self.config.DATA_CONFIG.get('batch_size', 4)
            total_batches = len(self.train_loader)  # 使用实际的数据加载器批次数
            print(f"📊 训练数据集: {total_samples} 样本, {total_batches} 批次")
        except:
            total_batches = float('inf')  # 未知大小

        # 预处理所有数据
        curve_names = self.config.DATA_CONFIG['curve_names']
        cache_count = 0

        try:
            # 临时设置数据加载器为非随机模式以获得一致的结果
            original_shuffle = getattr(self.train_loader, 'shuffle', None)
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'shuffle'):
                self.train_loader.sampler.shuffle = False

            start_time = time.time()

            # 批量处理策略：一次性处理多个批次以提高效率
            batch_buffer = []
            batch_indices = []

            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # 解包批次
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            data, labels, _ = batch
                        elif len(batch) == 2:
                            data, labels = batch
                        else:
                            continue
                    else:
                        continue

                    # 收集批次数据
                    batch_buffer.append((data, labels))
                    batch_indices.append(batch_idx)

                    # 当收集到足够的批次或达到缓存限制时，进行批量处理
                    if (len(batch_buffer) >= 10 or  # 每10个批次处理一次
                        cache_count + len(batch_buffer) >= self._cache_max_size):

                        # 批量GPU处理
                        processed_batches = self._batch_gpu_wavelet_transform(batch_buffer, curve_names)

                        # 存储到缓存中（优化：保持GPU存储）
                        for i, enhanced_data in enumerate(processed_batches):
                            batch_idx_actual = batch_indices[i]
                            # 确保数据在GPU上存储，避免CPU-GPU传输
                            if enhanced_data.device.type == 'cuda':
                                self._enhanced_data_cache[batch_idx_actual] = enhanced_data.clone().detach()
                            else:
                                self._enhanced_data_cache[batch_idx_actual] = enhanced_data.to(self.device, non_blocking=True)

                        cache_count += len(processed_batches)

                        # 清空缓冲区
                        batch_buffer = []
                        batch_indices = []

                        # 显示进度
                        elapsed = time.time() - start_time
                        print(f"   已缓存 {cache_count} 批次，用时 {elapsed:.1f}s")

                        # 如果达到缓存限制，停止
                        if cache_count >= self._cache_max_size:
                            break

                except Exception as e:
                    print(f"⚠️  预处理批次 {batch_idx} 失败: {e}")
                    continue

            # 处理剩余的批次
            if batch_buffer:
                try:
                    processed_batches = self._batch_gpu_wavelet_transform(batch_buffer, curve_names)
                    for i, enhanced_data in enumerate(processed_batches):
                        batch_idx_actual = batch_indices[i]
                        # 确保数据在GPU上存储
                        if enhanced_data.device.type == 'cuda':
                            self._enhanced_data_cache[batch_idx_actual] = enhanced_data.clone().detach()
                        else:
                            self._enhanced_data_cache[batch_idx_actual] = enhanced_data.to(self.device, non_blocking=True)
                    cache_count += len(processed_batches)
                except Exception as e:
                    print(f"⚠️  处理剩余批次失败: {e}")

            total_time = time.time() - start_time
            print(f"✅ GPU加速全量预处理完成，共缓存 {cache_count} 个批次，用时 {total_time:.1f}s")
            print(f"   平均每批处理时间: {total_time/cache_count:.3f}s" if cache_count > 0 else "")

            # 恢复原始的shuffle设置
            if original_shuffle is not None and hasattr(self.train_loader, 'sampler'):
                self.train_loader.sampler.shuffle = original_shuffle

        except Exception as e:
            print(f"⚠️  GPU加速全量预处理缓存失败: {e}")
            import traceback
            traceback.print_exc()
            self._enhanced_data_cache = {}

    def _batch_gpu_wavelet_transform(self, batch_buffer, curve_names):
        """批量GPU小波变换处理"""
        processed_batches = []

        for data, labels in batch_buffer:
            try:
                # 确保数据在GPU上
                data = data.to(self.device)

                # 使用优化的GPU处理
                enhanced_data = self._optimized_gpu_wavelet_transform(data, curve_names)
                processed_batches.append(enhanced_data)

            except Exception as e:
                print(f"⚠️  批量GPU处理失败: {e}")
                # 回退到CPU处理
                try:
                    enhanced_data = self._cpu_wavelet_transform(data, curve_names)
                    processed_batches.append(enhanced_data)
                except Exception as cpu_e:
                    print(f"⚠️  CPU回退也失败: {cpu_e}")
                    # 创建零张量作为fallback
                    batch_size, num_curves, height, width = data.shape
                    fallback_data = torch.zeros((batch_size, num_curves, 64, 64), device=self.device)
                    processed_batches.append(fallback_data)

        return processed_batches

    def _optimized_gpu_wavelet_transform(self, data, curve_names):
        """高度优化的GPU小波变换 - 解决计算瓶颈"""
        try:
            # 策略：使用向量化操作和内存优化
            batch_size, num_curves, height, width = data.shape

            # 预分配输出tensor（使用连续内存布局）
            enhanced_data = torch.zeros((batch_size, num_curves, 64, 64), 
                                      device=self.device, dtype=data.dtype)

            # 批量处理所有曲线 - 使用向量化操作
            for j, curve_name in enumerate(curve_names):
                try:
                    # 获取曲线数据 [batch_size, height, width]
                    curve_batch = data[:, j, :, :]  # [batch_size, height, width]

                    # 使用优化的GPU增强处理
                    enhanced_curve = self._vectorized_gpu_enhance(curve_batch, curve_name)

                    # 存储结果
                    enhanced_data[:, j, :, :] = enhanced_curve

                except Exception as curve_e:
                    print(f"⚠️  处理曲线 {curve_name} 失败: {curve_e}")
                    # 使用原始数据作为fallback
                    enhanced_data[:, j, :, :] = curve_batch

            return enhanced_data

        except Exception as e:
            print(f"⚠️  GPU处理失败: {e}")
            # 回退到GPU简单处理，避免CPU转换
            return self._simple_gpu_enhance(data, curve_names)

    def _vectorized_gpu_enhance(self, curve_batch, curve_name):
        """向量化GPU增强处理 - 纯GPU实现，避免CPU回退"""
        try:
            # 直接使用GPU进行频率域处理，不依赖CPU小波库
            # 这比CPU小波处理快得多，且完全在GPU上运行
            
            # 获取曲线配置，添加默认值
            if not hasattr(self, 'wavelet_processor') or self.wavelet_processor is None:
                # 如果没有小波处理器，直接使用GPU简单处理
                return self._simple_resize_enhance(curve_batch)
                
            curve_config = self.wavelet_processor.curve_configs.get(curve_name, self.wavelet_processor.curve_configs.get('GR', {}))
            
            # 确保配置有必要的参数
            if 'type' not in curve_config:
                curve_config['type'] = 'low_freq'
            if 'cutoff_freq' not in curve_config:
                curve_config['cutoff_freq'] = 0.5

            # 纯GPU增强策略 - 使用FFT替代小波变换
            if curve_config['type'] == 'high_freq':
                # 高频处理：GPU FFT频率增强
                enhanced = self._vectorized_high_freq_enhance(curve_batch, curve_config)
            else:
                # 低频处理：GPU FFT时频变换
                enhanced = self._vectorized_low_freq_enhance(curve_batch, curve_config)

            return enhanced

        except Exception as e:
            print(f"⚠️  GPU增强处理失败: {e}")
            # 回退到GPU简单处理，避免CPU转换
            return self._simple_resize_enhance(curve_batch)

    def _simple_resize_enhance(self, curve_batch):
        """简单的resize增强处理 - 作为回退方案"""
        try:
            return torch.nn.functional.interpolate(
                curve_batch.unsqueeze(1), 
                size=(64, 64), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        except Exception as e:
            print(f"⚠️  简单resize增强失败: {e}")
            return curve_batch

    def _vectorized_high_freq_enhance(self, curve_batch, curve_config):
        """向量化高频增强处理"""
        try:
            batch_size, height, width = curve_batch.shape
            
            # 使用向量化的FFT处理
            # 对每个样本进行2D FFT
            fft_data = torch.fft.fft2(curve_batch, dim=(-2, -1))
            
            # 创建频率掩码（向量化操作）
            freq_mask = self._create_frequency_mask(height, width, curve_config, device=curve_batch.device)
            
            # 应用频率掩码 - 确保维度匹配
            if freq_mask.dim() == 2:
                freq_mask = freq_mask.unsqueeze(0)  # 添加批次维度
            masked_fft = fft_data * freq_mask
            
            # 逆FFT
            enhanced = torch.fft.ifft2(masked_fft, dim=(-2, -1)).real

            # 根据filter_type和sigma进行后处理
            filter_type = curve_config.get('filter_type', 'high_pass')
            sigma = curve_config.get('sigma', 0.1)

            if filter_type == 'high_pass':
                # 锐化处理，模拟高频小波的特性
                # 简单示例：可以通过与原始数据差异叠加来实现
                # enhanced = curve_batch + (enhanced - curve_batch) * (1 + sigma * 5) # 增强差异
                # 更高级的锐化操作，例如使用拉普拉斯算子
                # blurred = F.conv2d(curve_batch.unsqueeze(1), weight=torch.ones(1, 1, 3, 3).to(curve_batch.device)/9, padding=1)
                # enhanced = curve_batch + (curve_batch - blurred.squeeze(1)) * sigma * 5
                # 这里我们保持简单，让频率掩码本身来控制
                pass 
            elif filter_type == 'low_pass':
                # 平滑处理，模拟低频小波的特性
                # 类似地，频率掩码本身已经包含了平滑效果
                pass
            
            return enhanced
        except Exception as e:
            print(f"⚠️  向量化高频增强失败: {e}")
            return curve_batch

    def _vectorized_low_freq_enhance(self, curve_batch, curve_config):
        """向量化低频增强处理"""
        try:
            batch_size, height, width = curve_batch.shape
            
            # 使用向量化的时频变换
            # 对时间维度进行FFT
            time_fft = torch.fft.fft(curve_batch, dim=-1)
            
            # 创建时频掩码
            time_freq_mask = self._create_time_freq_mask(height, width, curve_config, device=curve_batch.device)
            
            # 应用掩码 - 确保维度匹配
            if time_freq_mask.dim() == 2:
                time_freq_mask = time_freq_mask.unsqueeze(0)  # 添加批次维度
            masked_time_fft = time_fft * time_freq_mask
            
            # 逆FFT
            enhanced = torch.fft.ifft(masked_time_fft, dim=-1).real
            
            # 归一化
            enhanced = (enhanced - enhanced.mean(dim=(-2, -1), keepdim=True)) / (enhanced.std(dim=(-2, -1), keepdim=True) + 1e-8)
            
            # 调整到目标尺寸
            if enhanced.shape[-2:] != (64, 64):
                enhanced = torch.nn.functional.interpolate(
                    enhanced.unsqueeze(1), 
                    size=(64, 64), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️  向量化低频增强失败: {e}")
            # 回退到简单处理
            return self._simple_resize_enhance(curve_batch)

    def _create_frequency_mask(self, height, width, curve_config, device):
        """创建频率掩码（向量化）- 带缓存优化"""
        try:
            # 创建缓存键
            cache_key = f"freq_mask_{height}_{width}_{curve_config.get('cutoff_freq', 0.5)}"
            
            # 检查缓存
            if not hasattr(self, '_freq_mask_cache'):
                self._freq_mask_cache = {}
            
            if cache_key in self._freq_mask_cache:
                return self._freq_mask_cache[cache_key]
            
            # 创建频率网格
            freq_y = torch.fft.fftfreq(height, device=device)
            freq_x = torch.fft.fftfreq(width, device=device)
            freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
            
            # 计算频率幅度
            freq_magnitude = torch.sqrt(freq_grid_y**2 + freq_grid_x**2)
            
            # 创建掩码（基于曲线配置） - 引入高斯滤波器和类型控制
            cutoff_freq = curve_config.get('cutoff_freq', 0.25)  # 默认截止频率调整
            filter_type = curve_config.get('filter_type', 'low_pass') # 默认低通
            sigma = curve_config.get('sigma', 0.1) # 高斯滤波器的标准差

            if filter_type == 'low_pass':
                # 高斯低通滤波器
                mask = torch.exp(-(freq_magnitude**2) / (2 * sigma**2))
                # 确保在截止频率处有陡峭的衰减，模拟小波特性
                mask = mask * (freq_magnitude <= cutoff_freq).float() + (freq_magnitude > cutoff_freq).float() * torch.exp(-((freq_magnitude - cutoff_freq)**2) / (2 * (sigma*0.5)**2)) # 平滑过渡

            elif filter_type == 'high_pass':
                # 高斯高通滤波器
                mask = 1 - torch.exp(-(freq_magnitude**2) / (2 * sigma**2))
                # 确保在截止频率处有陡峭的衰减，模拟小波特性
                mask = mask * (freq_magnitude >= cutoff_freq).float() + (freq_magnitude < cutoff_freq).float() * (1 - torch.exp(-((freq_magnitude - cutoff_freq)**2) / (2 * (sigma*0.5)**2))) # 平滑过渡

            else:
                # 默认行为（之前的硬截止滤波器）
                mask = (freq_magnitude <= cutoff_freq).float()
            
            # 缓存结果
            self._freq_mask_cache[cache_key] = mask
            
            return mask
        except Exception as e:
            print(f"⚠️  创建频率掩码失败: {e}")
            # 返回默认掩码
            return torch.ones((height, width), device=device)

    def _create_time_freq_mask(self, height, width, curve_config, device):
        """创建时频掩码（向量化）- 带缓存优化"""
        try:
            # 创建缓存键
            cache_key = f"time_freq_mask_{height}_{width}_{curve_config.get('cutoff_freq', 0.3)}"
            
            # 检查缓存
            if not hasattr(self, '_time_freq_mask_cache'):
                self._time_freq_mask_cache = {}
            
            if cache_key in self._time_freq_mask_cache:
                return self._time_freq_mask_cache[cache_key]
            
            # 创建时间频率网格
            time_freq = torch.fft.fftfreq(width, device=device)
            time_freq_grid = time_freq.unsqueeze(0).expand(height, -1)
            
            # 创建掩码
            cutoff_freq = curve_config.get('cutoff_freq', 0.3)
            mask = (torch.abs(time_freq_grid) <= cutoff_freq).float()
            
            # 缓存结果
            self._time_freq_mask_cache[cache_key] = mask
            
            return mask
        except Exception as e:
            print(f"⚠️  创建时频掩码失败: {e}")
            # 返回默认掩码
            return torch.ones((height, width), device=device)

    def _simple_gpu_enhance(self, data, curve_names):
        """简化的GPU增强处理 - 纯GPU实现，避免CPU转换"""
        try:
            # 直接使用GPU进行简单增强，不依赖CPU小波库
            batch_size, num_curves, height, width = data.shape
            
            # 预分配输出tensor
            enhanced_data = torch.zeros((batch_size, num_curves, 64, 64), 
                                      device=data.device, dtype=data.dtype)
            
            # 对每个曲线进行简单的GPU增强
            for j, curve_name in enumerate(curve_names):
                curve_batch = data[:, j, :, :]
                
                # 简单的GPU增强：FFT + 频率滤波 + 逆FFT
                enhanced_curve = self._gpu_simple_enhance(curve_batch)
                enhanced_data[:, j, :, :] = enhanced_curve
            
            return enhanced_data
            
        except Exception as e:
            print(f"⚠️  简单GPU增强失败: {e}")
            # 最后回退到简单resize
            return self._simple_resize_enhance(data)
    
    def _gpu_simple_enhance(self, curve_batch):
        """GPU简单增强处理 - 纯GPU实现"""
        try:
            # 使用GPU FFT进行简单增强
            fft_data = torch.fft.fft2(curve_batch, dim=(-2, -1))
            
            # 简单的频率滤波
            freq_mask = torch.ones_like(fft_data)
            freq_mask[:, :, :fft_data.shape[-2]//4, :] = 0.5  # 低频增强
            freq_mask[:, :, -fft_data.shape[-2]//4:, :] = 0.5  # 高频增强
            
            # 应用掩码
            enhanced_fft = fft_data * freq_mask
            
            # 逆FFT
            enhanced = torch.fft.ifft2(enhanced_fft, dim=(-2, -1)).real
            
            # 调整到目标尺寸
            if enhanced.shape[-2:] != (64, 64):
                enhanced = torch.nn.functional.interpolate(
                    enhanced.unsqueeze(1), 
                    size=(64, 64), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️  GPU简单增强失败: {e}")
            return self._simple_resize_enhance(curve_batch)

    def _gpu_frequency_enhance(self, data, config):
        """GPU频率增强 - 简化的高频处理"""
        try:
            # 简单的频率增强：高通滤波近似
            # 使用简单的卷积核进行边缘增强

            # 创建简单的拉普拉斯算子
            kernel = torch.tensor([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=torch.float32, device=data.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

            # 应用卷积
            enhanced = torch.nn.functional.conv2d(
                data.unsqueeze(1), kernel,
                padding=1
            ).squeeze(1)

            # 归一化到目标尺寸
            enhanced = torch.nn.functional.interpolate(
                enhanced.unsqueeze(1),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            return enhanced

        except Exception as e:
            print(f"⚠️  GPU频率增强失败: {e}")
            # 返回简单resize处理
            return self._simple_resize_enhance(data)

    def _gpu_spatial_smooth(self, data, config):
        """GPU空间平滑 - 简化的低频处理"""
        try:
            # 简单的空间平滑：高斯模糊近似
            # 使用简单的均值滤波

            kernel = torch.ones((3, 3), dtype=torch.float32, device=data.device) / 9
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

            # 应用卷积
            smoothed = torch.nn.functional.conv2d(
                data.unsqueeze(1), kernel,
                padding=1
            ).squeeze(1)

            # 归一化到目标尺寸
            smoothed = torch.nn.functional.interpolate(
                smoothed.unsqueeze(1),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            return smoothed

        except Exception as e:
            print(f"⚠️  GPU空间平滑失败: {e}")
            # 返回简单resize处理
            return self._simple_resize_enhance(data)

    def _optimize_data_loader(self, data_loader):
        """优化数据加载器性能"""
        try:
            # 创建优化的数据加载器
            optimized_loader = torch.utils.data.DataLoader(
                data_loader.dataset,
                batch_size=data_loader.batch_size,
                shuffle=getattr(data_loader, 'shuffle', False),
                num_workers=getattr(data_loader, 'num_workers', 0),
                pin_memory=True,  # 启用内存钉扎
                prefetch_factor=getattr(data_loader, 'prefetch_factor', 4),  # 大幅增加预取因子以支持大批次192
                persistent_workers=getattr(data_loader, 'persistent_workers', False)
            )
            return optimized_loader
        except Exception as e:
            print(f"⚠️  数据加载器优化失败，使用原始加载器: {e}")
            return data_loader

    def _gpu_accelerated_wavelet_transform(self, data: torch.Tensor, curve_names: list) -> torch.Tensor:
        """GPU加速的小波变换处理 - 带完整错误处理的版本"""
        if not self._gpu_wavelet_enabled or not torch.cuda.is_available():
            # 回退到CPU处理
            return self._cpu_wavelet_transform(data, curve_names)

        try:
            # 输入验证
            if data is None or len(data.shape) != 4:
                print(f"⚠️  输入数据格式错误: {data.shape if data is not None else None}")
                return self._cpu_wavelet_transform(data, curve_names)

            batch_size, num_curves, height, width = data.shape

            # 内存检查
            try:
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                if gpu_total > 0 and torch.cuda.memory_allocated() / gpu_total > 0.9:
                    print("⚠️  GPU内存不足，回退到CPU处理")
                    return self._cpu_wavelet_transform(data, curve_names)
            except Exception as e:
                print(f"⚠️  GPU内存检查失败: {e}")
                # 如果无法检查内存，继续GPU处理

            print(f"   GPU处理输入形状: {data.shape}, 设备: {data.device}")

            # 直接调用纯GPU小波变换方法
            result = self._optimized_gpu_wavelet_transform(data, curve_names)

            # 最终验证
            expected_shape = (batch_size, num_curves, 64, 64)
            if result.shape != expected_shape:
                print(f"⚠️  输出形状不匹配，期望: {expected_shape}，实际: {result.shape}")
                # 如果形状不匹配，尝试通过插值调整或返回零张量
                if result.size > 0:
                    try:
                        # 确保维度匹配，以便插值
                        if result.dim() == 3 and num_curves == 1: # (B, H, W) -> (B, 1, H, W)
                            result = result.unsqueeze(1)
                        elif result.dim() == 2 and batch_size == 1 and num_curves == 1: # (H, W) -> (1, 1, H, W)
                            result = result.unsqueeze(0).unsqueeze(0)
                        
                        if result.dim() == 4 and result.shape[1] == num_curves:
                            result = torch.nn.functional.interpolate(
                                result,
                                size=(64, 64),
                                mode='bilinear',
                                align_corners=False
                            )
                        else:
                            print(f"⚠️  无法对形状 {result.shape} 的张量进行插值，返回零张量")
                            result = torch.zeros(expected_shape, device=data.device)
                    except Exception as interp_e:
                        print(f"⚠️  插值失败: {interp_e}，返回零张量")
                        result = torch.zeros(expected_shape, device=data.device)
                else:
                    result = torch.zeros(expected_shape, device=data.device)
            
            # 数据后处理：数值稳定性检查
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"⚠️  GPU加速小波变换产生无效值，切换到简单数据增强")
                # 使用简单的数据增强作为fallback
                result = self._simple_data_augmentation(data)

            print(f"   GPU处理输出形状: {result.shape}, 设备: {result.device}")
            return result

        except Exception as e:
            print(f"⚠️  GPU加速处理失败，回退到纯CPU: {e}")
            import traceback
            traceback.print_exc()
            return self._cpu_wavelet_transform(data, curve_names)

    def _gpu_high_freq_transform(self, data: torch.Tensor, config: dict) -> torch.Tensor:
        """GPU加速的高频曲线变换"""
        try:
            # 使用GPU上的信号处理
            import torch.nn.functional as F

            # 确保输入数据是1D的
            if len(data.shape) > 1:
                data = data.flatten()

            # 检查数据长度是否足够
            if data.shape[0] < 4:
                # 数据太短，无法进行小波变换
                print(f"⚠️  数据长度 {data.shape[0]} 太短，使用简单处理")
                # 返回固定尺寸的输出 (64, 64)
                return torch.zeros((64, 64), device=data.device)

            # 创建Morlet小波核 (在GPU上)
            sigma = config.get('sigma', 1.0)
            scales = torch.arange(1, config['level'] + 1, dtype=torch.float32, device=data.device)

            # GPU上的连续小波变换近似
            spectrogram = self._approximate_cwt_gpu(data, scales, sigma)

            # 应用频率掩蔽 (在GPU上)
            if torch.rand(1).item() < 0.2:
                spectrogram = self._apply_frequency_mask_gpu(spectrogram)

            return spectrogram

        except Exception as e:
            print(f"⚠️  GPU高频变换失败: {e}")
            # 回退到CPU处理
            try:
                data_cpu = data.cpu().numpy()
                result_cpu = self.wavelet_processor._process_high_freq_curve(data_cpu, config)
                return torch.FloatTensor(result_cpu).to(data.device)
            except Exception as cpu_e:
                print(f"⚠️  CPU回退也失败: {cpu_e}")
                # 返回固定尺寸的零矩阵
                target_size = config.get('output_size', (64, 64))
                return torch.zeros(target_size, device=data.device)

    def _gpu_low_freq_transform(self, data: torch.Tensor, config: dict) -> torch.Tensor:
        """GPU加速的低频曲线变换"""
        try:
            # 确保输入数据是1D的
            if len(data.shape) > 1:
                data = data.flatten()

            # 检查数据长度是否足够
            if data.shape[0] < 4:
                # 数据太短，无法进行小波变换
                print(f"⚠️  数据长度 {data.shape[0]} 太短，使用简单处理")
                # 返回固定尺寸的输出 (64, 64)
                return torch.zeros((64, 64), device=data.device)

            # GPU上的小波包分解近似
            levels = config.get('level', 4)
            wavelet = config.get('wavelet', 'sym8')

            # 使用GPU上的离散小波变换近似
            coeffs = self._approximate_dwt_gpu(data, levels)

            # 重构时频图谱
            spectrogram = self._reconstruct_spectrogram_gpu(coeffs, config)

            return spectrogram

        except Exception as e:
            print(f"⚠️  GPU低频变换失败: {e}")
            # 回退到CPU处理
            try:
                data_cpu = data.cpu().numpy()
                result_cpu = self.wavelet_processor._process_low_freq_curve(data_cpu, config)
                return torch.FloatTensor(result_cpu).to(data.device)
            except Exception as cpu_e:
                print(f"⚠️  CPU回退也失败: {cpu_e}")
                # 返回固定尺寸的零矩阵
                target_size = config.get('output_size', (64, 64))
                return torch.zeros(target_size, device=data.device)

    def _approximate_cwt_gpu(self, data: torch.Tensor, scales: torch.Tensor, sigma: float) -> torch.Tensor:
        """GPU上的连续小波变换近似"""
        try:
            # 确保数据是1D的
            if len(data.shape) > 1:
                data = data.flatten()

            data_length = data.shape[0]
            if data_length < 8:
                # 数据太短，返回简单的频谱
                print(f"⚠️  数据长度 {data_length} 太短，使用简单频谱")
                num_scales = len(scales)
                return torch.zeros((num_scales, data_length), device=data.device)

            # 创建Morlet小波核
            t = torch.linspace(-4, 4, min(32, data_length), device=data.device)
            morlet_kernel = torch.exp(-t**2 / (2 * sigma**2)) * torch.cos(5 * t)

            spectrogram_list = []
            for scale in scales:
                try:
                    scale_val = float(scale.item())

                    # 缩放小波 - 确保不超过数据长度
                    if scale_val * len(morlet_kernel) > data_length:
                        scale_val = data_length / len(morlet_kernel)

                    try:
                        scaled_kernel = torch.nn.functional.interpolate(
                            morlet_kernel.unsqueeze(0).unsqueeze(0),
                            scale_factor=scale_val,
                            mode='linear'
                        ).squeeze()
                    except Exception as e:
                        print(f"⚠️  小波核插值失败: {e}")
                        scaled_kernel = morlet_kernel  # 使用原始核作为fallback

                    # 确保卷积核不为空
                    if len(scaled_kernel) == 0:
                        magnitude = torch.zeros_like(data)
                    else:
                        # 卷积操作 (保持在GPU上)
                        try:
                            padding = min(len(scaled_kernel) // 2, data_length - 1)
                            convolved = torch.nn.functional.conv1d(
                                data.unsqueeze(0).unsqueeze(0),
                                scaled_kernel.unsqueeze(0).unsqueeze(0),
                                padding=padding
                            ).squeeze()
                        except Exception as e:
                            print(f"⚠️  卷积操作失败: {e}")
                            convolved = torch.zeros_like(data)

                        # 计算幅度谱
                        magnitude = torch.abs(convolved)

                        # 确保输出长度与输入相同
                        if magnitude.shape[0] != data_length:
                            try:
                                magnitude = torch.nn.functional.interpolate(
                                    magnitude.unsqueeze(0).unsqueeze(0),
                                    size=data_length,
                                    mode='linear'
                                ).squeeze()
                            except Exception as e:
                                print(f"⚠️  幅度谱插值失败: {e}")
                                magnitude = torch.zeros(data_length, device=magnitude.device)

                    spectrogram_list.append(magnitude)

                except Exception as scale_e:
                    print(f"⚠️  尺度 {scale.item()} 处理失败: {scale_e}")
                    # 添加零向量作为fallback
                    spectrogram_list.append(torch.zeros_like(data))

            # 在GPU上堆叠
            if len(spectrogram_list) == 0:
                return torch.zeros((len(scales), data_length), device=data.device)

            spectrogram = torch.stack(spectrogram_list, dim=0)
            return spectrogram

        except Exception as e:
            print(f"⚠️  GPU CWT 失败: {e}")
            # 返回简单的频谱作为fallback
            num_scales = len(scales)
            data_length = data.shape[0] if len(data.shape) > 0 else 64
            return torch.zeros((num_scales, data_length), device=data.device)

    def _approximate_dwt_gpu(self, data: torch.Tensor, levels: int) -> dict:
        """GPU上的离散小波变换近似"""
        try:
            # 确保数据是1D的
            if len(data.shape) > 1:
                # 如果是多维数据，取第一个维度
                data = data.flatten()

            coeffs = {'a': data}  # 近似系数

            for level in range(levels):
                # 简单的下采样近似 (实际应该使用更精确的小波变换)
                # 这里使用简单的低通滤波近似
                kernel = torch.tensor([0.5, 0.5], device=data.device)

                # 确保输入格式正确
                input_tensor = coeffs['a']
                if len(input_tensor.shape) == 0:  # scalar
                    input_tensor = input_tensor.unsqueeze(0)
                elif len(input_tensor.shape) == 1:  # 1D
                    pass
                else:  # 多维
                    input_tensor = input_tensor.flatten()

                # 确保有足够的长度进行卷积
                if input_tensor.shape[0] < 2:
                    break

                filtered = torch.nn.functional.conv1d(
                    input_tensor.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    stride=2
                ).squeeze()

                coeffs[f'd{level+1}'] = filtered
                coeffs['a'] = filtered  # 更新近似系数

            return coeffs

        except Exception as e:
            print(f"⚠️  GPU DWT 失败: {e}")
            # 返回简单的近似
            return {'a': data.flatten() if len(data.shape) > 1 else data}

    def _reconstruct_spectrogram_gpu(self, coeffs: dict, config: dict) -> torch.Tensor:
        """GPU上的时频图谱重构"""
        try:
            # 简单的重构方法 (实际应该使用逆小波变换)
            spectrogram = torch.abs(coeffs['a'])

            # 扩展到目标尺寸
            target_size = config.get('output_size', (64, 64))

            if len(spectrogram.shape) == 1:
                # 1D 到 2D 的转换
                length = spectrogram.shape[0]
                if length == 0:
                    # 如果长度为0，返回零矩阵
                    return torch.zeros(target_size, device=spectrogram.device)

                # 创建2D表示：将1D数据扩展为2D
                try:
                    # 首先将1D数据转换为2D格式
                    spectrogram_2d = spectrogram.unsqueeze(0)  # [1, length]

                    # 使用双线性插值调整到目标尺寸
                    if length != target_size[1] or 1 != target_size[0]:
                        spectrogram_2d = torch.nn.functional.interpolate(
                            spectrogram_2d.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, length]
                            size=(target_size[0], target_size[1]),  # [height, width]
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # [height, width]

                    return spectrogram_2d

                except Exception as e:
                    print(f"⚠️  1D到2D转换失败: {e}")
                    # 返回零矩阵作为fallback
                    return torch.zeros(target_size, device=spectrogram.device)

            elif len(spectrogram.shape) == 2:
                # 如果已经是2D，直接插值到目标尺寸
                current_h, current_w = spectrogram.shape
                if (current_h, current_w) != target_size:
                    try:
                        spectrogram = torch.nn.functional.interpolate(
                            spectrogram.unsqueeze(0).unsqueeze(0),  # [1, 1, h, w]
                            size=target_size,  # [height, width]
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)  # [height, width]
                    except Exception as e:
                        print(f"⚠️  2D插值失败: {e}")
                        # 返回零矩阵作为fallback
                        return torch.zeros(target_size, device=spectrogram.device)

                return spectrogram

            else:
                # 对于更高维度的tensor，展平后处理
                spectrogram = spectrogram.flatten()
                return self._reconstruct_spectrogram_gpu({'a': spectrogram}, config)

        except Exception as e:
            print(f"⚠️  GPU时频图谱重构失败: {e}")
            # 返回目标尺寸的零矩阵作为fallback
            target_size = config.get('output_size', (64, 64))
            return torch.zeros(target_size, device=coeffs['a'].device)

    def _apply_frequency_mask_gpu(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """GPU上的频率掩蔽"""
        freq_mask = torch.rand_like(spectrogram) > 0.1  # 随机掩蔽10%的频率
        return spectrogram * freq_mask

    def _cpu_single_curve_transform(self, curve_data: np.ndarray, curve_name: str) -> np.ndarray:
        """处理单个曲线的CPU小波变换"""
        try:
            # 使用原始的CPU小波处理器处理单个曲线
            enhanced_curve = self.wavelet_processor.process_curve_data(
                curve_data, curve_name, enable_augmentation=True
            )
            return enhanced_curve
        except Exception as e:
            print(f"⚠️  单曲线CPU处理失败 {curve_name}: {e}")
            # 返回原始数据形状的零数组作为fallback
            return np.zeros((64, 64), dtype=np.float32)

    def _cpu_wavelet_transform(self, data: torch.Tensor, curve_names: list) -> torch.Tensor:
        """CPU回退的小波变换处理"""
        try:
            # 转换为numpy进行CPU处理
            data_np = data.cpu().numpy()

            # 使用原始的CPU小波处理器
            enhanced_data = self.wavelet_processor.process_batch_data(
                data_np, curve_names, enable_augmentation=True
            )

            # 转换回tensor
            result = torch.FloatTensor(enhanced_data).to(data.device)

            # 数据后处理：数值稳定性检查
            if torch.isnan(result).any() or torch.isinf(result).any():
                print(f"⚠️  小波变换产生无效值，切换到简单数据增强")
                # 使用简单的数据增强作为fallback
                result = self._simple_data_augmentation(data)

            return result
        except Exception as e:
            print(f"⚠️  CPU批量处理失败: {e}")
            # 返回原始数据作为fallback
            return data

    def _process_batch_with_diversity(self, data, curve_names, epoch, batch_idx):
        """多样化小波处理 - 每个epoch使用不同的变换"""
        try:
            enhanced_data = self._apply_diverse_wavelet_transform(
                data, curve_names, epoch, batch_idx
            )
            
            return enhanced_data
            
        except Exception as e:
            print(f"⚠️ 多样化小波处理失败: {e}")
            # 回退到简单增强
            return self._simple_data_augmentation(data)
    
    def _apply_diverse_wavelet_transform(self, data, curve_names, epoch, batch_idx):
        """应用多样化的小波变换"""
        try:
            # 转换为numpy进行CPU处理
            data_np = data.cpu().numpy()
            
            # 使用小波处理器进行变换
            enhanced_data = self.wavelet_processor.process_batch_data(
                data_np, curve_names, 
                enable_augmentation=True
            )
            
            # 转换回tensor
            result = torch.FloatTensor(enhanced_data).to(data.device)
            
            # 应用额外的数据增强
            if self.config.DATA_CONFIG.get('enable_frequency_masking', False):
                result = self._apply_frequency_masking(result, epoch, batch_idx)
            
            if self.config.DATA_CONFIG.get('enable_time_shift', False):
                result = self._apply_time_shift(result, epoch, batch_idx)
                
            if self.config.DATA_CONFIG.get('enable_noise_augmentation', False):
                result = self._apply_noise_augmentation(result, epoch, batch_idx)
            
            return result
            
        except Exception as e:
            print(f"⚠️ 小波变换失败: {e}")
            return self._simple_data_augmentation(data)
    
    def _apply_frequency_masking(self, data, epoch, batch_idx):
        """应用频率掩蔽"""
        # 使用epoch和batch_idx设置随机种子，确保可重现性
        seed = epoch * 1000 + batch_idx + 100
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if np.random.random() < self.config.DATA_CONFIG.get('frequency_masking_ratio', 0.2):
            # 连续频带掩蔽（更贴近真实频带丢失）
            b, c, h, w = data.shape
            band_width = max(1, int(h * 0.1))  # 10% 频带宽度
            start = np.random.randint(0, max(1, h - band_width + 1))
            band_mask = torch.ones((h,), device=data.device, dtype=data.dtype)
            band_mask[start:start+band_width] = 0
            band_mask = band_mask.view(1, 1, h, 1)
            data = data * band_mask
        return data
    
    def _apply_time_shift(self, data, epoch, batch_idx):
        """应用时间偏移"""
        # 使用epoch和batch_idx设置随机种子，确保可重现性
        seed = epoch * 1000 + batch_idx + 200
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if np.random.random() < 0.5:
            shift_range = self.config.DATA_CONFIG.get('time_shift_range', 3)
            shift = np.random.randint(-shift_range, shift_range + 1)
            if shift != 0:
                data = torch.roll(data, shifts=shift, dims=-1)
        return data
    
    def _apply_noise_augmentation(self, data, epoch, batch_idx):
        """应用噪声增强"""
        # 使用epoch和batch_idx设置随机种子，确保可重现性
        seed = epoch * 1000 + batch_idx + 300
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if np.random.random() < 0.3:
            snr_db = self.config.DATA_CONFIG.get('noise_snr_db', 25)
            # 计算每个样本的信号功率并按SNR生成噪声
            signal_power = (data ** 2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(data) * noise_power.sqrt()
            data = data + noise
        return data

    def _simple_data_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """简单的数据增强作为fallback，确保数值稳定性"""
        try:
            augmented = data
            if bool(self.config.DATA_CONFIG.get('enable_light_augmentations', True)):
                # 轻量噪声
                snr_db = float(self.config.DATA_CONFIG.get('light_noise_snr_db', 22))
                signal_power = (augmented ** 2).mean(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = torch.randn_like(augmented) * noise_power.sqrt()
                augmented = augmented + noise

                # 轻量频带遮挡
                if float(self.config.DATA_CONFIG.get('light_frequency_masking_ratio', 0.10)) > 0:
                    b, c, h, w = augmented.shape
                    band_width = max(1, int(h * float(self.config.DATA_CONFIG.get('light_frequency_masking_ratio', 0.10))))
                    start = torch.randint(low=0, high=max(1, h - band_width + 1), size=(1,), device=augmented.device).item()
                    band_mask = torch.ones((h,), device=augmented.device, dtype=augmented.dtype)
                    band_mask[start:start+band_width] = 0
                    band_mask = band_mask.view(1, 1, h, 1)
                    augmented = augmented * band_mask

                # 轻量时移
                shift_range = int(self.config.DATA_CONFIG.get('light_time_shift_range', 1))
                if shift_range > 0:
                    shift = int(torch.randint(low=-shift_range, high=shift_range+1, size=(1,)).item())
                    if shift != 0:
                        augmented = torch.roll(augmented, shifts=shift, dims=-1)

            augmented = torch.clamp(augmented, -10.0, 10.0)
            return augmented
        except Exception as e:
            print(f"⚠️  简单数据增强也失败: {e}")
            return data

    def _safe_resize_array(self, array: np.ndarray, target_size: tuple) -> np.ndarray:
        """安全地调整numpy数组尺寸"""
        try:
            if array.shape == target_size:
                return array

            # 创建目标尺寸的数组
            result = np.zeros(target_size, dtype=array.dtype)

            # 计算可以复制的区域
            min_h = min(array.shape[0], target_size[0])
            min_w = min(array.shape[1] if len(array.shape) > 1 else 1, target_size[1])

            if len(array.shape) == 1:
                # 1D数组，复制到第一行
                result[0, :min_w] = array[:min_w]
            else:
                # 2D数组
                result[:min_h, :min_w] = array[:min_h, :min_w]

            return result

        except Exception as e:
            print(f"⚠️  数组尺寸调整失败: {e}")
            return np.zeros(target_size, dtype=np.float32)

    def validate_epoch(self, epoch):
        """验证一个epoch - 优化版本"""
        self.model.eval()
        # 在验证/测试时临时切换到EMA权重
        using_ema = False
        if self.ema is not None:
            try:
                self.ema.store(self.model)
                self.ema.copy_to(self.model)
                using_ema = True
            except Exception as _e:
                using_ema = False
        total_loss = 0
        correct = 0
        total = 0
        total_f1 = 0
        num_batches = 0
        
        # 用于计算极少数类召回率的缓冲区
        self._val_labels_buffer = []
        self._val_preds_buffer = []
        # 按井评估缓冲
        self._val_wells_buffer = []
        
        # 添加验证时间监控
        import time
        val_start_time = time.time()
        
        with torch.no_grad():
            # 优化：创建快速验证数据加载器
            val_dataset_size = len(self.val_loader.dataset)

            # 使用与训练相同的批次大小，确保批次大小一致性
            train_batch_size = self.config.DATA_CONFIG['batch_size']
            val_batch_size = train_batch_size  # 直接使用训练批次大小，确保一致性

            fast_val_loader = DataLoader(
                self.val_loader.dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=0,  # 减少worker数量提升速度
                pin_memory=False,  # 禁用pin_memory以减少开销
                drop_last=False
            )

            val_batches = list(fast_val_loader)
            # 优化验证策略：平衡速度和准确性
            total_val_batches = len(val_batches)

            # 验证时处理所有批次，确保验证结果准确
            max_val_batches = total_val_batches
            print(f"📊 验证集信息: 总批次数={total_val_batches}, 将处理={max_val_batches}个批次")

            max_val_batches = min(max_val_batches, total_val_batches)  # 确保不超过总数

            # 保存max_val_batches用于后续质量评估
            self._current_max_val_batches = max_val_batches

            # 验证循环（删除详细日志输出）
            processed_batches = 0
            for batch_idx, batch in enumerate(val_batches[:max_val_batches]):
                try:
                    # 解包批次
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            data, labels, wells = batch
                        elif len(batch) == 2:
                            data, labels = batch
                            wells = None
                        else:
                            continue
                    else:
                        continue
                    
                    # 优化数据传输
                    try:
                        data = data.to(self.device, non_blocking=True)
                        labels = labels.squeeze().to(self.device, non_blocking=True)
                    except Exception as device_e:
                        print(f"⚠️  验证数据传输失败: {device_e}")
                        continue
                    
                    # GPU内存监控 (仅监控，不调整批次大小)
                    if self._psutil_available and batch_idx % self._memory_monitor_interval == 0:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_memory_used = info.used / (1024**3)  # GB
                            gpu_memory_total = info.total / (1024**3)  # GB
                            gpu_memory_utilization = gpu_memory_used / gpu_memory_total
                            print(f"🔍 验证阶段GPU内存使用率: {gpu_memory_utilization:.2f} (已用: {gpu_memory_used:.2f}GB / 总共: {gpu_memory_total:.2f}GB)")
                        except Exception as e:
                            print(f"⚠️  验证阶段GPU内存监控失败: {e}")
                            # 不禁用_psutil_available，因为可能只是偶尔的读取失败

                    # 验证数据优化 - 使用相同的预处理以保持一致性
                    # 验证时使用相同的预处理，避免训练验证分布不一致
                    # 这样可以确保模型评估的准确性和公平性

                    # 验证数据已经通过小波包分解处理，无需再次处理
                    # 注释掉重复的小波包分解处理，避免数据过度处理

                    # 简化数据验证 - 只在调试模式下进行
                    if self.config.TRAIN_CONFIG.get('debug_mode', False):
                        if torch.isnan(data).any() or torch.isinf(data).any():
                            continue
                    
                    # 前向传播
                    try:
                        if self.scaler is not None:
                            with autocast():
                                outputs = self.model(data)
                                loss = self.criterion(outputs, labels)
                        else:
                            outputs = self.model(data)
                            loss = self.criterion(outputs, labels)
                    except Exception as forward_e:
                        print(f"⚠️  验证前向传播失败: {str(forward_e)[:100]}...")
                        continue
                    
                    # 统计 - 优化版本，减少计算开销
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集预测结果用于计算极少数类召回率
                    self._val_labels_buffer.append(labels.detach().cpu())
                    self._val_preds_buffer.append(predicted.detach().cpu())
                    # 收集井名
                    try:
                        if wells is not None:
                            if isinstance(wells, (list, tuple)):
                                self._val_wells_buffer.extend(list(wells))
                            else:
                                self._val_wells_buffer.extend([str(wells)] * labels.size(0))
                        else:
                            self._val_wells_buffer.extend(["unknown"] * labels.size(0))
                    except Exception:
                        pass
                    
                    # 简化的F1分数计算 - 使用准确率近似
                    f1 = (predicted == labels).float().mean().item()
                    total_f1 += f1
                    num_batches += 1
                    processed_batches += 1
                    
                except Exception as e:
                    if self.config.TRAIN_CONFIG.get('debug_mode', False):
                        print(f"⚠️  验证批次 {batch_idx} 失败: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        # 在非调试模式下也记录严重错误
                        print(f"⚠️  验证批次 {batch_idx} 处理失败")
                        # 打印更详细的错误信息用于调试
                        error_type = type(e).__name__
                        error_msg = str(e)[:150]
                        print(f"   错误类型: {error_type}")
                        print(f"   错误信息: {error_msg}")
                    continue
        
        # 计算平均指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_acc = 100. * correct / total if total > 0 else 0
        avg_f1 = total_f1 / num_batches if num_batches > 0 else 0

        # 验证时间统计（删除详细日志输出）
        val_time = time.time() - val_start_time
        
        # 恢复非EMA权重
        try:
            if using_ema and self.ema is not None:
                self.ema.restore(self.model)
        except Exception:
            pass
        
        return avg_loss, avg_acc, avg_f1

    def evaluate_test(self):
        """測試集評估：與validate一致但讀取test_loader；使用EMA權重（若可用）"""
        if self.test_loader is None:
            raise RuntimeError("未配置測試集數據加載器")
        self.model.eval()

        # 暂存并切换EMA权重
        using_ema = False
        if self.ema is not None:
            try:
                self.ema.store(self.model)
                self.ema.copy_to(self.model)
                using_ema = True
            except Exception:
                using_ema = False
        total_loss = 0.0
        correct = 0
        total = 0
        total_f1 = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.test_loader:
                try:
                    data, labels = self._get_original_data_from_batch(batch)
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                    elif labels.dim() > 1:
                        labels = labels.view(-1)
                    data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(data)
                    # 温度缩放（仅评估阶段应用，不影响训练）
                    if getattr(self, '_temperature', 1.0) != 1.0:
                        outputs = outputs / float(self._temperature)
                    loss = self.criterion(outputs, labels)
                    total_loss += float(loss.item())
                    _, predicted = torch.max(outputs, 1)
                    # 推理平滑（可选）
                    try:
                        if bool(self.config.DATA_CONFIG.get('enable_smoothing', True)) and int(self.config.DATA_CONFIG.get('smooth_window', 5)) > 1:
                            k = int(self.config.DATA_CONFIG.get('smooth_window', 5))
                            if k % 2 == 0:
                                k = k + 1
                            # 简单多数投票平滑（批内近似，严格时应在序列维度上按井拼接）
                            win = max(1, k)
                            if predicted.numel() >= win:
                                pred_np = predicted.cpu().numpy()
                                from collections import Counter as _Counter
                                smoothed = pred_np.copy()
                                half = win // 2
                                for i in range(len(pred_np)):
                                    l = max(0, i - half)
                                    r = min(len(pred_np), i + half + 1)
                                    smoothed[i] = _Counter(pred_np[l:r]).most_common(1)[0][0]
                                predicted = torch.tensor(smoothed, device=predicted.device)
                    except Exception:
                        pass
                    correct += int((predicted == labels).sum().item())
                    total += int(labels.size(0))
                    f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                    total_f1 += float(f1)
                    num_batches += 1
                except Exception as e:
                    print(f"⚠️  測試批次處理失敗: {e}")
                    continue
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        avg_f1 = total_f1 / max(num_batches, 1)

        # 恢复参数
        try:
            if using_ema and self.ema is not None:
                self.ema.restore(self.model)
        except Exception:
            pass
        return avg_loss, accuracy, avg_f1

    def _evaluate_and_summarize(self, loader, split_name: str = "Test") -> None:
        """對指定數據集進行詳細評估，輸出每類指標、混淆矩陣與按井macro-F1。"""
        self.model.eval()
        using_ema = False
        if self.ema is not None:
            try:
                self.ema.store(self.model)
                self.ema.copy_to(self.model)
                using_ema = True
            except Exception:
                using_ema = False

        import numpy as _np
        from sklearn.metrics import classification_report as _cls_report
        from sklearn.metrics import confusion_matrix as _conf_mat
        from sklearn.metrics import f1_score as _f1

        y_true_list = []
        y_pred_list = []
        wells_list = []
        with torch.no_grad():
            for batch in loader:
                try:
                    # 直接解包，保留井名
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            data, labels, wells = batch
                        elif len(batch) == 2:
                            data, labels = batch
                            wells = None
                        else:
                            continue
                    else:
                        continue
                    if labels.dim() == 0:
                        labels = labels.unsqueeze(0)
                    elif labels.dim() > 1:
                        labels = labels.view(-1)
                    data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    outputs = self.model(data)
                    if getattr(self, '_temperature', 1.0) != 1.0:
                        outputs = outputs / float(self._temperature)
                    _, predicted = torch.max(outputs, 1)
                    y_true_list.append(labels.detach().cpu())
                    y_pred_list.append(predicted.detach().cpu())
                    # 井名
                    try:
                        if wells is not None:
                            if isinstance(wells, (list, tuple)):
                                wells_list.extend(list(wells))
                            else:
                                wells_list.extend([str(wells)] * labels.size(0))
                        else:
                            wells_list.extend(["unknown"] * labels.size(0))
                    except Exception:
                        pass
                except Exception:
                    continue

        if len(y_true_list) == 0:
            print(f"⚠️  {split_name} 無法收集到有效樣本，跳過詳細彙總")
        else:
            y_true = torch.cat(y_true_list).numpy()
            y_pred = torch.cat(y_pred_list).numpy()
            print(_cls_report(y_true, y_pred, digits=3, zero_division=0))
            cm = _conf_mat(y_true, y_pred)
            print(f"{split_name} Confusion Matrix:\n{cm}")
            # 输出排除“无”类的报告（若配置了 none_class_id 且存在于标签中）
            try:
                none_id = self.config.MODEL_CONFIG.get('none_class_id', None)
                if none_id is not None and int(none_id) >= 0 and (int(none_id) in y_true):
                    mask = y_true != none_id
                    if mask.sum() > 0:
                        y_true_ex = y_true[mask]
                        y_pred_ex = y_pred[mask]
                        print(f"{split_name} (排除'无'类) 指标：")
                        print(_cls_report(y_true_ex, y_pred_ex, digits=3, zero_division=0))
                        cm_ex = _conf_mat(y_true_ex, y_pred_ex)
                        print(f"{split_name} (排除'无'类) Confusion Matrix:\n{cm_ex}")
            except Exception as _e:
                print(f"⚠️  {split_name} 排除'无'类指标失败: {_e}")
            try:
                if wells_list:
                    wells_arr = _np.array(wells_list)
                    unique_wells = _np.unique(wells_arr)
                    print(f"{split_name} Per-well macro-F1:")
                    for wn in unique_wells:
                        mask = (wells_arr == wn)
                        wt = y_true[mask]
                        wp = y_pred[mask]
                        if wt.size > 0:
                            wf1 = _f1(wt, wp, average='macro', zero_division=0)
                            print(f"  {wn}: {wf1:.4f} (n={wt.size})")
            except Exception as _e:
                print(f"⚠️  {split_name} 按井彙總失敗: {_e}")

        try:
            if using_ema and self.ema is not None:
                self.ema.restore(self.model)
        except Exception:
            pass
    
    def _calculate_f1_score(self, predicted, labels):
        """计算F1分数 - 纯GPU实现，避免CPU转换"""
        try:
            # 使用GPU计算F1分数，避免CPU转换
            # 计算准确率作为F1分数的近似
            correct = (predicted == labels).float()
            accuracy = correct.mean().item()
            
            # 简单的F1分数估算（基于准确率）
            # 这比CPU转换快得多
            return accuracy
            
        except:
            return 0.0
    
    def _update_learning_rate(self, epoch, val_loss, val_f1):
        """更新学习率（余弦+预热）"""
        warmup_epochs = int(self.config.TRAIN_CONFIG.get('warmup_epochs', 0))
        base_lr = float(self.config.TRAIN_CONFIG.get('learning_rate', 1e-3))
        
        if epoch <= warmup_epochs and warmup_epochs > 0:
            # 线性预热到 base_lr，从 base_lr/10 开始
            min_lr = base_lr * 0.1
            warmup_lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)
            for g in self.optimizer.param_groups:
                g['lr'] = warmup_lr
            print(f"🔥 预热阶段 Epoch {epoch}: LR = {warmup_lr:.6f}")
        else:
            # 正常调度器更新
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_f1)
            else:
                self.scheduler.step()
    
    def _check_early_stopping(self, epoch, val_loss, val_acc, val_f1):
        """检查早停：优先监控极少数类召回率（含油水层），其次macro-F1"""
        # 获取极少数类ID (含油水层，通常是类别5)
        RARE_CLASS_ID = 4  # 油水层（合并后的类别）
        
        # 如果有验证集预测结果，计算极少数类召回率
        rare_recall = None
        if hasattr(self, '_val_preds_buffer') and hasattr(self, '_val_labels_buffer') and self._val_labels_buffer:
            try:
                y_true_val = torch.cat(self._val_labels_buffer).cpu().numpy()
                y_pred_val = torch.cat(self._val_preds_buffer).cpu().numpy()
                
                # 计算所有类别的召回率
                from sklearn.metrics import classification_report
                report = classification_report(y_true_val, y_pred_val, output_dict=True, zero_division=0)
                rare_recall = report.get(str(RARE_CLASS_ID), {}).get('recall', 0.0)
                
                # 存储供其他地方使用
                self._last_rare_recall = rare_recall
            except Exception as e:
                print(f"⚠️  计算极少数类召回率失败: {e}")
                rare_recall = None
        
        # 决定监控指标
        rare_target = float(self.config.TRAIN_CONFIG.get('rare_recall_target', 0.7))
        use_rare = bool(self.config.TRAIN_CONFIG.get('monitor_rare_recall', True))
        min_epochs = int(self.config.TRAIN_CONFIG.get('min_epochs', 50))
        
        improved = False
        metric_name = 'F1'
        metric_value = val_f1
        
        # 优先监控极少数类召回率（过滤掉zero-support导致的伪高分情况）
        if use_rare and rare_recall is not None:
            metric_name = 'RareRecall'
            metric_value = rare_recall
            
            # 更新最佳极少数类召回率
            if not hasattr(self, 'best_val_rare_recall'):
                self.best_val_rare_recall = 0.0
                
            if rare_recall > self.best_val_rare_recall:
                self.best_val_rare_recall = rare_recall
                self.best_val_f1 = val_f1  # 同时更新F1
                self.best_val_acc = val_acc
                self.patience_counter = 0
                improved = True
                self._save_best_model(epoch, val_loss, val_acc, val_f1)
                print(f"🏆 当前最佳极少数类召回率: {self.best_val_rare_recall:.4f}")
                print(f"🏆 当前最佳F1分数: {self.best_val_f1:.4f}")
                print(f"🏆 当前最佳验证准确率: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                
            # 早停条件：极少数类召回率达到目标且连续rare_patience轮无提升，且达到min_epochs
            rare_patience = int(self.config.TRAIN_CONFIG.get('rare_patience', 10))
            if self.best_val_rare_recall >= rare_target and self.patience_counter >= rare_patience and epoch >= min_epochs:
                print(f"✅ 极少数类召回率达到目标 ({self.best_val_rare_recall:.4f} >= {rare_target:.2f}) 且连续 {self.patience_counter} 轮无提升")
                print(f"🛑 早停触发，训练结束于第 {epoch} 轮")
                return True
        else:
            # 回退到监控F1分数
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.patience_counter = 0
                improved = True
                self._save_best_model(epoch, val_loss, val_acc, val_f1)
                print(f"🏆 当前最佳F1分数: {self.best_val_f1:.4f}")
                print(f"🏆 当前最佳验证准确率: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
        
        # 检查早停 - 更合理的条件；若稀有類無支撐或召回恒為0，回退到F1監控
        if use_rare and (rare_recall is None or rare_recall == 0.0):
            # 回退到F1的耐心計數
            if val_f1 > getattr(self, 'best_val_f1', 0.0):
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.patience_counter = 0
                improved = True
                self._save_best_model(epoch, val_loss, val_acc, val_f1)
                print(f"🏆 回退監控F1：最佳F1={self.best_val_f1:.4f} Acc={self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1

        # 檢查早停門檻
        if self.patience_counter >= self.early_stopping_patience:
            if epoch >= min_epochs:
                print(f"🛑 早停触发，训练结束于第 {epoch} 轮")
                return True
            else:
                print(f"⚠️  早停条件满足但训练轮数不足({epoch}<{min_epochs})，重置patience继续训练")
                # 重置patience计数器，但保持最佳指标
                self.patience_counter = max(0, self.patience_counter - 5)  # 减少一些patience而不是完全重置
        
        return False
    
    def _save_best_model(self, epoch, val_loss, val_acc, val_f1):
        """保存最佳模型"""
        model_path = os.path.join(self.output_dir, 'best_model.pth')
        # 安全獲取調度器狀態（none 調度器無 state_dict）
        scheduler_state = None
        try:
            if hasattr(self.scheduler, 'state_dict') and callable(getattr(self.scheduler, 'state_dict')):
                scheduler_state = self.scheduler.state_dict()
        except Exception:
            scheduler_state = None

        payload = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'config': self.config,
            'class_weights': self.class_weights
        }
        torch.save(payload, model_path)
    
    def _get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def _gpu_warmup(self):
        """GPU预热和优化 - 针对RTX A2000 6GB显存优化"""
        if not torch.cuda.is_available():
            return
            
        print("🔥 执行RTX A2000 GPU预热和优化...")
        try:
            # 检查GPU信息
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory = gpu_memory_total / 1024**3 if gpu_memory_total > 0 else 0
            except Exception as e:
                gpu_name = "未知"
                gpu_memory = 0
                print(f"⚠️  GPU信息获取失败: {e}")
            print(f"   GPU: {gpu_name}, 显存: {gpu_memory:.1f}GB")
            
            # 创建预热数据 - 使用合理大小进行预热
            warmup_batch_size = min(32, self.config.DATA_CONFIG['batch_size'])
            warmup_data = torch.randn(warmup_batch_size, 6, 64, 64, device=self.device)
            warmup_labels = torch.randint(0, self.num_classes, (warmup_batch_size,), device=self.device)
            
            # 预热模型和损失函数 - 避免重复计算
            self.model.train()
            with torch.no_grad():
                warmup_outputs = self.model(warmup_data)
                _ = self.criterion(warmup_outputs, warmup_labels)
            
            # 预热优化器
            self.optimizer.zero_grad()
            
            # 清理预热数据
            del warmup_data, warmup_labels
            torch.cuda.empty_cache()
            
            # 设置RTX A2000专用优化参数
            torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
            torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提升性能
            
            # RTX A2000专用优化
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.allow_tf32 = True  # 启用TF32以提升性能
            torch.backends.cuda.matmul.allow_tf32 = True  # 启用矩阵乘法TF32
            
            # 设置内存管理策略
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            print("✅ RTX A2000 GPU预热完成，启用专业级优化")
            print(f"   📊 显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / {gpu_memory:.1f}GB")
            
        except Exception as e:
            print(f"⚠️  RTX A2000 GPU预热失败: {e}")
    
    def _apply_breakthrough_optimizations(self):
        """应用突破精度停滞的专用优化"""
        if not hasattr(self.config, 'BREAKTHROUGH_CONFIG') or not self.config.BREAKTHROUGH_CONFIG.get('enable_breakthrough_mode', False):
            return
            
        print("🎯 应用突破精度停滞的专用优化...")
        
        # 1. 动态调整学习率
        if hasattr(self.optimizer, 'param_groups'):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], 0.002)  # 确保最低学习率
                
        # 2. 减少正则化
        if self.config.BREAKTHROUGH_CONFIG.get('reduce_regularization', False):
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Dropout):
                    module.p = max(0.1, module.p * 0.5)  # 减半dropout率
                    
        # 3. 禁用复杂的预处理缓存
        if self.config.BREAKTHROUGH_CONFIG.get('disable_complex_preprocessing', False):
            self._preprocessing_cache_enabled = False
            self._enhanced_data_cache.clear()
            
        print("✅ 突破优化应用完成")
    
    def train(self):
        """开始增强训练 - 突破精度停滞"""
        print("🚀 开始增强优化训练 (突破精度停滞模式)...")
        print("=" * 60)
        print("🎯 精度突破策略:")
        print("   🔥 高学习率 + 低正则化")
        print("   📈 大批次 + 长训练")
        print("   🎯 BSMOTE数据平衡")
        print("   🚀 大容量模型")
        print("   ⚡ 简化数据预处理")
        print("   💪 类别权重优化")
        print("   🛡️  数值稳定性保证")
        print("=" * 60)
        
        # Tiny 模式開關：檢測環境變數
        tiny_overfit = str(os.environ.get('TINY_OVERFIT', '')).strip().lower() in ('1','true','yes','y')
        
        # 应用突破精度停滞的优化
        self._apply_breakthrough_optimizations()

        # Tiny 模式：自動關閉正則/增強/平衡/調度器，改用純CE、固定LR
        if tiny_overfit:
            print("🧪 Tiny 模式：自動關閉正則/增強/平衡/調度器，使用純CE與固定LR")
            # 關閉 Focal / 類別權重 / 平滑
            try:
                self.config.TRAIN_CONFIG['use_focal_loss'] = False
                self.config.TRAIN_CONFIG['label_smoothing'] = 0.0
            except Exception:
                pass
            # 關閉資料增強/小波增強
            try:
                self._preprocessing_cache_enabled = False
                self.config.DATA_CONFIG['enable_wavelet_augmentation'] = False
            except Exception:
                pass
            # 關閉梯度裁剪、調度器
            try:
                self.config.TRAIN_CONFIG['gradient_clip'] = False
                self.config.TRAIN_CONFIG['scheduler_type'] = 'none'
            except Exception:
                pass
            # 重新設置損失為純CE（不帶權重）
            try:
                self.criterion = nn.CrossEntropyLoss()
            except Exception:
                pass
        
        # 设置随机种子
        set_global_seed(42)

        # 训练前数值稳定性自检
        print("🔍 执行训练前数值稳定性自检...")
        self._pre_training_stability_check()
        print("-" * 60)

        # 创建训练分析器
        try:
            analyzer = IntegratedTrainingAnalyzer(
                output_dir=self.output_dir
            )
        except Exception as e:
            print(f"⚠️  训练分析器创建失败: {e}")
            analyzer = None
        
        # Tiny Overfit 诊断模式（通过环境变量启用）
        tiny_overfit_batches = int(os.environ.get('TINY_OVERFIT_BATCHES', '8'))

        # 训练循环 - 高级GPU优化
        for epoch in range(1, self.config.TRAIN_CONFIG['epochs'] + 1):
            epoch_start_time = time.time()
            
            # GPU预热和优化
            if epoch == 1:
                self._gpu_warmup()
            
            # 训练一个epoch
            if tiny_overfit:
                # 仅使用少量批次进行过拟合诊断
                original_loader = self.train_loader
                # 構造一個僅包含前 N 個批次的數據載入器視圖；若設置 TINY_SINGLE_BATCH=1，則僅用單批次反覆訓練
                use_single_batch = str(os.environ.get('TINY_SINGLE_BATCH', '')).strip().lower() in ('1','true','yes','y')
                if use_single_batch:
                    small_indices = list(range(min(len(original_loader.dataset), self.config.DATA_CONFIG.get('batch_size', 64))))
                    print("🧪 Tiny: 單批次重複訓練模式已啟用")
                else:
                small_indices = list(range(min(len(original_loader.dataset), self.config.DATA_CONFIG.get('batch_size', 64) * tiny_overfit_batches)))
                small_subset = torch.utils.data.Subset(original_loader.dataset, small_indices)
                self.train_loader = DataLoader(
                    small_subset,
                    batch_size=self.config.DATA_CONFIG.get('batch_size', 64),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False
                )
                print(f"🧪 Tiny Overfit 模式: 使用 {len(self.train_loader)} 个批次进行快速过拟合诊断")
                train_loss, train_acc, train_f1 = self.train_epoch(epoch)
                # 还原原始加载器
                self.train_loader = original_loader
            else:
                train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # 计算训练时间
            train_time = time.time() - epoch_start_time
            
            # 验证（每5个epoch执行一次，或最后一个epoch）
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            val_time = 0.0
            validation_interval = self.config.TRAIN_CONFIG.get('validation_interval', 5)
            do_validate = (epoch % validation_interval == 0) or (epoch == self.config.TRAIN_CONFIG['epochs']) or epoch <= 5
            if do_validate:
                val_start_time = time.time()
                val_loss, val_acc, val_f1 = self.validate_epoch(epoch)
                # 額外輸出每類指標與混淆矩陣
                try:
                    if hasattr(self, '_val_labels_buffer') and self._val_labels_buffer:
                        import numpy as _np
                        from sklearn.metrics import classification_report, confusion_matrix
                        y_true_val = torch.cat(self._val_labels_buffer).cpu().numpy()
                        y_pred_val = torch.cat(self._val_preds_buffer).cpu().numpy()
                        print(classification_report(y_true_val, y_pred_val, digits=3, zero_division=0))
                        cm = confusion_matrix(y_true_val, y_pred_val)
                        print(f"Confusion Matrix:\n{cm}")
                        # 按井输出macro-F1（若可用）
                        try:
                            if hasattr(self, '_val_wells_buffer') and self._val_wells_buffer:
                                wells_arr = _np.array(self._val_wells_buffer)
                                unique_wells = _np.unique(wells_arr)
                                print("Per-well macro-F1:")
                                from sklearn.metrics import f1_score
                                for wn in unique_wells:
                                    mask = (wells_arr == wn)
                                    wt = y_true_val[mask]
                                    wp = y_pred_val[mask]
                                    if wt.size > 0:
                                        wf1 = f1_score(wt, wp, average='macro', zero_division=0)
                                        print(f"  {wn}: {wf1:.4f} (n={wt.size})")
                        except Exception as _perwell_e:
                            print(f"⚠️  按井指标输出失败: {_perwell_e}")
                except Exception as _e:
                    print(f"⚠️  輸出分類報告失敗: {_e}")
                val_time = time.time() - val_start_time
                print(f"✅ 第{epoch}轮验证完成: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")
            
            # 计算总epoch时间
            epoch_time = time.time() - epoch_start_time
            
            # 记录训练历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['train_f1'].append(train_f1)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_time'].append(epoch_time)
            
            # 记录GPU内存使用
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.training_history['gpu_memory'].append(gpu_memory)
            else:
                self.training_history['gpu_memory'].append(0.0)
            
            # 更新学习率
            self._update_learning_rate(epoch, val_loss, val_f1)
            
            # 优化打印训练信息 - 减少输出频率
            if epoch % 10 == 0 or epoch <= 5:  # 每10个epoch或前5个epoch打印
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Train F1={train_f1:.4f}")
                if do_validate:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}: "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val F1={val_f1:.4f}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}: "
                      f"LR={self.optimizer.param_groups[0]['lr']:.6f}, Train Time={train_time:.1f}s, Val Time={val_time:.1f}s, Total Time={epoch_time:.1f}s")
            
            # 优化TensorBoard记录 - 减少记录频率
            if self.writer is not None and (epoch % 10 == 0 or epoch <= 5):
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
                self.writer.add_scalar('F1/Train', train_f1, epoch)
                self.writer.add_scalar('F1/Val', val_f1, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 检查早停
            if do_validate:
            if self._check_early_stopping(epoch, val_loss, val_acc, val_f1):
                print(f"🛑 早停触发，训练结束于第 {epoch} 轮")
                break
        
            # 每20轮进行详细分析（增加分析频率）
            if epoch % 20 == 0 and analyzer is not None:
                try:
                    analyzer.analyze_training_progress(epoch, self.training_history)
                except Exception as e:
                    print(f"⚠️  训练分析失败: {e}")
        
        # 训练完成后的分析
        print("\n📊 训练完成，开始最终分析...")

        # 使用统一可视化管理器生成完整报告
        # 無頭環境：跳過報告/可視化生成，僅保留核心指標與模型保存
        try:
            model_info = {
                'model_type': type(self.model).__name__,
                'num_params': sum(p.numel() for p in self.model.parameters()),
                'model_size': f"{sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024:.2f} MB",
            }
            print(f"📊 模型信息: {model_info}")
            # 只保存模型
            try:
            model_path = self.viz_manager.save_model(self.model, "best_model.pth")
            print(f"💾 最佳模型已保存: {model_path}")
            except Exception:
                pass
        except Exception:
            pass

        # 如果存在測試集：此處進行一次最終測試評估
        try:
            if self.test_loader is not None:
                print("🧪 最終測試集評估...")
                test_loss, test_acc, test_f1 = self.evaluate_test()
                print(f"[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2f}%, F1={test_f1:.4f}")
        except Exception as e:
            print(f"⚠️  最終測試集評估失敗: {e}")

        # 清理旧的可视化文件（已禁用，保留所有历史文件）
        # try:
        #     self.viz_manager.cleanup_old_files(keep_recent=5)
        # except Exception as e:
        #     print(f"⚠️  清理旧文件失败: {e}")
        print("🗂️  保留所有可视化文件，不进行清理")

        # 关闭TensorBoard写入器
        if self.writer is not None:
            self.writer.close()

        print("✅ 增强优化训练完成！")
        print(f"   最佳F1分数: {self.best_val_f1:.4f}")
        print(f"   最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"   输出目录: {self.output_dir}")

        # 如果存在測試集，結束時進行一次測試評估並輸出詳細彙總
        try:
            if self.test_loader is not None:
                print("🧪 最終測試集評估...")
                test_loss, test_acc, test_f1 = self.evaluate_test()
                print(f"[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2f}%, F1={test_f1:.4f}")
                # 詳細輸出
                self._evaluate_and_summarize(self.test_loader, split_name="Test")
        except Exception as e:
            print(f"⚠️  最終測試集評估失敗: {e}")

    def _get_original_data_from_batch(self, batch):
        """从批次中安全解包原始数据和标签"""
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                data, labels, _ = batch
            elif len(batch) == 2:
                data, labels = batch
            else:
                raise ValueError(f"批次格式不正确: {len(batch)} 个元素")
        else:
            raise TypeError(f"批次类型不正确: {type(batch)}")
        return data, labels

    def _update_data_loaders(self, new_batch_size):
        """根据新的批次大小更新数据加载器"""
        num_workers = self.train_loader.num_workers if hasattr(self.train_loader, 'num_workers') else 0
        pin_memory = self.train_loader.pin_memory if hasattr(self.train_loader, 'pin_memory') else False
        
        # 重新创建训练数据加载器
        self.train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=new_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        # 重新创建验证数据加载器（不丢尾批）
        self.val_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=new_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        # 若存在测试集，同步更新
        if self.test_loader is not None and hasattr(self, 'test_loader'):
            self.test_loader = DataLoader(
                self.test_loader.dataset,
            batch_size=new_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        print(f"📊 数据加载器已更新，新批次大小: {new_batch_size}")
        self.config.DATA_CONFIG['batch_size'] = new_batch_size # 更新配置中的批次大小

def main():
    """主函数"""
    # 设置OpenBLAS线程数限制，消除警告
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    
    print("🚀 增强优化训练脚本启动")
    print("=" * 60)
    
    try:
        # 定义一个函数来运行单个实验
        def run_experiment(exp_name, exp_config):
            print(f"\n🚀 运行实验: {exp_name}")
            print("=" * 60)
            current_config = create_enhanced_config() # 获取基础配置
            
            # 更新配置
            for k, v in exp_config.get('DATA_CONFIG', {}).items():
                current_config.DATA_CONFIG[k] = v
            for k, v in exp_config.get('CNN_CONFIG', {}).items():
                current_config.CNN_CONFIG[k] = v
            for k, v in exp_config.get('TRANSFORMER_CONFIG', {}).items():
                current_config.TRANSFORMER_CONFIG[k] = v
            for k, v in exp_config.get('TRAIN_CONFIG', {}).items():
                current_config.TRAIN_CONFIG[k] = v
            # 接受 MODEL_CONFIG 覆蓋
            for k, v in exp_config.get('MODEL_CONFIG', {}).items():
                current_config.MODEL_CONFIG[k] = v

            # 允许通过环境变量覆盖 epochs / focal_gamma / cross-well，以便快速诊断
            try:
                env_epochs = os.environ.get('EPOCHS', '').strip()
                if env_epochs:
                    ep = int(env_epochs)
                    if ep > 0:
                        current_config.TRAIN_CONFIG['epochs'] = ep
                        print(f"⏱️  EPOCHS 覆盖: {ep}")
            except Exception:
                pass

            try:
                env_gamma = os.environ.get('FOCAL_GAMMA', '').strip()
                if env_gamma:
                    gv = float(env_gamma)
                    current_config.TRAIN_CONFIG['focal_gamma'] = gv
                    print(f"🎯 FOCAL_GAMMA 覆盖: {gv}")
            except Exception:
                pass

            # Cross-well split: 通过 env CONTROL_CROSS_WELL=1 启用
            custom_split = None
            try:
                if str(os.environ.get('CONTROL_CROSS_WELL', '')).strip().lower() in ('1','true','yes','y'):
                    # 简单示例：将井按名称前缀划分（可按需要调整）
                    # 这里从数据集载入后再构建split
                    pass
            except Exception:
                pass

            # 若启用评估专用模式：直接加载模型并评估
            try:
                if str(os.environ.get('EVAL_ONLY', '')).strip().lower() in ('1','true','yes','y'):
                    print("🧪 评估专用模式: 直接加载best_model并进行评估")
                    trainer = OptimizedFluidIdentificationTrainer(current_config)
                    # 覆盖none_class_id（若提供）
                    try:
                        none_env = os.environ.get('NONE_CLASS_ID', '').strip()
                        if none_env:
                            trainer.config.MODEL_CONFIG['none_class_id'] = int(none_env)
                            print(f"🔖 覆盖 none_class_id={trainer.config.MODEL_CONFIG['none_class_id']}")
                    except Exception:
                        pass
                    # 加载最近best_model
                    try:
                        best_path = os.path.join(trainer.output_dir, 'models', 'best_model.pth')
                        payload = torch.load(best_path, map_location=trainer.device)
                        trainer.model.load_state_dict(payload['model_state_dict'])
                        print(f"💾 已加载模型: {best_path}")
                    except Exception as _e:
                        print(f"⚠️  加载best_model失败: {_e}")
                    # 评估
                    try:
                        if trainer.test_loader is not None:
                            print("🧪 最終測試集評估...")
                            test_loss, test_acc, test_f1 = trainer.evaluate_test()
                            print(f"[Final Test] Loss={test_loss:.4f}, Acc={test_acc:.2f}%, F1={test_f1:.4f}")
                            trainer._evaluate_and_summarize(trainer.test_loader, split_name="Test")
                        else:
                            print("⚠️  無測試集可用，輸出驗證集彙總")
                            trainer._evaluate_and_summarize(trainer.val_loader, split_name="Val")
                    except Exception as _ee:
                        print(f"⚠️  评估失败: {_ee}")
                    return
            except Exception:
                pass

            # 创建训练器
            trainer = OptimizedFluidIdentificationTrainer(current_config)

            # 如需Cross-well：根据已加载的dataset的井名创建自定义分割并重建loader
            try:
                if str(os.environ.get('CONTROL_CROSS_WELL', '')).strip().lower() in ('1','true','yes','y'):
                    base_ds = getattr(trainer.train_loader.dataset, 'dataset', trainer.train_loader.dataset)
                    if hasattr(base_ds, 'well_names'):
                        wells = list(set(base_ds.well_names))
                        wells.sort()
                        # 将前70%井用于训练，后30%用于验证（可按需要替换为明确名单）
                        pivot = max(1, int(len(wells) * 0.7))
                        train_wells = wells[:pivot]
                        val_wells = wells[pivot:]
                        print(f"🔀 Cross-well启用: 训练井={len(train_wells)}, 验证井={len(val_wells)}")
                        from data.spectrogram_data_loader import create_spectrogram_data_loaders
                        loaders = create_spectrogram_data_loaders(
                            welldata_dir=trainer.config.DATA_PATHS.get('welldata_dir', 'welldata'),
                            curve_names=trainer.config.DATA_CONFIG['curve_names'],
                            scale_range=(5, 36),
                            time_window=32,
                            time_step=1,
                            feature_size=(64, 64),
                            batch_size=trainer.config.DATA_CONFIG['batch_size'],
                            train_ratio=0.8,
                            val_ratio=0.2,
                            test_ratio=0.0,
                            enable_cleaning=trainer.config.DATA_CONFIG['enable_cleaning'],
                            num_workers=0,
                            test_mode=False,
                            use_fluid_mapping=False,
                            custom_split={'train': train_wells, 'val': val_wells, 'test': []},
                            use_stratified_split=False,
                            random_state=42,
                            wavelet_config={
                                'curve_configs': {
                                    'AC':  {'wavelet': 'morl', 'level': 5, 'mode': 'periodic', 'time_window': 16, 'output_size': (32, 32)},
                                    'RT':  {'wavelet': 'morl', 'level': 5, 'mode': 'periodic', 'time_window': 16, 'output_size': (32, 32)},
                                    'GR':  {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                                    'SP':  {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                                    'DEN': {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)},
                                    'CNL': {'wavelet': 'sym8', 'level': 4, 'mode': 'periodic', 'time_window': 32, 'output_size': (64, 64)}
                                },
                                'overlap_ratio': 0.75,
                                'force_time_step': True,
                                'label_mode': 'majority',
                                'majority_threshold': float(trainer.config.DATA_CONFIG.get('majority_threshold', 0.70)),
                                'threshold_discard': True,
                                'enable_rare_fallback': False
                            }
                        )
                        trainer.train_loader = loaders['train']
                        trainer.val_loader = loaders['val']
                        if 'test' in loaders:
                            trainer.test_loader = loaders['test']
                        print("   ✅ 已替换为Cross-well数据加载器")
            except Exception as _cw_e:
                print(f"   ⚠️ Cross-well设置失败: {_cw_e}")
            
            # 开始训练
            trainer.train()
            print(f"✅ 实验 '{exp_name}' 完成！")

        # 定义不同的实验配置 - 专门针对精度停滞问题
        experiments = {
            "Breakthrough_Accuracy_Experiment": {
                # 突破精度停滞的专用配置
                'DATA_CONFIG': {
                    'batch_size': 128,  # 大批次
                    'enable_bsmote': False,  # 禁用BSMOTE，避免6類權重
                    'bsmote_target_ratio': 0.9,  # 高平衡比例
                    'enable_wavelet_augmentation': False,  # 禁用复杂处理
                    'use_simple_preprocessing': True  # 简化预处理
                },
                'TRAIN_CONFIG': {
                    'learning_rate': 0.005,  # 高学习率
                    'weight_decay': 1e-6,  # 极低正则化
                    'epochs': 300,  # 长训练
                    'early_stopping_patience': 50,  # 大耐心值
                    'label_smoothing': 0.01,  # 极低标签平滑
                    'use_amp': False  # 禁用混合精度确保稳定性
                },
                'CNN_CONFIG': {
                    'conv_channels': [128, 256, 512, 1024],  # 大模型
                    'dropout_rate': 0.1  # 低dropout
                },
                'TRANSFORMER_CONFIG': {
                    'd_model': 1024,  # 大维度
                    'num_layers': 8,  # 深层网络
                    'dim_feedforward': 4096  # 大FFN
                }
            },
            "Simplified_Baseline": {
                # 优化版基线实验
                'DATA_CONFIG': {
                    'batch_size': 64,  # 减小批次大小，提高梯度更新频率
                    'enable_bsmote': False,  # 禁用BSMOTE，改用抽樣+CB權重
                    'bsmote_target_ratio': 0.8,  # 提高平衡比例
                    'enable_wavelet_augmentation': False  # 暂时禁用复杂的小波增强
                },
                'TRAIN_CONFIG': {
                    'learning_rate': 0.003,  # 提高学习率
                    'weight_decay': 1e-5,    # 减少权重衰减
                    'epochs': 150,           # 增加训练轮数
                    'validation_interval': 3,  # 更频繁的验证
                    'early_stopping_patience': 20,  # 增加patience
                    'warmup_epochs': 10,     # 增加预热轮数
                    'label_smoothing': 0.02,  # 减少标签平滑
                    'use_amp': False  # 暂时禁用混合精度以确保稳定性
                },
                'MODEL_CONFIG': {
                    'use_simple_model': True
                },
                'CNN_CONFIG': {
                    'dropout_rate': 0.2  # 减少dropout
                },
                'TRANSFORMER_CONFIG': {
                    'dropout': 0.2  # 减少transformer dropout
                }
            },
            "High_Capacity_Model": {
                # 高容量模型实验
                'CNN_CONFIG': {
                    'conv_channels': [64, 128, 256, 512, 1024],
                    'dropout_rate': 0.15
                },
                'TRANSFORMER_CONFIG': {
                    'd_model': 768,
                    'num_layers': 12,
                    'nhead': 12
                },
                'TRAIN_CONFIG': {
                    'learning_rate': 0.002,
                    'weight_decay': 1e-5
                }
            }
        }

        # 僅運行簡化基線（避免默認大模型與長訓練）
        baseline_only = {
            k: v for k, v in experiments.items() if k == "Simplified_Baseline"
        }
        for name, exp_config in baseline_only.items():
            run_experiment(name, exp_config)

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
