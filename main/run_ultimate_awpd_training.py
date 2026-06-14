#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWPD Training Program - Fixed Version
==========================================
Based on comprehensive diagnosis report:
1. Fixed training strategy (120 epochs, 25 patience)
2. Fixed class weights (oil-water layer x2.5)
3. Optimized hyperparameters
4. Stable training strategy
==========================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ✅ Windows console UTF-8 support (avoid encoding errors)
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Fixed random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.focal_loss import FocalLoss

# ==================== 終極優化配置 ====================
class UltimateAWPDConfig:
    """Ultimate AWPD Config - Adapted for well_split_dataset_jiyuan"""
    
    # ✅ Data path (auto-resolved relative to script location)
    DATA_DIR = Path(__file__).parent.parent / 'well_split_dataset_jiyuan'
    
    # ✅ STRONG REGULARIZATION (Prevent overfitting after epoch 6!)
    BATCH_SIZE = 32               # ✅ Stable batch size
    NUM_EPOCHS = 100              # ✅ Reduced epochs
    LEARNING_RATE = 5e-4          # ✅ REDUCED: 7e-4->5e-4 (more stable)
    WEIGHT_DECAY = 3e-4           # ✅ STRONG: 1e-5->3e-4 (prevent memorization!)
    DROPOUT = 0.25                # ✅ STRONG: 0.05->0.25 (dropout 0.05 did NOTHING!)
    LABEL_SMOOTHING = 0.1         # ✅ Added back: 0.0->0.1 (prevent overconfidence)
    EARLY_STOPPING_PATIENCE = 12  # ✅ CRITICAL: 40->12 (stop when F1 plateaus!)
    WARMUP_EPOCHS = 5             # ✅ Quick warmup
    WARMUP_START_FACTOR = 0.1     # Warmup start factor
    
    # ✅ Early stopping config (STRICT - stop overfitting early!)
    MIN_EPOCHS = 10               # ✅ REDUCED: 20->10 (allow early stopping sooner)
    WARMUP_NO_EARLY_STOP = 10     # ✅ Match MIN_EPOCHS
    MAX_TRAIN_VAL_GAP = 20.0      # ✅ STRICT: 35->20 (gap 43% is insane!)
    
    # ✅ BALANCED class weights (prevent training instability)
    # CRITICAL FIX: Weight 5.5 caused loss_ratio explosion (3.23x at epoch 14)
    # Model obsessed with 4.8% oil-water samples, ignored 95.2% majorities
    CLASS_WEIGHTS = {
        0: 1.0,   # Oil (35.3%, majority)
        1: 1.8,   # Water (10.9%, moderate minority) - REDUCED from 2.2
        2: 0.9,   # Dry (32.4%, majority) - INCREASED from 0.85 for stability
        3: 2.2,   # Poor-oil (16.7%, moderate minority) - REDUCED from 2.8
        4: 3.5,   # Oil-water (4.8%, extreme minority) - CRITICAL: REDUCED from 5.5!
    }
    # 3.5x is still strong boost (vs 1.0 baseline) but won't destabilize training
    
    # ✅ REDUCED Focal Loss gamma (prevent hard sample over-focusing)
    FOCAL_GAMMA = 1.5    # REDUCED from 2.0 (gentler on hard samples, more stable)
    
    CLASS_NAMES = ['Oil', 'Water', 'Dry', 'Poor-oil', 'Oil-water']
    NUM_CLASSES = 5
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ✅ Output directories (relative to project root)
    LOG_DIR = Path(__file__).parent.parent / 'logs' / 'awpd_ultimate'
    CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints' / 'awpd_ultimate'

# ==================== AWPD數據集 ====================
class AWPDDataset(Dataset):
    """Load well_split_dataset_jiyuan .npz format data with data augmentation"""
    
    # ⭐ Class variables: Store training set min/max (for Min-Max normalization to [0,1])
    _train_aux_min = None
    _train_aux_max = None
    
    # ✅ Minority classes that need more augmentation
    MINORITY_CLASSES = [1, 3, 4]  # Water, Poor-oil, Oil-water
    
    def __init__(self, data_dir, split='train', augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Load .npz file
        data_file = self.data_dir / f'{split}.npz'
        print(f"  Loading data file: {data_file}")
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        data = np.load(data_file, allow_pickle=True)
        
        # Extract data
        self.spectrograms = data['spectrograms']    # (N, 7, 64, 64)
        self.aux_features = data['aux_features']    # (N, 44)
        self.labels = data['labels']                # (N,)
        
        # CRITICAL FIX: Min-Max normalize auxiliary features to [0, 1]
        # Problem: Aux features std=82.7 vs Spec std=0.2 (415x difference!)
        # Solution: Normalize both to same [0, 1] range for equal gradient contribution
        if AWPDDataset._train_aux_min is None:
            train_file = self.data_dir / 'train.npz'
            train_data = np.load(train_file, allow_pickle=True)
            
            AWPDDataset._train_aux_min = train_data['aux_features'].min()
            AWPDDataset._train_aux_max = train_data['aux_features'].max()
            
            print(f"  [CRITICAL FIX] Auxiliary feature normalization:")
            print(f"    Original range: [{AWPDDataset._train_aux_min:.2f}, {AWPDDataset._train_aux_max:.2f}]")
            print(f"    Original std: {train_data['aux_features'].std():.2f}")
            print(f"    Spectrogram std: {train_data['spectrograms'].std():.4f}")
            print(f"    Scale mismatch: {train_data['aux_features'].std() / train_data['spectrograms'].std():.1f}x")
        
        self.aux_features = (self.aux_features - AWPDDataset._train_aux_min) / (AWPDDataset._train_aux_max - AWPDDataset._train_aux_min + 1e-8)
        
        print(f"  Aux features normalized to [0,1]: mean={self.aux_features.mean():.4f}, std={self.aux_features.std():.4f}")
        
        print(f"  {split} samples: {len(self.labels)}")
        print(f"  Spectrogram shape: {self.spectrograms.shape}")
        print(f"  Aux features shape: {self.aux_features.shape}")
        
        # Show class distribution
        if split == 'train':
            from collections import Counter
            label_counts = Counter(self.labels)
            print(f"  Train class distribution:")
            for label_id, count in sorted(label_counts.items()):
                fluid_name = ['Oil', 'Water', 'Dry', 'Poor-oil', 'Oil-water'][label_id]
                percentage = count / len(self.labels) * 100
                print(f"    {fluid_name}: {count} samples ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.labels)
    
    def _augment_spectrogram(self, spec, label):
        """Apply targeted augmentation (STRONGER for oil-water layer)"""
        is_minority = label in self.MINORITY_CLASSES
        is_oil_water = (label == 4)  # Oil-water layer needs special treatment
        
        # REDUCED augmentation (was too strong, preventing learning)
        if is_oil_water:
            aug_prob = 0.5  # REDUCED: 0.7 -> 0.5
        elif is_minority:
            aug_prob = 0.35  # REDUCED: 0.5 -> 0.35
        else:
            aug_prob = 0.15  # REDUCED: 0.2 -> 0.15
        
        if np.random.rand() > aug_prob:
            return spec
        
        spec_aug = spec.copy()
        
        # REDUCED noise (let model learn real features first)
        if is_oil_water:
            noise_std = 0.01  # REDUCED: 0.015 -> 0.01
        elif is_minority:
            noise_std = 0.007  # REDUCED: 0.01 -> 0.007
        else:
            noise_std = 0.003  # REDUCED: 0.005 -> 0.003
        spec_aug += np.random.randn(*spec_aug.shape) * noise_std
        
        # Masking
        if np.random.rand() < 0.2:
            mask_channel = np.random.randint(0, spec_aug.shape[0])
            spec_aug[mask_channel] *= 0.15
        
        if np.random.rand() < 0.2:
            mask_width = np.random.randint(3, 7)
            mask_start = np.random.randint(0, spec_aug.shape[2] - mask_width)
            spec_aug[:, :, mask_start:mask_start+mask_width] *= 0.15
        
        return spec_aug
    
    def _augment_aux_features(self, aux, label):
        """Apply targeted augmentation to auxiliary features"""
        is_oil_water = (label == 4)
        is_minority = label in self.MINORITY_CLASSES
        
        if is_oil_water:
            aug_prob = 0.6
            noise_std = 0.06
        elif is_minority:
            aug_prob = 0.4
            noise_std = 0.04
        else:
            aug_prob = 0.15
            noise_std = 0.02
        
        if np.random.rand() > aug_prob:
            return aux
        
        aux_aug = aux.copy()
        aux_aug += np.random.randn(*aux_aug.shape) * noise_std
        
        return aux_aug
    
    def __getitem__(self, idx):
        # Load data
        spec = self.spectrograms[idx]
        aux = self.aux_features[idx]
        label = int(self.labels[idx])
        
        # Apply augmentation during training
        if self.augment:
            spec = self._augment_spectrogram(spec, label)
            aux = self._augment_aux_features(aux, label)
        
        # Convert to tensors
        spec = torch.from_numpy(spec).float()  # (7, 64, 64)
        aux = torch.from_numpy(aux).float()    # (44,)
        
        return spec, aux, label

# ==================== Simplified Fluid Classifier (Anti-Overfitting) ====================
class SimplifiedFluidClassifier(nn.Module):
    """
    BALANCED model for fluid classification
    - 2 conv stages with MODERATE capacity (96->192 channels)
    - No BatchNorm (better train/val consistency)  
    - REDUCED dropout (0.15-0.25 vs previous 0.3-0.5)
    - ~620K parameters (balanced: not too complex, not too simple)
    """
    
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        
        # ✅ Stage 1: 7 → 96 channels (increased capacity)
        self.conv1 = nn.Sequential(
            nn.Conv2d(7, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # STRONG: 0.03->0.2 (prevent overfitting!)
            nn.MaxPool2d(2)  # 64×64 → 32×32
        )
        
        # ✅ Stage 2: 96 → 192 channels (increased capacity)
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),  # STRONG: 0.05->0.25 (prevent overfitting!)
            nn.AdaptiveAvgPool2d((1, 1))  # 32×32 → 1×1
        )
        
        # ✅ Auxiliary feature branch (increased to 96)
        self.aux_fc = nn.Sequential(
            nn.Linear(44, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # STRONG: 0.03->0.2 (prevent overfitting!)
        )
        
        # ✅ Classifier (192+96=288 input)
        self.classifier = nn.Sequential(
            nn.Linear(192 + 96, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Only ONE dropout
            nn.Linear(192, num_classes)
        )
    
    def forward(self, spec, aux):
        # Extract spectrogram features
        x = self.conv1(spec)      # (B, 96, 32, 32)
        x = self.conv2(x)         # (B, 192, 1, 1)
        spec_feat = x.view(x.size(0), -1)  # (B, 192)
        
        # Process auxiliary features
        aux_feat = self.aux_fc(aux)  # (B, 96)
        
        # Combine and classify
        combined = torch.cat([spec_feat, aux_feat], dim=1)  # (B, 288)
        output = self.classifier(combined)  # (B, num_classes)
        
        return output


# ==================== ORIGINAL COMPLEX MODEL (DEPRECATED) ====================
class AWPD_MDSC_TAM_Model(nn.Module):
    """
    ⚠️ DEPRECATED: This model is too complex for 7870 samples (2.8M parameters)
    It causes severe overfitting (train 99% vs val 48%)
    Use SimplifiedFluidClassifier instead!
    """
    
    def __init__(self, num_classes=5, dropout=0.010):
        super().__init__()
        
        # MDSC分支：多尺度特徵提取
        self.mdsc_backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(7, 64, kernel_size=3, padding=1),  # ⭐ 改為7通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.3),
            
            # Stage 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.4),
            
            # Stage 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.4),
            
            # Stage 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # 輔助特徵分支（處理44維輔助特徵）
        self.global_fc = nn.Sequential(
            nn.Linear(44, 256),  # ⭐ 改為44維輔助特徵
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        
        # TAM：跨模態注意力機制
        self.cross_attention = nn.Sequential(
            nn.Linear(512 + 256, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768),  # 512+256
            nn.Sigmoid()
        )
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512 + 256, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(192, num_classes)
        )
    
    def forward(self, spec, global_feat):
        B = spec.size(0)
        
        # MDSC特徵提取
        mdsc_feat = self.mdsc_backbone(spec)
        mdsc_feat = mdsc_feat.view(B, -1)  # (B, 512)
        
        # 全局特徵處理
        global_feat_processed = self.global_fc(global_feat)  # (B, 256)
        
        # TAM：跨模態注意力融合
        combined = torch.cat([mdsc_feat, global_feat_processed], dim=1)  # (B, 768)
        attention_weight = self.cross_attention(combined)  # (B, 768)
        
        # 加權融合
        fused_feat = combined * attention_weight
        
        # 分類
        logits = self.classifier(fused_feat)
        
        return logits

# ==================== 訓練器 ====================
class AWPDUltimateTrainer:
    """終極AWPD訓練器"""
    
    def __init__(self, config: UltimateAWPDConfig):
        self.config = config
        self.device = config.DEVICE
        
        # ⭐ 混合精度训练（关键优化！）
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("MDSC+TAM Training Program - Fixed Version")
        print("="*80)
        print(f"Dataset: {config.DATA_DIR}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Dropout: {config.DROPOUT}")
        print(f"Weight decay: {config.WEIGHT_DECAY}")
        print(f"Warmup epochs: {config.WARMUP_EPOCHS} (start factor {config.WARMUP_START_FACTOR*100:.0f}%)")
        print(f"Min epochs before early stop: {config.MIN_EPOCHS}")
        print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
        print(f"Class weights: Oil {config.CLASS_WEIGHTS[0]}, Water {config.CLASS_WEIGHTS[1]}, Dry {config.CLASS_WEIGHTS[2]}, Poor-oil {config.CLASS_WEIGHTS[3]}, Oil-water {config.CLASS_WEIGHTS[4]}")
        print(f"Focal Loss gamma: {config.FOCAL_GAMMA}")
        print(f"Total epochs: {config.NUM_EPOCHS}")
        print("="*80)
        
        # Load datasets
        print("\nLoading data...")
        train_dataset = AWPDDataset(config.DATA_DIR, split='train', augment=True)  # ✅ Enable augmentation
        valid_dataset = AWPDDataset(config.DATA_DIR, split='val', augment=False)
        test_dataset = AWPDDataset(config.DATA_DIR, split='test', augment=False)
        
        # Windows compatibility: num_workers=0
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True,
            num_workers=0,  # ⭐ Windows兼容
            pin_memory=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,  # ⭐ Windows兼容
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=0,  # ⭐ Windows兼容
            pin_memory=True
        )
        
        # 計算類別權重
        self.class_weights = self._compute_class_weights(train_dataset)
        
        print("\nCreating Simplified Fluid Classifier...")
        print("  Using simplified architecture (150K params vs 2.8M)")
        print("  No BatchNorm (better train/val consistency)")
        print("  2 conv stages (vs 4, prevents over-extraction)")
        self.model = SimplifiedFluidClassifier(
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT
        )
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        
        # ✅ Focal Loss（配合溫和的類別權重）
        self.criterion = FocalLoss(
            alpha=self.class_weights,
            gamma=config.FOCAL_GAMMA,
            label_smoothing=config.LABEL_SMOOTHING
        )
        
        # ✅ Optimizer with INCREASED momentum (stabilize training)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.95, 0.999)  # ✅ Increased: 0.86 -> 0.95 (stronger momentum)
        )
        
        # Verify learning rate
        actual_lr = self.optimizer.param_groups[0]['lr']
        print(f"\nOptimizer created:")
        print(f"   Configured LR: {config.LEARNING_RATE:.6f}")
        print(f"   Actual LR: {actual_lr:.6f}")
        assert abs(actual_lr - config.LEARNING_RATE) < 1e-9, "Learning rate mismatch!"
        
        # ✅ CRITICAL FIX: ReduceLROnPlateau (prevent premature LR decay)
        # Use warmup manually, then ReduceLROnPlateau
        self.warmup_epochs = config.WARMUP_EPOCHS
        self.warmup_start_lr = config.LEARNING_RATE * config.WARMUP_START_FACTOR
        self.base_lr = config.LEARNING_RATE
        self.current_epoch = 0
        
        # ReduceLROnPlateau: only reduce LR when validation stops improving
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',              # Monitor val_f1 (higher is better)
            factor=0.6,              # GENTLER: 0.5->0.6 (less aggressive reduction)
            patience=12,             # INCREASED: 8->12 (allow more time at each LR)
            min_lr=5e-6,             # HIGHER: 1e-6->5e-6 (don't go too low)
            threshold=0.001,         # REDUCED: 0.002->0.001 (accept smaller improvements)
            threshold_mode='abs'     # Absolute threshold
        )
        print("\nLearning rate scheduler: ReduceLROnPlateau")
        print(f"  Initial LR: {config.LEARNING_RATE} (INCREASED from 2e-4)")
        print(f"  Mode: maximize val_f1")
        print(f"  Factor: 0.6 (gentler reduction)")
        print(f"  Patience: 12 epochs (allow more exploration)")
        print(f"  Threshold: 0.001 (accept smaller improvements)")
        print(f"  Min LR: 5e-6")
        
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')  # ✅ Track best validation loss
        self.patience_counter = 0
        self.history = []
    
    def _compute_class_weights(self, dataset):
        print("\nComputing class weights...")
        
        class_counts = {i: 0 for i in range(self.config.NUM_CLASSES)}
        for label in dataset.labels:
            class_counts[label] += 1
        
        total = len(dataset)
        
        print("Train class weights:")
        for i in range(self.config.NUM_CLASSES):
            count = class_counts[i]
            percentage = count / total * 100
            weight = self.config.CLASS_WEIGHTS[i]
            print(f"  {self.config.CLASS_NAMES[i]}: {count} ({percentage:.2f}%) - Weight: {weight}")
        
        weights = torch.tensor([self.config.CLASS_WEIGHTS[i] for i in range(self.config.NUM_CLASSES)], 
                               dtype=torch.float32).to(self.device)
        return weights
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]', ncols=100)
        
        for batch_idx, (spec, global_feat, target) in enumerate(pbar):
            spec = spec.to(self.device)
            global_feat = global_feat.to(self.device)
            target = target.to(self.device)
            
            # ✅ Disable Mixup (avoid disrupting minority class learning)
            use_mixup = False
            if use_mixup:
                lam = np.random.beta(0.2, 0.2)  # Mixup lambda
                index = torch.randperm(spec.size(0)).to(self.device)
                mixed_spec = lam * spec + (1 - lam) * spec[index]
                mixed_aux = lam * global_feat + (1 - lam) * global_feat[index]
                target_a, target_b = target, target[index]
            else:
                mixed_spec, mixed_aux = spec, global_feat
            
            self.optimizer.zero_grad()
            
            # ⭐ 混合精度训练（关键！）
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(mixed_spec, mixed_aux)
                    if use_mixup:
                        loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                    else:
                        loss = self.criterion(output, target)
                
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(mixed_spec, mixed_aux)
                if use_mixup:
                    loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)
                
                if torch.isnan(loss):
                    print(f"\nWarning: NaN loss at batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, loader, desc='Valid', show_report=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'[{desc}]', ncols=100)
            for spec, global_feat, target in pbar:
                spec = spec.to(self.device)
                global_feat = global_feat.to(self.device)
                target = target.to(self.device)
                
                # ⭐ 验证也使用混合精度
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(spec, global_feat)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(spec, global_feat)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                acc = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
                pbar.set_postfix({
                    'loss': f'{total_loss/(len(all_preds)/self.config.BATCH_SIZE):.4f}',
                    'acc': f'{acc:.2f}%'
                })
        
        accuracy = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        # Calculate per-class F1 scores
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        
        if show_report:
            print("\nClassification Report:")
            print(classification_report(
                all_targets, 
                all_preds, 
                target_names=self.config.CLASS_NAMES,
                zero_division=0,
                digits=4
            ))
        
        return total_loss / len(loader), accuracy * 100, macro_f1, per_class_f1
    
    def train(self):
        print("\nStarting training...")
        print("="*80)
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # ✅ Manual warmup for first few epochs
            if epoch < self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, per_class_f1 = self.validate(self.valid_loader, desc='Valid')
            
            # ✅ ReduceLROnPlateau: step with validation metric (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step(val_f1)  # Monitor val_f1
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄歷史
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_f1': float(val_f1),
                'per_class_f1': per_class_f1.tolist(),
                'lr': current_lr
            }
            self.history.append(epoch_info)
            
            print("-"*80)
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Valid: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Macro-F1={val_f1:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # ✅ Monitor loss ratio (but DON'T stop based on it!)
            loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
            
            # DISABLED: Loss ratio early stop is UNRELIABLE for imbalanced data
            # REASON: Keeps triggering at epoch 17-20 even though model hasn't learned enough
            #         Train acc 67% = underfitting, but loss_ratio 5x says overfitting
            #         This metric is contradictory and unreliable!
            # NEW STRATEGY: Only use F1-based patience early stopping
            if loss_ratio > 5.0:
                print(f"  [INFO] Loss ratio high: {loss_ratio:.2f}x (monitoring only, not stopping)")
            
            # Signal 2: REMOVED oil-water F1 collapse check
            # REASON: Oil-water is only 4.8% of data, F1 < 0.10 happens due to
            #         data imbalance, not training failure. This was causing
            #         premature stopping in epochs 1-3.
            # NEW: Only warn, don't stop
            oil_water_f1 = per_class_f1[4]
            if oil_water_f1 < 0.10 and epoch + 1 >= self.config.MIN_EPOCHS:
                print(f"  [WARNING] Oil-water F1 low: {oil_water_f1:.3f} (expected for 4.8% minority)")
            
            # Track best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            # ✅ OVERFITTING DETECTION (ENFORCED - gap 43% is memorization!)
            train_val_gap = train_acc - val_acc
            if train_val_gap > self.config.MAX_TRAIN_VAL_GAP:
                print(f"  [WARNING] Large Train-Val gap: {train_val_gap:.1f}% (threshold: {self.config.MAX_TRAIN_VAL_GAP}%)")
                if epoch + 1 >= self.config.MIN_EPOCHS:
                    print(f"  [STOP] Severe overfitting detected! Stopping training.")
                    break
            
            # ✅ Early stopping logic (use F1 score, not accuracy)
            if epoch + 1 < self.config.MIN_EPOCHS:
                # Min epochs period: only record best
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_val_acc = val_acc
                    self._save_best_model(epoch, val_acc, val_f1, per_class_f1)
                    print(f"  [MIN-EPOCH-BEST] Epoch {epoch+1}/{self.config.MIN_EPOCHS}")
                else:
                    print(f"  [MIN-EPOCH] Epoch {epoch+1}/{self.config.MIN_EPOCHS} (no early stop)")
            else:
                # Formal training period: use F1 for early stopping
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.best_val_acc = val_acc
                    self.patience_counter = 0
                    self._save_best_model(epoch, val_acc, val_f1, per_class_f1)
                else:
                    self.patience_counter += 1
                    print(f"  [WAIT] Waiting for improvement: {self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")
                    
                    if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print(f"\nEarly stopping triggered! Best F1: {self.best_val_f1:.4f}")
                        break
            
            print("-"*80)
            
            # 保存歷史記錄
            with open(self.config.CHECKPOINT_DIR / 'training_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Test evaluation
        print("\n" + "="*80)
        print("Test Evaluation (using best model)")
        print("="*80)
        self.model.load_state_dict(torch.load(self.config.CHECKPOINT_DIR / 'best_model.pth')['model_state_dict'])
        test_loss, test_acc, test_f1, test_per_class_f1 = self.validate(self.test_loader, desc='Test', show_report=True)
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  Macro F1: {test_f1:.4f}")
        print(f"\n  Per-class F1 scores:")
        for i, (name, f1) in enumerate(zip(self.config.CLASS_NAMES, test_per_class_f1)):
            print(f"    {name}: {f1:.4f}")
        print("="*80)
    
    def _save_best_model(self, epoch, val_acc, val_f1, per_class_f1):
        """Save best model and show per-class F1"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'val_f1': val_f1,
            'per_class_f1': per_class_f1.tolist(),
            'history': self.history
        }
        torch.save(checkpoint, self.config.CHECKPOINT_DIR / 'best_model.pth')
        
        # Show per-class F1 scores
        print(f"  [BEST] New best model! Acc: {val_acc:.2f}%, Macro-F1: {val_f1:.4f}")
        print(f"      Per-class F1:")
        for i, (name, f1) in enumerate(zip(self.config.CLASS_NAMES, per_class_f1)):
            print(f"        {name}: {f1:.4f}")

# ==================== Main Program ====================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("MDSC+TAM Training Program - Fixed Version")
    print("="*80)
    print("Features:")
    print("  Dataset: well_split_dataset_jiyuan (7870 train + 2013 val + 883 test)")
    print("  Input: 7-channel spectrogram (64x64) + 44-dim auxiliary features")
    print("  Model: SimplifiedFluidClassifier (~290K params)")
    print("  Architecture: 2 conv stages + simple fusion (NO BatchNorm)")
    print("  Config: BALANCED TRAINING (Stability + Minority class learning)")
    print("  Batch: 64, LR: 2e-4, Dropout: 0.5, WD: 1e-4, Momentum: 0.95")
    print("  Class weights: Oil 1.0, Water 2.0, Dry 0.9, Poor-oil 2.5, Oil-water 4.0")
    print("  Optimizer: AdamW (reduced weight decay for minority classes)")
    print("  Focal Loss gamma: 2.0")
    print("  Data augmentation: MILD (40%/20% prob)")
    print("")
    print("  LATEST FIX: Increased minority weights (Water 1.5->2.0, Poor-oil 1.8->2.5,")
    print("              Oil-water 2.5->4.0), Reduced WD (5e-4->1e-4)")
    print("              -> Prevent minority class F1 collapse (Oil-water was 0.01!)")
    print("="*80 + "\n")
    
    try:
        config = UltimateAWPDConfig()
        trainer = AWPDUltimateTrainer(config)
        trainer.train()
        
        print("\n" + "="*80)
        print("Training process completed!")
        print("="*80)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

