#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督微调：Mean-Teacher + 高置信伪标签（awpd_depth_aligned_mcms）
- 加载 main/run_final_optimized_training.py 的模型构造依赖
- 从 final_optimized_best_model.pth 加载权重
- 使用 awpd_depth_aligned_mcms/train_unlabeled.npz 做一致性训练
- 在 awpd_depth_aligned_mcms/val_samples.npz 上评估 Macro-F1
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import torch.nn.functional as F

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入模型与类型
from models.lightweight_mdsc_tam import create_lightweight_model

DATA_DIR = Path(project_root) / 'awpd_depth_aligned_mcms'
BEST_MODEL = Path(project_root) / 'main' / 'final_optimized_best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 构建验证集加载
class ValDataset(td.Dataset):
    def __init__(self, data_dir: Path):
        d = np.load(data_dir / 'val_samples.npz', allow_pickle=True)
        self.h = torch.from_numpy(d['spectrogram_high']).float()  # (N, 32, 32)
        self.l = torch.from_numpy(d['spectrogram_low']).float()   # (N, 64, 64)
        self.m = torch.from_numpy(d['spectrogram_mixed']).float() # (?, ?, ?)
        # 统一到64x64
        self.h64 = F.interpolate(self.h.unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False).squeeze(1)
        self.m64 = F.interpolate(self.m.unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False).squeeze(1)
        # 6通道
        self.spec6 = torch.stack([self.h64, self.l, self.m64, self.l, self.h64, self.m64], dim=1)
        self.labels = torch.from_numpy(d['labels']).long()
        self.wells = d['well_names']
        # 预计算增强/辅助
        pre = data_dir / 'precomputed' / 'val'
        aug_list, aux_list = [], []
        for i in range(len(self.labels)):
            sid = f"{str(self.wells[i])}_{i}"
            try:
                aug = np.load(pre / 'aug_channels' / f'{sid}.npy')
                aux = np.load(pre / 'aux_vec' / f'{sid}.npy')
            except Exception:
                aug = np.zeros((2,64,64), dtype=np.float32)
                aux = np.zeros(46, dtype=np.float32)
            aug_list.append(torch.from_numpy(aug).float())
            aux_list.append(torch.from_numpy(aux).float())
        self.aug = torch.stack(aug_list, dim=0)
        self.aux = torch.stack(aux_list, dim=0)
        self.spec8 = torch.cat([self.spec6, self.aug], dim=1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.spec8[idx], self.aux[idx], self.labels[idx]

# 构建无标注集
class UnlabeledDataset(td.Dataset):
    def __init__(self, data_dir: Path):
        d = np.load(data_dir / 'train_unlabeled.npz', allow_pickle=True)
        self.h = torch.from_numpy(d['spectrogram_high']).float()
        self.l = torch.from_numpy(d['spectrogram_low']).float()
        self.m = torch.from_numpy(d['spectrogram_mixed']).float()
        self.h64 = F.interpolate(self.h.unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False).squeeze(1)
        self.m64 = F.interpolate(self.m.unsqueeze(1), size=(64,64), mode='bilinear', align_corners=False).squeeze(1)
        self.spec6 = torch.stack([self.h64, self.l, self.m64, self.l, self.h64, self.m64], dim=1)
        self.wells = d['well_names']
        pre = data_dir / 'precomputed' / 'unlabeled'
        aug_list, aux_list = [], []
        for i in range(len(self.wells)):
            sid = f"{str(self.wells[i])}_{i}"
            try:
                aug = np.load(pre / 'aug_channels' / f'{sid}.npy')
                aux = np.load(pre / 'aux_vec' / f'{sid}.npy')
            except Exception:
                aug = np.zeros((2,64,64), dtype=np.float32)
                aux = np.zeros(46, dtype=np.float32)
            aug_list.append(torch.from_numpy(aug).float())
            aux_list.append(torch.from_numpy(aux).float())
        self.aug = torch.stack(aug_list, dim=0)
        self.aux = torch.stack(aux_list, dim=0)
        self.spec8 = torch.cat([self.spec6, self.aug], dim=1)
    def __len__(self):
        return len(self.wells)
    def __getitem__(self, idx):
        return self.spec8[idx], self.aux[idx]

def build_model():
    model_config = {
        'num_classes': 5,
        'input_channels': 8,
        'use_multiscale': False,
        'conv_channels': [80, 160, 320],
        'fusion_channels': 640,
        'dropout_rate': 0.30,
        'aux_vec_dim': 46,
        'wavelet_vec_dim': 0,
        'stats_dim': 0
    }
    model = create_lightweight_model(model_config, input_channels=8).to(DEVICE)
    return model

def load_best_weights(model: torch.nn.Module, ckpt_path: Path):
    if not ckpt_path.exists():
        print(f"[警告] 未找到最佳模型权重: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    print(f"✓ 已加载最佳模型: {ckpt_path}")

@torch.no_grad()
def evaluate(model: torch.nn.Module, ds: td.Dataset):
    model.eval()
    loader = td.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    total = 0
    y_true, y_pred = [], []
    for xb, ab, yb in loader:
        xb = xb.to(DEVICE)
        ab = ab.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb, ab)
        # 兼容模型返回 (logits, aux) 或其他tuple形式
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred = torch.argmax(logits, dim=1)
        total += len(yb)
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    import numpy as _np
    from sklearn.metrics import f1_score, accuracy_score
    yt = _np.concatenate(y_true); yp = _np.concatenate(y_pred)
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro')
    wf1 = f1_score(yt, yp, average='weighted')
    print(f"评估: Acc={acc*100:.2f}%, Macro-F1={mf1:.4f}, Weighted-F1={wf1:.4f}")
    return acc, mf1, wf1

def semi_supervised_train(model: torch.nn.Module, teacher: torch.nn.Module, unlabeled_ds: UnlabeledDataset, epochs=10):
    model.train()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ema = 0.999
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=8e-5)
    loader = td.DataLoader(unlabeled_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    print(f"半监督训练: epochs={epochs}, 无标注样本={len(unlabeled_ds)}")
    for ep in range(1, epochs+1):
        total_loss = 0.0
        t0 = time.time()
        for xb, ab in loader:
            xb = xb.to(DEVICE); ab = ab.to(DEVICE)
            # 伪标签（冷启动）：用teacher获得高置信预测
            with torch.no_grad():
                t_logits = teacher(xb, ab)
                if isinstance(t_logits, (tuple, list)):
                    t_logits = t_logits[0]
                t_prob = torch.softmax(t_logits, dim=1)
                conf, pseudo = torch.max(t_prob, dim=1)
                mask = conf >= 0.80
            if mask.sum().item() == 0:
                continue
            xb = xb[mask]; ab = ab[mask]; pseudo = pseudo[mask]
            # 类别均衡采样：各类别取相同上限
            with torch.no_grad():
                classes = pseudo.unique().tolist()
            per_class_indices = []
            min_count = None
            for c in classes:
                idxs = torch.nonzero(pseudo == c, as_tuple=False).flatten()
                per_class_indices.append(idxs)
                cnt = idxs.numel()
                min_count = cnt if min_count is None else min(min_count, cnt)
            # 每类最多取k个，k为各类最小数量，且不超过批次上限
            k = int(min(32, max(1, min_count or 1)))
            keep = []
            for idxs in per_class_indices:
                if idxs.numel() <= k:
                    keep.append(idxs)
                else:
                    perm = torch.randperm(idxs.numel(), device=idxs.device)[:k]
                    keep.append(idxs[perm])
            keep = torch.cat(keep, dim=0)
            xb = xb[keep]; ab = ab[keep]; pseudo = pseudo[keep]
            # 一致性视图：加轻噪声
            xb2 = xb + 0.01*torch.randn_like(xb)
            # 前向
            logits1 = model(xb, ab)
            if isinstance(logits1, (tuple, list)):
                logits1 = logits1[0]
            with torch.no_grad():
                logits2 = teacher(xb2, ab)
                if isinstance(logits2, (tuple, list)):
                    logits2 = logits2[0]
            ce = nn.functional.cross_entropy(logits1, pseudo)
            mse = nn.functional.mse_loss(logits1, logits2)
            ramp = min(1.0, ep/10.0)
            loss = ce + 1.0*ramp*mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())*len(xb)
            # EMA更新teacher
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), model.parameters()):
                    tp.copy_(ema*tp + (1-ema)*sp)
        dt = time.time()-t0
        avg_loss = total_loss / max(1, len(unlabeled_ds))
        print(f"  [半监督] Epoch {ep}/{epochs} loss={avg_loss:.4f} time={dt:.1f}s")


def main():
    assert DATA_DIR.exists(), f"{DATA_DIR} 不存在"
    # 构建模型并加载权重
    model = build_model().to(DEVICE)
    load_best_weights(model, BEST_MODEL)
    teacher = build_model().to(DEVICE)
    teacher.load_state_dict(model.state_dict(), strict=False)

    # 构建数据集
    val_ds = ValDataset(DATA_DIR)
    unlabeled_ds = UnlabeledDataset(DATA_DIR)

    print("半监督前验证：")
    evaluate(model, val_ds)

    # 训练
    semi_supervised_train(model, teacher, unlabeled_ds, epochs=20)

    print("半监督后验证：")
    evaluate(model, val_ds)

    # 保存微调权重
    torch.save({'model_state_dict': model.state_dict()}, Path('final_optimized_best_model_semi.pth'))
    print("✓ 半监督权重已保存: final_optimized_best_model_semi.pth")

if __name__ == '__main__':
    main()
