# 🎯 层位对齐 + 相对深度标准化方案

## 📊 原始数据分析

### 数据结构
```
✅ 数据集划分:
   - train_wells: 12口井
   - val_wells: 4口井  
   - test_wells: 3口井

✅ 数据列（15列）:
   1. 井名
   2. 深度 ✨ (关键！)
   3-13. 测井曲线（GR, SP, SH, AT90, AC, PERM, POR, SW, CNL, DEN, RT）
   14. 解释结论 ✨ (流体类型标签)
   15. 层位 ✨ (地层信息，关键！)

✅ 关键特征:
   - 有深度列
   - 有层位列（chang3, chang4+5等）
   - 有流体标签列
```

---

## 🔍 用户方案评估

### ✅ 您的方案（非常合理！）

```python
核心思路:
1. 层位对齐: 只使用chang4+5层数据
2. 相对深度标准化: 对每个储层段归一化
3. 按深度段划分: 保障流体分布一致
4. 多尺度时频图谱生成
5. 小数类增强

评分: ⭐⭐⭐⭐⭐ (5/5)
```

### ✅ 优点分析

1. **层位对齐 - 完美解决深度偏移问题**
   ```
   问题: 不同井的绝对深度不同
   解决: 只用同一地层（chang4+5）
   效果: ✅ 同一层位的物理特性相似
   ```

2. **相对深度标准化 - 对齐层内位置**
   ```python
   rel_depth = (depth - layer_top) / (layer_bottom - layer_top)
   # 0.0 = 层顶
   # 0.5 = 层中
   # 1.0 = 层底
   
   效果: ✅ 同类储层在层内相同位置对齐
   ```

3. **多尺度时频图谱 - 提取丰富特征**
   ```
   高频(32×32): 局部变化细节
   低频(64×64): 全局趋势特征
   混合: 高低频互补
   
   效果: ✅ 多尺度特征融合，提升识别能力
   ```

4. **流体分布均衡 - 防止分布偏移**
   ```
   问题: 训练集70%准确率，验证集30%
   原因: 训练/验证分布不一致
   解决: 按流体比例分层抽样
   
   效果: ✅ 消除Distribution Shift
   ```

---

## 💡 改进建议

### 1. 层位筛选策略优化

**您的方案**: 只用chang4+5层

**建议**: 
```python
# 方案A（保守，您的方案）: 单层
layers_to_use = ['chang4+5']

# 方案B（推荐）: 多层对齐
# 如果chang4+5样本不足，可以包含其他层
# 但每层单独处理，层内归一化
layers_to_use = ['chang4+5', 'chang3', 'chang6']
# 为每层添加层ID编码，让模型知道是哪一层

# 实施建议: 先用方案A，如果样本足够就不需要方案B
```

### 2. 相对深度计算细化

**您的方案**: 基本的相对深度

**建议**: 增加更丰富的深度特征
```python
# 基础特征
rel_depth = (depth - layer_top) / (layer_bottom - layer_top)

# 扩展特征（可选）
features = {
    'rel_depth': rel_depth,  # 相对深度 [0,1]
    'abs_depth': depth / 3000.0,  # 归一化绝对深度
    'depth_from_top': depth - layer_top,  # 距层顶距离
    'depth_from_bottom': layer_bottom - depth,  # 距层底距离
    'layer_thickness': layer_bottom - layer_top,  # 层厚度
    'relative_position_sin': np.sin(2*np.pi*rel_depth),  # 周期编码
    'relative_position_cos': np.cos(2*np.pi*rel_depth),  # 周期编码
}

# 推荐: 至少使用rel_depth + layer_thickness
```

### 3. 深度段划分策略优化

**您的方案**: 按深度段划分数据集

**关键问题**: 如何划分？

**推荐方案**:
```python
# 方案A: 按井划分（已经做了，保持不变）
# train_wells, val_wells, test_wells 已按井分开
# ✅ 优点: 测试井间泛化能力
# ⚠️ 需确保: 每个数据集中chang4+5层的流体分布相似

# 方案B: 井内深度段混合（如果单层样本不足）
# 对每口井的chang4+5层，按深度均匀采样到train/val/test
# ⚠️ 缺点: 可能造成数据泄露（同一井的相邻样本被分到不同集）

# 推荐: 方案A（保持按井划分）
# 但需要检查并平衡各数据集的流体分布
```

### 4. 多尺度时频图谱生成优化

**您的方案**:
```
高频: Morlet, 4层, 32×32
低频: Sym8, 3层, 64×64  
混合: Morlet+Sym8拼接裁剪 → 64×64
```

**评估**: ✅ 非常合理！

**细微建议**:
```python
# 1. 混合模式优化
# 原方案: 64×128拼接后裁剪 → 64×64
# 问题: 裁剪会丢失信息

# 建议方案A: 加权融合
mixed = 0.6 * high_freq_upsampled + 0.4 * low_freq
# ✅ 保留高低频信息

# 建议方案B: 通道拼接（最优）
# 不要拼接后裁剪，而是将高低频作为不同通道
channels = [high_freq_32, low_freq_64, mixed_48]
# resize to same size then stack
# ✅ 让模型自己学习如何融合

# 2. 参数调整
高频: Morlet, 分解层数3→4 ✅
低频: Sym8, 分解层数3 ✅  
建议: 尝试不同小波基的组合
  - Morlet: 高频敏感 ✅
  - Sym8: 平滑低频 ✅
  - 可选: mexh（墨西哥帽）、cgau（复高斯）
```

### 5. 数据增强策略优化

**您的方案**:
```
时移: ±5%深度偏移
频移: ±10%频率平移
噪声: SNR=20dB
```

**评估**: ✅ 基本合理

**建议优化**:
```python
# 1. 分类别增强（您已提到，很好！）
# 只增强小数类（水层、油水层）

# 2. 增强参数调整
augmentation_params = {
    '水层': {  # 最稀有
        'time_shift': 0.08,  # ±8%（更激进）
        'freq_shift': 0.15,  # ±15%
        'noise_snr': 18,  # 稍强噪声
        'mixup_prob': 0.3,  # 30%概率Mixup
        'copies': 3  # 生成3倍样本
    },
    '油水层': {  # 次稀有
        'time_shift': 0.05,
        'freq_shift': 0.10,
        'noise_snr': 20,
        'mixup_prob': 0.2,
        'copies': 2
    },
    '差油层': {  # 中等
        'time_shift': 0.05,
        'freq_shift': 0.10,
        'noise_snr': 20,
        'mixup_prob': 0.1,
        'copies': 1.5
    },
    '油层': {  # 多数类
        'copies': 1  # 不增强或少增强
    },
    '干层': {  # 多数类
        'copies': 1
    }
}

# 3. 物理约束（关键！）
# ⚠️ 不要破坏物理意义
# 例如: 油层的RT应该高，增强时保持这个关系
# 建议: 只做轻微扰动，不要改变曲线趋势
```

### 6. 流体分布平衡策略

**问题**: 如何保证train/val/test的流体分布相似？

**方案**:
```python
# 步骤1: 先统计每口井chang4+5层的流体分布
well_fluid_distribution = {
    '涧39': {'油层': 100, '水层': 20, '干层': 150, ...},
    '耿181': {...},
    ...
}

# 步骤2: 计算目标分布（全数据集平均）
target_distribution = {
    '油层': 0.25,
    '水层': 0.10,
    '干层': 0.35,
    '差油层': 0.20,
    '油水层': 0.10
}

# 步骤3: 对每个数据集，根据井的流体分布加权采样
# 例如: val_wells中如果'水层'偏少，就多采样含水层的井

# 步骤4: 数据增强时优先增强稀有类
# 最终目标: train/val/test的流体比例都接近target_distribution

# 容忍度: ±5%以内
```

---

## 🚀 完整实施方案

### 阶段1: 数据预处理（1-2小时）

```python
# 脚本: generate_depth_aligned_awpd_data.py

def main():
    # 1. 加载所有井数据
    train_data = load_wells('train_wells')
    val_data = load_wells('val_wells')
    test_data = load_wells('test_wells')
    
    # 2. 筛选chang4+5层
    train_chang = filter_layer(train_data, 'chang4+5')
    val_chang = filter_layer(val_data, 'chang4+5')
    test_chang = filter_layer(test_data, 'chang4+5')
    
    # 3. 计算每口井每个储层段的层顶/层底
    layer_boundaries = compute_layer_boundaries(all_data)
    # 输出: {'涧39_chang4+5_seg1': {'top': 2230.0, 'bottom': 2245.5, 'fluid': '油层'}, ...}
    
    # 4. 计算相对深度
    for sample in all_samples:
        segment_id = sample['segment_id']
        depth = sample['深度']
        top = layer_boundaries[segment_id]['top']
        bottom = layer_boundaries[segment_id]['bottom']
        
        sample['relative_depth'] = (depth - top) / (bottom - top)
        sample['layer_thickness'] = bottom - top
        sample['abs_depth_norm'] = depth / 3000.0
    
    # 5. 统计流体分布
    check_fluid_distribution(train_chang, val_chang, test_chang)
    
    # 6. 分层抽样平衡（如需要）
    train_balanced = stratified_sample(train_chang, target_dist)
    val_balanced = stratified_sample(val_chang, target_dist)
    test_balanced = stratified_sample(test_chang, target_dist)
    
    return train_balanced, val_balanced, test_balanced
```

### 阶段2: 多尺度时频图谱生成（2-3小时）

```python
# 脚本: generate_multiscale_spectrograms.py

def generate_sample(well_data, sample_window=64):
    """
    为每个样本窗口生成3种时频图谱
    """
    # 提取窗口数据
    window_curves = extract_window(well_data, center_idx, window=sample_window)
    # window_curves: shape=(64, 11)  # 11条测井曲线
    
    # 1. 高频图谱（32×32）
    spec_high = generate_high_freq_spectrogram(
        window_curves,
        wavelet='morl',  # Morlet
        scales=np.arange(1, 33),  # 32个尺度
        output_size=(32, 32)
    )
    
    # 2. 低频图谱（64×64）
    spec_low = generate_low_freq_spectrogram(
        window_curves,
        wavelet='sym8',  # Symlet8
        scales=np.arange(1, 65),  # 64个尺度
        output_size=(64, 64)
    )
    
    # 3. 混合图谱（48×48）
    spec_mixed = generate_mixed_spectrogram(
        window_curves,
        wavelet_high='morl',
        wavelet_low='sym8',
        output_size=(48, 48)
    )
    
    return {
        'spectrogram_high': spec_high,  # (32, 32)
        'spectrogram_low': spec_low,    # (64, 64)
        'spectrogram_mixed': spec_mixed,  # (48, 48)
        'relative_depth': sample['relative_depth'],  # ✨ 关键！
        'layer_thickness': sample['layer_thickness'],
        'abs_depth_norm': sample['abs_depth_norm'],
        'well_name': sample['井名'],
        'label': sample['label']
    }
```

### 阶段3: 数据增强（1小时）

```python
# 脚本: augment_minority_classes.py

def augment_dataset(samples, target_balance=None):
    """
    增强小数类样本
    """
    # 统计当前分布
    class_counts = count_classes(samples)
    
    # 确定增强策略
    if target_balance is None:
        # 均衡策略: 所有类别样本数相同
        target_count = max(class_counts.values())
    else:
        # 自定义策略
        target_count = calculate_target_counts(class_counts, target_balance)
    
    augmented_samples = []
    
    for class_label, current_count in class_counts.items():
        class_samples = [s for s in samples if s['label'] == class_label]
        
        if current_count < target_count[class_label]:
            # 需要增强
            n_augment = target_count[class_label] - current_count
            
            for _ in range(n_augment):
                # 随机选一个样本
                original = random.choice(class_samples)
                
                # 增强
                augmented = apply_augmentation(
                    original,
                    time_shift_range=0.05,
                    freq_shift_range=0.10,
                    noise_snr=20
                )
                
                augmented_samples.append(augmented)
    
    return samples + augmented_samples
```

### 阶段4: 保存数据集（30分钟）

```python
# 保存到新目录: awpd_depth_aligned_data/

def save_dataset(samples, output_dir, split='train'):
    """
    保存时频图谱 + 深度特征 + 元数据
    """
    output_dir = Path(output_dir) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分离数据
    spectrograms_high = []
    spectrograms_low = []
    spectrograms_mixed = []
    labels = []
    well_names = []
    relative_depths = []
    layer_thicknesses = []
    abs_depths_norm = []
    
    for sample in samples:
        spectrograms_high.append(sample['spectrogram_high'])
        spectrograms_low.append(sample['spectrogram_low'])
        spectrograms_mixed.append(sample['spectrogram_mixed'])
        labels.append(sample['label'])
        well_names.append(sample['well_name'])
        relative_depths.append(sample['relative_depth'])  # ✨
        layer_thicknesses.append(sample['layer_thickness'])  # ✨
        abs_depths_norm.append(sample['abs_depth_norm'])  # ✨
    
    # 保存NPZ
    np.savez(
        output_dir / f'{split}_samples.npz',
        spectrogram_high=np.array(spectrograms_high),
        spectrogram_low=np.array(spectrograms_low),
        spectrogram_mixed=np.array(spectrograms_mixed),
        labels=np.array(labels),
        well_names=np.array(well_names),
        relative_depths=np.array(relative_depths),  # ✨ 关键！
        layer_thicknesses=np.array(layer_thicknesses),  # ✨
        abs_depths_norm=np.array(abs_depths_norm)  # ✨
    )
    
    print(f"✅ 保存 {split} 数据集: {len(samples)} 样本")
```

### 阶段5: 修改训练脚本（30分钟）

```python
# 修改: main/run_final_optimized_training.py

class AWPDDatasetWithDepth(torch.utils.data.Dataset):
    """新数据集类，包含深度特征"""
    
    def __init__(self, data_dir, split='train'):
        data = np.load(data_dir / f'{split}_samples.npz', allow_pickle=True)
        
        # 加载时频图谱
        self.spec_high = torch.from_numpy(data['spectrogram_high']).float()
        self.spec_low = torch.from_numpy(data['spectrogram_low']).float()
        self.spec_mixed = torch.from_numpy(data['spectrogram_mixed']).float()
        
        # 加载深度特征 ✨
        self.relative_depths = torch.from_numpy(data['relative_depths']).float()
        self.layer_thicknesses = torch.from_numpy(data['layer_thicknesses']).float()
        self.abs_depths_norm = torch.from_numpy(data['abs_depths_norm']).float()
        
        # 加载标签
        self.labels = torch.from_numpy(data['labels']).long()
        self.well_names = data['well_names']
    
    def __getitem__(self, idx):
        # 拼接6通道时频图谱
        spec_6ch = torch.stack([
            F.interpolate(self.spec_high[idx].unsqueeze(0).unsqueeze(0), (64, 64)).squeeze(),
            self.spec_low[idx],
            F.interpolate(self.spec_mixed[idx].unsqueeze(0).unsqueeze(0), (64, 64)).squeeze(),
            # ... （共6通道）
        ], dim=0)
        
        # 深度特征向量 ✨
        depth_features = torch.tensor([
            self.relative_depths[idx],
            self.layer_thicknesses[idx],
            self.abs_depths_norm[idx]
        ], dtype=torch.float32)
        
        # 合并为辅助特征
        aux_features = torch.cat([
            depth_features,  # 3维深度特征 ✨
            # ... 其他57维特征
        ], dim=0)  # 总共60维
        
        return spec_6ch, self.labels[idx], {
            'aux_vec': aux_features,
            'well_name': self.well_names[idx]
        }
```

---

## 📊 预期效果

### 基线（当前）vs 优化后

| 指标 | 当前（无深度） | 优化后（有深度对齐） | 提升 |
|------|--------------|-------------------|------|
| 验证准确率 | 30-36% | **45-55%** | +15-20% ✨ |
| 训练准确率 | 70% | **55-65%** | -5% (减少过拟合) |
| 训练/验证差距 | 40% | **<15%** | -25% (关键改善) |
| Macro-F1 | 0.30-0.32 | **0.42-0.50** | +0.12-0.18 |
| 水层F1 | 0.08-0.15 | **0.25-0.35** | +0.15-0.25 |
| 油水层F1 | 0.25-0.35 | **0.35-0.45** | +0.10 |

---

## ✅ 总结

### 您的方案评分: ⭐⭐⭐⭐⭐

**核心亮点**:
1. ✅ 层位对齐（chang4+5） - 解决深度偏移
2. ✅ 相对深度标准化 - 对齐层内位置
3. ✅ 多尺度时频图谱 - 丰富特征
4. ✅ 流体分布平衡 - 防止偏移
5. ✅ 小数类增强 - 平衡样本

**建议优化**:
1. 💡 深度特征多样化（相对+绝对+层厚）
2. 💡 混合图谱用通道拼接而非裁剪
3. 💡 增强参数按类别细化
4. 💡 物理约束下的增强

**实施优先级**:
1. 🔥 P0: 层位筛选 + 相对深度计算
2. 🔥 P0: 多尺度时频图谱生成
3. 🔥 P1: 数据增强 + 分布平衡
4. ⚡ P2: 训练脚本修改

**预计工作量**: 4-6小时
**预期提升**: 验证准确率从30-36%提升到**45-55%**

---

## 🚀 下一步

我现在立即为您创建完整的数据生成脚本！

**生成的脚本**:
1. `generate_depth_aligned_awpd_data.py` - 主脚本
2. `depth_alignment_utils.py` - 工具函数
3. `multiscale_spectrogram_generator.py` - 时频图谱生成
4. `data_augmentation.py` - 数据增强

准备好了吗？我开始创建！🎯


