# 🔍 数据策略问题分析与改进方案

## ❌ 当前方案的问题

### 问题1: 数据划分不是分层的

**当前做法**（错误）:
```python
# 按井划分（不是分层划分！）
train_wells/  → 训练集
val_wells/    → 验证集
test_wells/   → 测试集
```

**问题**:
- ❌ 不同井的流体分布可能完全不同
- ❌ 例如: train_wells全是干层，val_wells全是油层
- ❌ 导致训练/验证/测试集流体分布不一致

**正确做法**（分层划分）:
```python
# 应该：从所有数据中按流体类型比例分层抽样
all_samples → 按流体类型统计 → 按比例划分到train/val/test
```

---

### 问题2: 数据增强时机错误

**当前做法**（错误）:
```python
# 我在全局配置中固定了增强倍数
augmentation = {'水层': 3, '油水层': 2, ...}
```

**问题**:
- ❌ 应该**先划分数据集**，再看训练集的分布
- ❌ 如果训练集中水层本身就多，就不需要增强3倍
- ❌ 应该根据**训练集的实际分布**动态决定增强倍数

**正确做法**:
```python
# 1. 先划分数据集（分层划分）
train, val, test = stratified_split(all_samples)

# 2. 统计训练集的流体分布
train_fluid_counts = count_fluids(train)
# 例如: {'油层': 500, '水层': 50, '干层': 800, ...}

# 3. 根据训练集分布决定增强倍数
target_count = max(train_fluid_counts.values())  # 或者目标数量
augmentation_ratios = {
    '水层': target_count / 50,   # 如果水层只有50个，增强到800
    '油层': target_count / 500,  # 油层有500个，增强到800
    ...
}
```

---

### 问题3: 多尺度采样不够丰富

**当前做法**（不够好）:
```python
# 固定3种尺度
32×32 (高频)
64×64 (低频)
48×48 (混合)

# 固定滑动步长
stride = 32  # 50%重叠
```

**问题**:
- ❌ 只有3种尺度，信息不够丰富
- ❌ 步长太大，丢失了很多中间样本
- ❌ 没有利用多尺度的组合

**改进方案**:

#### 方案A: 多尺度滑动窗口
```python
# 不同窗口大小的组合
window_sizes = [32, 48, 64, 96, 128]
strides = {
    32: 8,    # 75%重叠
    48: 12,   # 75%重叠
    64: 16,   # 75%重叠
    96: 24,   # 75%重叠
    128: 32   # 75%重叠
}

# 每个窗口大小都生成时频图谱
for window_size in window_sizes:
    for start in range(0, len(data) - window_size, strides[window_size]):
        window = data[start:start+window_size]
        # 生成时频图谱（统一resize到64×64）
        spectrogram = generate_cwt(window)
        spectrogram_64 = resize(spectrogram, (64, 64))
```

**优点**:
- ✅ 提取更多样本（5种窗口 × 更小步长）
- ✅ 捕捉不同尺度的特征
- ✅ 增加数据多样性

#### 方案B: 多分辨率时频图谱
```python
# 对同一窗口，生成多种分辨率
for window in sliding_windows:
    # 3种小波基 × 3种输出尺寸 = 9种时频图谱
    spectrograms = []
    
    for wavelet in ['morl', 'sym8', 'mexh']:
        for size in [(32, 32), (48, 48), (64, 64)]:
            spec = generate_cwt(window, wavelet=wavelet, output_size=size)
            spectrograms.append(spec)
    
    # 选择最有代表性的3-4个保存
    # 或者全部保存作为不同样本
```

**优点**:
- ✅ 同一窗口的多种表示
- ✅ 增加数据量
- ✅ 不同小波基捕捉不同特征

---

## ✅ 改进后的完整方案

### 核心思路

```
1. 收集所有数据（不按井划分）
   ↓
2. 提取样本（多尺度滑动窗口）
   ↓
3. 分层划分（按流体类型比例）
   ↓
4. 统计训练集分布
   ↓
5. 动态数据增强（根据训练集分布）
   ↓
6. 生成时频图谱
   ↓
7. 保存
```

---

### 详细步骤

#### 步骤1: 收集所有chang4+5层数据

```python
def collect_all_data():
    """不按井划分，先收集所有数据"""
    all_segments = []
    
    # 遍历所有井目录
    for well_dir in ['train_wells', 'val_wells', 'test_wells']:
        for well_file in glob(f'{well_dir}/*.txt'):
            df = load_well(well_file)
            df_chang = filter_layer(df, 'chang4+5')
            segments = segment_by_fluid(df_chang)
            all_segments.extend(segments)
    
    return all_segments
```

#### 步骤2: 多尺度滑动窗口采样

```python
def multiscale_sampling(segment):
    """多尺度滑动窗口"""
    samples = []
    
    # 5种窗口大小
    window_configs = [
        {'size': 32, 'stride': 8},    # 小窗口，密集采样
        {'size': 48, 'stride': 12},
        {'size': 64, 'stride': 16},   # 标准窗口
        {'size': 96, 'stride': 24},
        {'size': 128, 'stride': 32}   # 大窗口，捕捉长趋势
    ]
    
    for config in window_configs:
        window_size = config['size']
        stride = config['stride']
        
        for start in range(0, len(segment) - window_size + 1, stride):
            window = segment[start:start+window_size]
            
            sample = {
                'curves': extract_curves(window),
                'window_size': window_size,
                'fluid': segment['fluid'],
                # ... 其他特征
            }
            samples.append(sample)
    
    return samples
```

#### 步骤3: 分层划分（关键！）

```python
def stratified_split(all_samples, ratios=(0.7, 0.15, 0.15)):
    """分层划分：保证train/val/test的流体分布一致"""
    from sklearn.model_selection import train_test_split
    
    # 提取标签
    labels = [s['fluid'] for s in all_samples]
    
    # 第一次划分：train + (val+test)
    train_samples, temp_samples = train_test_split(
        all_samples, 
        test_size=(1 - ratios[0]),
        stratify=labels,  # ✨ 关键：分层采样
        random_state=42
    )
    
    # 第二次划分：val + test
    temp_labels = [s['fluid'] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=ratios[2] / (ratios[1] + ratios[2]),
        stratify=temp_labels,  # ✨ 关键：分层采样
        random_state=42
    )
    
    # 验证分布
    print("训练集分布:")
    print(Counter([s['fluid'] for s in train_samples]))
    print("验证集分布:")
    print(Counter([s['fluid'] for s in val_samples]))
    print("测试集分布:")
    print(Counter([s['fluid'] for s in test_samples]))
    
    return train_samples, val_samples, test_samples
```

#### 步骤4: 动态数据增强

```python
def dynamic_augmentation(train_samples):
    """根据训练集分布动态增强"""
    # 统计训练集分布
    fluid_counts = Counter([s['fluid'] for s in train_samples])
    print(f"训练集原始分布: {fluid_counts}")
    
    # 方案A: 均衡到最大类
    max_count = max(fluid_counts.values())
    
    # 方案B: 均衡到目标数量（例如每类1000个）
    target_count = 1000
    
    # 计算每个类需要增强的倍数
    augmentation_ratios = {}
    for fluid, count in fluid_counts.items():
        # 使用方案A
        ratio = max_count / count
        # 或使用方案B
        # ratio = target_count / count
        
        augmentation_ratios[fluid] = ratio
        print(f"  {fluid}: {count} 个 → 增强 {ratio:.2f} 倍 → {int(count * ratio)} 个")
    
    # 应用增强
    augmented_samples = []
    for sample in tqdm(train_samples, desc="augment"):
        fluid = sample['fluid']
        ratio = augmentation_ratios[fluid]
        
        # 添加原始样本
        augmented_samples.append(sample)
        
        # 生成增强副本
        n_augments = int(ratio) - 1
        for _ in range(n_augments):
            aug_sample = augment(sample)
            augmented_samples.append(aug_sample)
    
    print(f"增强后训练集: {len(augmented_samples)} 个")
    print(f"增强后分布: {Counter([s['fluid'] for s in augmented_samples])}")
    
    return augmented_samples
```

#### 步骤5: 多尺度时频图谱生成

```python
def generate_multiscale_spectrograms(sample):
    """根据窗口大小生成合适的时频图谱"""
    curves = sample['curves']
    window_size = sample['window_size']
    
    # 根据窗口大小调整scales
    if window_size <= 32:
        scales_high = np.arange(1, 17)
        scales_low = np.arange(1, 33)
    elif window_size <= 64:
        scales_high = np.arange(1, 33)
        scales_low = np.arange(1, 65)
    else:
        scales_high = np.arange(1, 49)
        scales_low = np.arange(1, 97)
    
    # 生成时频图谱
    spec_high = generate_cwt(curves, wavelet='morl', scales=scales_high)
    spec_low = generate_cwt(curves, wavelet='sym8', scales=scales_low)
    
    # 统一resize到64×64
    spec_high_64 = resize(spec_high, (64, 64))
    spec_low_64 = resize(spec_low, (64, 64))
    
    # 混合
    spec_mixed_64 = 0.6 * spec_high_64 + 0.4 * spec_low_64
    
    return spec_high_64, spec_low_64, spec_mixed_64
```

---

## 📊 预期改进效果

### 改进前 vs 改进后

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| **数据划分** | 按井划分（不分层） | 分层划分 ✨ |
| **流体分布** | train/val/test可能不一致 | 严格一致 ✨ |
| **数据增强** | 全局固定倍数 | 根据训练集动态调整 ✨ |
| **采样方法** | 单一窗口64，stride=32 | 5种窗口，stride更小 ✨ |
| **样本数量** | ~3000-5000 | ~8000-12000 ✨ |
| **数据多样性** | 低 | 高 ✨ |

### 样本数量估算

假设总共有3000个窗口（64大小）的原始样本：

**改进前**:
```
窗口大小: 1种 (64)
步长: 32 (50%重叠)
样本数: ~3000
增强后: ~5000 (水层×3, 油水层×2)
```

**改进后**:
```
窗口大小: 5种 (32, 48, 64, 96, 128)
步长: 更小 (75%重叠)
原始样本: ~8000 (5种窗口 × 更密集采样)
增强后: ~12000 (根据实际分布动态增强)
```

---

## 🎯 推荐的最终方案

### 参数配置

```python
CONFIG = {
    # 分层划分比例
    'split_ratios': (0.7, 0.15, 0.15),  # train, val, test
    
    # 多尺度窗口
    'window_configs': [
        {'size': 32, 'stride': 8},     # 密集采样
        {'size': 48, 'stride': 12},
        {'size': 64, 'stride': 16},    # 标准
        {'size': 96, 'stride': 24},
        {'size': 128, 'stride': 32}    # 长趋势
    ],
    
    # 数据增强策略（在划分后动态计算）
    'augmentation_strategy': 'balance_to_max',  # 或 'balance_to_target'
    'target_samples_per_class': 1000,  # 如果使用 balance_to_target
    
    # 时频图谱生成
    'wavelets': {
        'high': 'morl',
        'low': 'sym8'
    },
    'output_size': (64, 64)  # 统一输出
}
```

---

## ✅ 总结

您的观察非常准确！我的方案确实有三个关键问题：

1. ❌ **不是分层划分** → ✅ 改用sklearn的stratified_split
2. ❌ **数据增强时机错误** → ✅ 先划分再增强，根据训练集分布
3. ❌ **采样方法不够丰富** → ✅ 多尺度窗口 + 更小步长

现在我立即创建改进后的完整脚本！


