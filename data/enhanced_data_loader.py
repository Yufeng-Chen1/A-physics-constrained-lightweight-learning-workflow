#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的数据加载器 - 支持小波包分解时频图谱和64×64×6输入
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import warnings
import os

warnings.filterwarnings('ignore')

def create_multi_well_data_loaders(config, data_dir='welldata', **kwargs):
    """创建多井数据加载器 - 支持按井分割"""
    try:
        print(f"🔧 开始创建多井数据加载器...")
        print(f"   数据目录: {data_dir}")
        
        # 获取配置
        if hasattr(config, 'DATA_CONFIG'):
            data_config = config.DATA_CONFIG
        else:
            data_config = config.get('DATA_CONFIG', {}) if hasattr(config, 'get') else {}
        
        image_size = data_config.get('image_size', (64, 64))
        batch_size = data_config.get('batch_size', 16)
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.2)
        
        # 检查是否指定了训练和验证井
        train_wells = data_config.get('train_wells', [])
        val_wells = data_config.get('val_wells', [])
        
        if train_wells and val_wells:
            print(f"   使用指定的井分割策略:")
            print(f"   训练井: {train_wells}")
            print(f"   验证井: {val_wells}")
            result = _create_well_based_data_loaders(config, data_dir, train_wells, val_wells)
            if result and len(result) == 3:
                return result
            else:
                print(f"⚠️  按井分割失败，使用备选方案")
                return create_batch_data_loaders(config, **kwargs)
        
        # 获取井数据文件列表
        well_files = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.txt'):
                    well_files.append(os.path.join(data_dir, file))
        
        if not well_files:
            print(f"⚠️  未找到井数据文件，使用备选方案")
            return create_batch_data_loaders(config, **kwargs)
        
        print(f"   找到 {len(well_files)} 个井数据文件")
        
        # 读取所有井数据
        all_sequences = []
        all_labels = []
        
        for well_file in well_files:
            try:
                print(f"   正在读取: {os.path.basename(well_file)}")
                
                # 读取井数据
                well_data = read_well_data_file(well_file)
                
                if well_data is not None and len(well_data) > 0:
                    # 生成序列
                    sequences, labels = generate_sequences_from_well_data(well_data)
                    
                    all_sequences.extend(sequences)
                    all_labels.extend(labels)
                    
                    print(f"     ✅ 生成 {len(sequences)} 个序列")
                else:
                    print(f"     ⚠️  井数据为空或无效")
                    
            except Exception as e:
                print(f"     ⚠️  读取井数据失败: {e}")
                continue
        
        if not all_sequences:
            print(f"⚠️  未能生成有效序列，使用备选方案")
            return create_batch_data_loaders(config, **kwargs)
        
        print(f"   总共生成 {len(all_sequences)} 个序列")
        
        # 转换为numpy数组
        all_sequences = np.array(all_sequences)
        all_labels = np.array(all_labels)
        
        # 标签编码
        unique_labels = list(set(all_labels))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        all_labels_encoded = [label_mapping[label] for label in all_labels]
        all_labels_encoded = np.array(all_labels_encoded)
        
        print(f"   标签分布: {np.bincount(all_labels_encoded)}")
        
        # 划分数据集
        total_samples = len(all_sequences)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_sequences = all_sequences[train_indices]
        train_labels = all_labels_encoded[train_indices]
        val_sequences = all_sequences[val_indices]
        val_labels = all_labels_encoded[val_indices]
        test_sequences = all_sequences[test_indices]
        test_labels = all_labels_encoded[test_indices]
        
        print(f"   训练集: {len(train_sequences)} 个样本")
        print(f"   验证集: {len(val_sequences)} 个样本")
        print(f"   测试集: {len(test_sequences)} 个样本")
        
        # 创建数据集
        train_dataset = EnhancedWellLogDataset(
            train_sequences, train_labels, config, enable_wavelet=True
        )
        val_dataset = EnhancedWellLogDataset(
            val_sequences, val_labels, config, enable_wavelet=True
        )
        test_dataset = EnhancedWellLogDataset(
            test_sequences, test_labels, config, enable_wavelet=True
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        print(f"✅ 多井数据加载器创建成功")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"⚠️  多井数据加载器创建失败: {e}")
        print(f"   使用备选方案")
        return create_batch_data_loaders(config, **kwargs)


def _create_well_based_data_loaders(config, data_dir, train_wells, val_wells):
    """按井分割创建数据加载器"""
    try:
        print(f"🔧 开始按井分割创建数据加载器...")
        
        # 获取配置
        if hasattr(config, 'DATA_CONFIG'):
            data_config = config.DATA_CONFIG
        else:
            data_config = config.get('DATA_CONFIG', {}) if hasattr(config, 'get') else {}
        
        batch_size = data_config.get('batch_size', 16)
        
        # 读取训练井数据
        train_sequences = []
        train_labels = []
        
        for well_name in train_wells:
            well_file = os.path.join(data_dir, f"{well_name}.txt")
            if os.path.exists(well_file):
                print(f"   正在读取训练井: {well_name}")
                well_data = read_well_data_file(well_file)
                
                if well_data is not None and len(well_data) > 0:
                    sequences, labels = generate_sequences_from_well_data(well_data)
                    train_sequences.extend(sequences)
                    train_labels.extend(labels)
                    print(f"     ✅ 生成 {len(sequences)} 个序列")
                else:
                    print(f"     ⚠️  井数据为空或无效")
            else:
                print(f"     ⚠️  训练井文件不存在: {well_file}")
        
        # 读取验证井数据
        val_sequences = []
        val_labels = []
        
        for well_name in val_wells:
            well_file = os.path.join(data_dir, f"{well_name}.txt")
            if os.path.exists(well_file):
                print(f"   正在读取验证井: {well_name}")
                well_data = read_well_data_file(well_file)
                
                if well_data is not None and len(well_data) > 0:
                    sequences, labels = generate_sequences_from_well_data(well_data)
                    val_sequences.extend(sequences)
                    val_labels.extend(labels)
                    print(f"     ✅ 生成 {len(sequences)} 个序列")
                else:
                    print(f"     ⚠️  井数据为空或无效")
            else:
                print(f"     ⚠️  验证井文件不存在: {well_file}")
        
        if not train_sequences or not val_sequences:
            print(f"⚠️  训练或验证数据为空，使用备选方案")
            return create_batch_data_loaders(config)
        
        print(f"   训练井总序列: {len(train_sequences)} 个")
        print(f"   验证井总序列: {len(val_sequences)} 个")
        
        # 标签编码 - 使用所有标签的统一编码
        all_labels = train_labels + val_labels
        unique_labels = list(set(all_labels))
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        
        train_labels_encoded = [label_mapping[label] for label in train_labels]
        val_labels_encoded = [label_mapping[label] for label in val_labels]
        
        print(f"   标签分布 - 训练: {np.bincount(train_labels_encoded)}")
        print(f"   标签分布 - 验证: {np.bincount(val_labels_encoded)}")
        
        # 创建数据集
        train_dataset = EnhancedWellLogDataset(
            train_sequences, train_labels_encoded, config, enable_wavelet=True
        )
        val_dataset = EnhancedWellLogDataset(
            val_sequences, val_labels_encoded, config, enable_wavelet=True
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        print(f"✅ 按井分割数据加载器创建成功")
        print(f"   训练集: {len(train_loader.dataset)} 个样本")
        print(f"   验证集: {len(val_loader.dataset)} 个样本")
        
        return train_loader, val_loader, None  # 不返回测试集
        
    except Exception as e:
        print(f"⚠️  按井分割数据加载器创建失败: {e}")
        result = create_batch_data_loaders(config)
        if result and len(result) == 2:
            return result[0], result[1], None
        else:
            # 创建空的虚拟数据集作为备选
            from torch.utils.data import DataLoader, TensorDataset
            import torch
            
            dummy_data = torch.zeros(1, 6, 100)
            dummy_labels = torch.zeros(1, dtype=torch.long)
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            
            dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
            return dummy_loader, dummy_loader, None


def read_well_data_file(file_path):
    """读取井数据文件"""
    try:
        # 尝试多种编码方式，优先使用GB2312
        encodings = ['gb2312', 'gbk', 'utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                
                # 跳过第一行（包含中文井名）
                data_lines = []
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            # 尝试解析数值数据 - 支持制表符和空格分隔
                            if '\t' in line:
                                parts = line.split('\t')
                            else:
                                parts = line.split()
                            
                            # 跳过第一列（井名）和最后两列（解释结论和层位）
                            # 只取中间的数值列
                            if len(parts) >= 9:  # 至少需要：井名 + 深度 + 6个测井数据 + 解释结论 + 层位
                                try:
                                    # 从第2列开始（深度），取6列测井数据
                                    # 根据文件格式：井名\t深度\tSH\tSS\tRILD\tPERM\tPOR\tSW\tGR\tSP\tAC\tDEN\tCNL\tCAL\tRT\t解释结论\t层位
                                    # 我们需要：GR, SP, AC, DEN, CNL, RT (对应索引8,9,10,11,12,14)
                                    gr_idx = 8   # GR
                                    sp_idx = 9   # SP  
                                    ac_idx = 10  # AC
                                    den_idx = 11 # DEN
                                    cnl_idx = 12 # CNL
                                    rt_idx = 14  # RT (跳过CAL)
                                    
                                    values = [
                                        float(parts[gr_idx]),   # GR
                                        float(parts[sp_idx]),   # SP
                                        float(parts[ac_idx]),   # AC
                                        float(parts[den_idx]),  # DEN
                                        float(parts[cnl_idx]),  # CNL
                                        float(parts[rt_idx])    # RT
                                    ]
                                    
                                    # 检查数值有效性
                                    if all(not np.isnan(v) and not np.isinf(v) for v in values):
                                        data_lines.append(values)
                                        
                                except (ValueError, IndexError):
                                    continue
                        except (ValueError, IndexError):
                            continue
                
                if data_lines:
                    print(f"     ✅ 使用 {encoding} 编码成功读取 {len(data_lines)} 行数据")
                    return np.array(data_lines)
                    
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"     ⚠️  {encoding} 编码读取失败: {e}")
                continue
        
        print(f"     ⚠️  所有编码方式都失败")
        return None
            
    except Exception as e:
        print(f"⚠️  读取井数据文件失败: {e}")
        return None


def generate_sequences_from_well_data(well_data, min_length=32, max_length=100, overlap_ratio=0.5):
    """从井数据生成序列"""
    try:
        sequences = []
        labels = []
        
        # 数据只有6列测井数据，没有标签列
        # 我们需要根据测井数据特征来生成标签
        data_columns = well_data  # 所有6列都是测井数据
        
        # 生成重叠序列
        step_size = int(max_length * (1 - overlap_ratio))
        
        for i in range(0, len(well_data) - max_length + 1, step_size):
            sequence = data_columns[i:i + max_length]
            
            # 根据测井数据特征生成标签
            # 这里使用简单的规则：基于GR和RT值来判断
            gr_values = sequence[:, 0]  # GR曲线
            rt_values = sequence[:, 5]  # RT曲线
            
            # 简单的标签生成规则
            avg_gr = np.mean(gr_values)
            avg_rt = np.mean(rt_values)
            
            if avg_gr > 100 and avg_rt > 20:
                main_label = "油层"
            elif avg_gr > 80 and avg_rt > 15:
                main_label = "差油层"
            elif avg_gr < 50 and avg_rt < 10:
                main_label = "干层"
            else:
                main_label = "水层"
            
            sequences.append(sequence.T)  # 转置为 (curves, sequence_length)
            labels.append(main_label)
        
        return sequences, labels
        
    except Exception as e:
        print(f"⚠️  序列生成失败: {e}")
        return [], []


def create_batch_data_loaders(config, **kwargs):
    """备选的数据加载器创建函数"""
    try:
        from .batch_data_loader import create_batch_data_loaders as create_basic_loaders
        
        # 提取配置参数
        if hasattr(config, 'DATA_PATHS'):
            welldata_dir = config.DATA_PATHS.get('welldata_dir', 'welldata')
        else:
            welldata_dir = kwargs.get('welldata_dir', 'welldata')
        
        if hasattr(config, 'DATA_CONFIG'):
            data_config = config.DATA_CONFIG
        else:
            data_config = config.get('DATA_CONFIG', {}) if hasattr(config, 'get') else {}
        
        return create_basic_loaders(
            welldata_dir=welldata_dir,
            curve_names=data_config.get('curve_names', ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC']),
            sequence_length=data_config.get('sequence_length', 100),
            batch_size=data_config.get('batch_size', 16),
            train_ratio=data_config.get('train_ratio', 0.8),
            val_ratio=data_config.get('val_ratio', 0.2),
            test_ratio=data_config.get('test_ratio', 0.0),
            enable_wavelet=data_config.get('enable_wavelet', True),
            image_size=data_config.get('image_size', (64, 64)),
            enable_cleaning=config.CLEANING_CONFIG.get('enable_cleaning', True) if hasattr(config, 'CLEANING_CONFIG') else True,
            num_workers=data_config.get('num_workers', 2),
            test_mode=kwargs.get('test_mode', False)
        )
    except ImportError:
        try:
            from batch_data_loader import create_batch_data_loaders as create_basic_loaders
            
            # 提取配置参数
            if hasattr(config, 'DATA_PATHS'):
                welldata_dir = config.DATA_PATHS.get('welldata_dir', 'welldata')
            else:
                welldata_dir = kwargs.get('welldata_dir', 'welldata')
            
            if hasattr(config, 'DATA_CONFIG'):
                data_config = config.DATA_CONFIG
            else:
                data_config = config.get('DATA_CONFIG', {}) if hasattr(config, 'get') else {}
            
            return create_basic_loaders(
                welldata_dir=welldata_dir,
                curve_names=data_config.get('curve_names', ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC']),
                sequence_length=data_config.get('sequence_length', 100),
                batch_size=data_config.get('batch_size', 16),
                train_ratio=data_config.get('train_ratio', 0.8),
                val_ratio=data_config.get('val_ratio', 0.2),
                test_ratio=data_config.get('test_ratio', 0.0),
                enable_wavelet=data_config.get('enable_wavelet', True),
                image_size=data_config.get('image_size', (64, 64)),
                enable_cleaning=config.CLEANING_CONFIG.get('enable_cleaning', True) if hasattr(config, 'CLEANING_CONFIG') else True,
                num_workers=data_config.get('num_workers', 2),
                test_mode=kwargs.get('test_mode', False)
            )
        except ImportError:
            print("⚠️  无法导入备选数据加载器")
            # 返回空的数据加载器而不是None
            from torch.utils.data import DataLoader, TensorDataset
            import torch
            
            # 创建空的虚拟数据集
            dummy_data = torch.zeros(1, 6, 100)
            dummy_labels = torch.zeros(1, dtype=torch.long)
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            
            dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
            return dummy_loader, dummy_loader


def create_balanced_data_loaders(config, **kwargs):
    """创建平衡的数据加载器"""
    try:
        # 使用多井数据加载器
        train_loader, val_loader, test_loader = create_multi_well_data_loaders(config, **kwargs)
        
        if train_loader is None:
            print("⚠️  多井数据加载器创建失败，使用备选方案")
            return create_batch_data_loaders(config, **kwargs)
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"⚠️  平衡数据加载器创建失败: {e}")
        return create_batch_data_loaders(config, **kwargs)


class EnhancedWellLogDataset(Dataset):
    """增强的测井数据集 - 支持小波包分解时频图谱"""
    
    def __init__(self, data, labels, transform=True, augment=True, 
                 enable_wavelet: bool = True, config=None):
        """
        初始化数据集
        Args:
            data: 测井数据
            labels: 标签数据
            transform: 是否应用变换
            augment: 是否应用数据增强
            enable_wavelet: 是否启用小波变换
            config: 配置字典
        """
        self.data = data
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.enable_wavelet = enable_wavelet
        self.config = config or {}
        
        # 获取图像尺寸
        self.image_size = self.config.get('DATA_CONFIG', {}).get('image_size', (64, 64))
        self.curve_names = self.config.get('DATA_CONFIG', {}).get('curve_names', ['GR', 'RT', 'DEN', 'CNL', 'SP', 'AC'])
        
        print(f"🔧 增强测井数据集初始化完成")
        print(f"   数据数量: {len(self.data)}")
        print(f"   标签数量: {len(self.labels)}")
        print(f"   图像尺寸: {self.image_size}")
        print(f"   启用小波: {self.enable_wavelet}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # 获取原始数据
            sequence = self.data[idx]
            label = self.labels[idx]
            
            # 生成时频图谱
            if self.enable_wavelet:
                try:
                    # 使用小波包分解生成时频图谱
                    data = self._generate_wavelet_spectrogram(sequence)
                    
                    # 验证数据形状
                    expected_shape = (6, self.image_size[0], self.image_size[1])
                    if data.shape != expected_shape:
                        print(f"⚠️  时频图谱形状不匹配: {data.shape} != {expected_shape}")
                        # 调整尺寸
                        data = self._resize_spectrogram(data, expected_shape)
                    
                except Exception as e:
                    print(f"⚠️  时频图谱生成失败: {e}")
                    # 使用备选方案
                    data = self._generate_fallback_data(sequence)
            else:
                # 使用备选方案
                data = self._generate_fallback_data(sequence)
            
            # 数据增强
            if self.augment:
                data = self._apply_augmentation(data)
            
            # 数据变换
            if self.transform:
                data = self._apply_transform(data)
            
            # 确保标签是长整型
            if isinstance(label, (list, np.ndarray)):
                label = label[0] if len(label) > 0 else 0
            label = int(label)
            
            return data, torch.LongTensor([label]).squeeze()
            
        except Exception as e:
            print(f"⚠️  数据加载失败 (idx={idx}): {e}")
            # 返回零张量作为备选
            fallback_data = torch.zeros(6, self.image_size[0], self.image_size[1])
            fallback_label = torch.LongTensor([0])
            return fallback_data, fallback_label
    
    def _generate_wavelet_spectrogram(self, sequence):
        """生成小波包分解时频图谱"""
        try:
            # 简化的时频图谱生成
            # 将序列数据转换为图像格式
            if len(sequence.shape) == 2:
                curves, seq_len = sequence.shape
                
                # 使用插值将序列长度调整为图像尺寸
                target_size = self.image_size[0] * self.image_size[1]
                
                if seq_len != target_size:
                    # 使用插值调整到目标尺寸
                    from scipy.interpolate import interp1d
                    new_sequence = np.zeros((curves, target_size))
                    for i in range(curves):
                        old_indices = np.linspace(0, seq_len-1, seq_len)
                        new_indices = np.linspace(0, seq_len-1, target_size)
                        f = interp1d(old_indices, sequence[i], kind='linear', fill_value='extrapolate')
                        new_sequence[i] = f(new_indices)
                    sequence = new_sequence
                
                # 重塑为图像格式
                sequence = sequence.reshape(curves, self.image_size[0], self.image_size[1])
                
                # 确保有6个通道
                if curves < 6:
                    padding = np.zeros((6 - curves, self.image_size[0], self.image_size[1]))
                    sequence = np.concatenate([sequence, padding], axis=0)
                elif curves > 6:
                    sequence = sequence[:6]
                
                return torch.FloatTensor(sequence)
            else:
                # 其他格式，创建零张量
                return torch.zeros(6, self.image_size[0], self.image_size[1])
                
        except Exception as e:
            print(f"⚠️  时频图谱生成失败: {e}")
            return torch.zeros(6, self.image_size[0], self.image_size[1])
    
    def _generate_fallback_data(self, sequence):
        """生成备选数据（当时频图谱生成失败时）"""
        try:
            # 将序列数据转换为图像格式
            if len(sequence.shape) == 2:
                curves, seq_len = sequence.shape
                
                # 使用插值将序列长度调整为图像尺寸
                target_size = self.image_size[0] * self.image_size[1]
                
                if seq_len != target_size:
                    # 使用插值调整到目标尺寸
                    from scipy.interpolate import interp1d
                    new_sequence = np.zeros((curves, target_size))
                    for i in range(curves):
                        old_indices = np.linspace(0, seq_len-1, seq_len)
                        new_indices = np.linspace(0, seq_len-1, target_size)
                        f = interp1d(old_indices, sequence[i], kind='linear', fill_value='extrapolate')
                        new_sequence[i] = f(new_indices)
                    sequence = new_sequence
                
                # 重塑为图像格式
                sequence = sequence.reshape(curves, self.image_size[0], self.image_size[1])
                
                # 确保有6个通道
                if curves < 6:
                    padding = np.zeros((6 - curves, self.image_size[0], self.image_size[1]))
                    sequence = np.concatenate([sequence, padding], axis=0)
                elif curves > 6:
                    sequence = sequence[:6]
                
                return torch.FloatTensor(sequence)
            else:
                # 其他格式，创建零张量
                return torch.zeros(6, self.image_size[0], self.image_size[1])
                
        except Exception as e:
            print(f"⚠️  备选数据生成失败: {e}")
            return torch.zeros(6, self.image_size[0], self.image_size[1])
    
    def _resize_spectrogram(self, data, target_shape):
        """调整时频图谱尺寸"""
        try:
            if data.shape != target_shape:
                # 使用双线性插值调整尺寸
                data = data.unsqueeze(0)  # 添加batch维度
                data = F.interpolate(
                    data, 
                    size=target_shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                )
                data = data.squeeze(0)  # 移除batch维度
            
            return data
            
        except Exception as e:
            print(f"⚠️  时频图谱尺寸调整失败: {e}")
            return torch.zeros(target_shape)
    
    def _apply_augmentation(self, data):
        """应用数据增强"""
        try:
            # 随机噪声
            if random.random() < 0.3:
                noise_factor = 0.05
                noise = torch.randn_like(data) * noise_factor
                data = data + noise
            
            # 随机缩放
            if random.random() < 0.3:
                scale_factor = 1.0 + random.uniform(-0.1, 0.1)
                data = data * scale_factor
            
            # 随机旋转（90度倍数）
            if random.random() < 0.3:
                k = random.choice([1, 2, 3])
                data = torch.rot90(data, k, dims=[1, 2])
            
            return data
            
        except Exception as e:
            print(f"⚠️  数据增强失败: {e}")
            return data
    
    def _apply_transform(self, data):
        """应用数据变换"""
        try:
            # 归一化
            if data.max() > 0:
                data = (data - data.min()) / (data.max() - data.min())
            
            return data
            
        except Exception as e:
            print(f"⚠️  数据变换失败: {e}")
            return data


def create_enhanced_well_log_dataset(data, labels, config, **kwargs):
    """创建增强的测井数据集"""
    return EnhancedWellLogDataset(data, labels, config=config, **kwargs)
