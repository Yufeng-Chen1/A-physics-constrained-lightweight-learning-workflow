import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Optional, Dict


class TXTWellLogDataset(Dataset):
    """
    TXT格式測井曲線數據集
    用於加載和預處理txt格式的測井數據
    只加載有解釋結論標籤的數據行
    支持多種txt格式：
    1. 空格分隔的數值數據
    2. 製表符分隔的數值數據
    3. 逗號分隔的數值數據
    """
    
    def __init__(self, data_path: str, sequence_length: int = 100, 
                 curve_names: List[str] = None, delimiter: str = None, 
                 has_header: bool = True, transform=None, 
                 require_labels: bool = True):
        """
        初始化TXT數據集
        Args:
            data_path: txt數據文件路徑
            sequence_length: 序列長度
            curve_names: 測井曲線名稱列表
            delimiter: 分隔符（None為自動檢測）
            has_header: 是否有標題行
            transform: 數據變換
            require_labels: 是否要求必須有標籤（默認True）
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.delimiter = delimiter
        self.has_header = has_header
        self.transform = transform
        self.require_labels = require_labels
        
        # 默認測井曲線名稱
        if curve_names is None:
            curve_names = ['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
        self.curve_names = curve_names
        
        # 加載數據
        self.data, self.labels = self._load_txt_data()
        
        # 數據預處理
        self._preprocess_data()
    
    def _detect_delimiter(self, file_path: str) -> str:
        """自動檢測分隔符"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # 常見分隔符
        delimiters = ['\t', ',', ';', '|', ' ']
        
        for delim in delimiters:
            if delim in first_line:
                parts = first_line.split(delim)
                if len(parts) >= 7:  # 至少7列（6列測井曲線數據 + 1列標籤）
                    return delim
        
        # 如果沒有檢測到明確的分隔符，使用空格
        return ' '
    
    def _load_txt_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加載TXT格式數據，只保留有標籤的數據行"""
        try:
            # 自動檢測分隔符
            if self.delimiter is None:
                self.delimiter = self._detect_delimiter(self.data_path)
            
            print(f"✅ 使用分隔符: '{self.delimiter}'")
            
            # 嘗試不同的編碼方式讀取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            data_content = None
            
            for encoding in encodings:
                try:
                    with open(self.data_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    data_content = lines
                    print(f"✅ 成功使用編碼: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if data_content is None:
                raise ValueError("無法讀取文件，請檢查文件編碼")
            
            # 解析數據
            if self.has_header:
                header_line = data_content[0].strip()
                headers = header_line.split(self.delimiter)
                data_lines = data_content[1:]
                print(f"✅ 檢測到標題行: {headers}")
            else:
                data_lines = data_content
                # 生成默認標題
                headers = [f'Curve_{i}' for i in range(7)]  # 7列：6列曲線 + 1列標籤
                print(f"✅ 使用默認標題: {headers}")
            
            # 解析數值數據，只保留有標籤的行
            parsed_data = []
            valid_labels = []
            skipped_count = 0
            
            for line_num, line in enumerate(data_lines, start=2 if self.has_header else 1):
                line = line.strip()
                if line and not line.startswith('#'):  # 跳過空行和註釋行
                    try:
                        values = [x.strip() for x in line.split(self.delimiter) if x.strip()]
                        if len(values) >= 17:  # 檢查是否有足夠的列數
                            # 根據實際數據格式，選擇六條主要的測井曲線
                            # 實際格式：井名, 深度, SH, SS, RILD, PERM, POR, SW, GR, SP, AC, DEN, CNL, CAL, RT, 解釋結論, 層位
                            
                            # 選擇6個主要的測井曲線：GR(第8列), SP(第9列), AC(第10列), DEN(第12列), CNL(第13列), RT(第14列)
                            curve_indices = [7, 8, 9, 11, 12, 13]  # GR, SP, AC, DEN, CNL, RT
                            curve_names = ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
                            
                            # 提取測井曲線數據
                            curve_values = []
                            for i, idx in enumerate(curve_indices):
                                if idx < len(values):
                                    try:
                                        curve_values.append(float(values[idx]))
                                    except ValueError:
                                        # 如果轉換失敗，使用0.0
                                        print(f"⚠️ 第{line_num}行第{idx+1}列({curve_names[i]})不是有效數值: {values[idx]}，使用0.0")
                                        curve_values.append(0.0)
                                else:
                                    curve_values.append(0.0)
                            
                            # 確保有6個值
                            while len(curve_values) < 6:
                                curve_values.append(0.0)
                            
                            # 檢查標籤列（解釋結論，第15列）
                            label_value = values[15] if len(values) > 15 else "未知"
                            # 簡單的標籤驗證：非空且不是純數字
                            if label_value and label_value.strip() and not label_value.strip().isdigit():
                                parsed_data.append(curve_values)
                                valid_labels.append(label_value)
                            else:
                                skipped_count += 1
                                if skipped_count <= 5:  # 只顯示前5個跳過的行
                                    print(f"⚠️ 第{line_num}行標籤無效，跳過: {label_value}")
                                elif skipped_count == 6:
                                    print(f"⚠️ ... 還有更多無效標籤行被跳過")
                        else:
                            skipped_count += 1
                            if skipped_count <= 5:
                                print(f"⚠️ 第{line_num}行列數不足，跳過: {len(values)}列")
                    except Exception as e:
                        skipped_count += 1
                        if skipped_count <= 5:
                            print(f"⚠️ 第{line_num}行解析失敗: {e}")
                        continue
            
            if not parsed_data:
                raise ValueError("沒有找到有效的帶標籤數據行")
            
            print(f"✅ 成功加載有效數據: {len(parsed_data)}行")
            if skipped_count > 0:
                print(f"⚠️ 跳過了 {skipped_count} 行無效數據")
            
            # 轉換為numpy數組
            data_array = np.array(parsed_data)
            print(f"✅ 測井曲線數據形狀: {data_array.shape}")
            
            # 檢查數據有效性
            data_array = self._validate_curve_data(data_array)
            
            # 轉換為標準格式 (N, 6)
            data = data_array
            
            # 處理標籤
            labels = self._process_labels(valid_labels)
            
            return data, labels
            
        except Exception as e:
            print(f"❌ TXT數據加載失敗: {e}")
            raise
    
    def _validate_curve_data(self, curve_data: np.ndarray) -> np.ndarray:
        """驗證測井曲線數據的有效性"""
        # 檢查是否有無窮大或NaN值
        curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 檢查數值範圍 - 現在數據格式是 (N, 6)
        for i in range(curve_data.shape[1]):  # 遍歷6個測井曲線
            curve = curve_data[:, i]
            # 移除異常值（超過3個標準差）
            mean_val = np.mean(curve)
            std_val = np.std(curve)
            if std_val > 0:
                threshold = 3 * std_val
                curve_data[:, i] = np.clip(curve, mean_val - threshold, mean_val + threshold)
        
        return curve_data
    
    def _process_labels(self, raw_labels: List[str]) -> np.ndarray:
        """處理標籤數據（統一為5類：油層/水層/乾層/差油層/油水同層）"""
        try:
            class_names = ['油層', '水層', '乾層', '差油層', '油水同層']
            text_to_class = {
                '油層': '油層', '水層': '水層', '乾層': '乾層', '干层': '乾層',
                '差油層': '差油層', '差油层': '差油層', '油水同層': '油水同層', '油水同层': '油水同層',
                'oil': '油層', 'water': '水層', 'dry': '乾層', 'poor-oil': '差油層', 'oil-water': '油水同層',
                # 若歷史上存在數字編碼，可選擇少量兼容
                '0': '油層', '2': '水層', '3': '乾層'
            }
            mapped = []
            dropped = 0
            for l in raw_labels:
                key = str(l).strip()
                target = text_to_class.get(key)
                if target is None:
                    dropped += 1
                    continue
                mapped.append(target)
            if not mapped:
                raise ValueError('沒有可映射到5類的標籤，請檢查標籤內容')
            index_map = {name: idx for idx, name in enumerate(class_names)}
            labels = np.array([index_map[m] for m in mapped], dtype=int)
            if dropped:
                print(f"⚠️ 有 {dropped} 條樣本標籤不屬於5類，已過濾")
            print(f"✅ 標籤統一為5類完成，分佈: {np.bincount(labels, minlength=len(class_names))}")
        except Exception as e:
            print(f"⚠️ 標籤處理失敗: {e}，使用默認標籤")
            labels = np.zeros(len(raw_labels), dtype=int)
        return labels
    
    def _preprocess_data(self):
        """數據預處理"""
        # 處理缺失值
        self.data = self._handle_missing_values(self.data)
        
        # 標準化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 編碼標籤
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        # 創建序列數據
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """處理缺失值"""
        # 使用前向填充
        df = pd.DataFrame(data)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')  # 處理開頭的缺失值
        df = df.fillna(0)  # 如果還有缺失值，用0填充
        
        return df.values
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """創建序列數據"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(self.data) - self.sequence_length + 1):
            sequence = self.data[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length // 2]  # 使用序列中間的標籤
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取數據項"""
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # 轉換為張量
        sequence_tensor = torch.FloatTensor(sequence).T  # (C, L)
        label_tensor = torch.LongTensor([label])
        
        # 重塑為2D圖像格式 (C, H, W)
        # 將一維測井曲線重塑為方形圖像
        h = w = int(np.sqrt(self.sequence_length))
        if h * w < self.sequence_length:
            h = w = int(np.ceil(np.sqrt(self.sequence_length)))
        
        # 填充到方形
        padded_length = h * w
        if sequence_tensor.shape[1] < padded_length:
            padding = torch.zeros(sequence_tensor.shape[0], padded_length - sequence_tensor.shape[1])
            sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
        
        # 重塑為圖像格式 - 每個通道保持一維，但整體為二維圖像
        # 這樣小波變換模組可以將每個通道轉換為時頻圖譜
        sequence_tensor = sequence_tensor.view(sequence_tensor.shape[0], h, w)
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor, label_tensor.squeeze()


class WellLogDataset(Dataset):
    """
    測井曲線數據集
    用於加載和預處理測井數據
    """
    
    def __init__(self, data_path: str, sequence_length: int = 100, 
                 curve_names: List[str] = None, transform=None):
        """
        初始化數據集
        Args:
            data_path: 數據文件路徑
            sequence_length: 序列長度
            curve_names: 測井曲線名稱列表
            transform: 數據變換
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 默認測井曲線名稱
        if curve_names is None:
            curve_names = ['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
        self.curve_names = curve_names
        
        # 加載數據
        self.data, self.labels = self._load_data()
        
        # 數據預處理
        self._preprocess_data()
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加載數據"""
        try:
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.xlsx'):
                df = pd.read_excel(self.data_path)
            else:
                raise ValueError("不支持的數據格式，請使用CSV或Excel文件")
            
            print(f"✅ 成功加載數據文件: {self.data_path}")
            print(f"✅ 數據形狀: {df.shape}")
            print(f"✅ 列名: {list(df.columns)}")
            
            # 提取測井曲線數據
            curve_data = []
            curve_mapping = {
                'GR': 'GR',
                'SP': 'SP', 
                'RES': 'RT',  # 電阻率對應RT列
                'DEN': 'DEN',
                'NEU': 'CNL',  # 中子對應CNL列
                'SON': 'AC'   # 聲波對應AC列
            }
            
            for curve in self.curve_names:
                mapped_curve = curve_mapping.get(curve, curve)
                if mapped_curve in df.columns:
                    curve_data.append(df[mapped_curve].values)
                    print(f"✅ 找到曲線: {mapped_curve}")
                else:
                    # 如果沒有該曲線，用零填充
                    print(f"⚠️ 未找到曲線 {mapped_curve}，使用零填充")
                    curve_data.append(np.zeros(len(df)))
            
            # 轉換為numpy數組
            data = np.array(curve_data).T  # (N, C)
            print(f"✅ 測井曲線數據形狀: {data.shape}")
            
            # 提取標籤 - 使用"解釋結論"列作為標籤
            if '解釋結論' in df.columns:
                # 將中文標籤轉換為數字標籤
                label_mapping = {
                    '油層': 0,
                    '氣層': 1, 
                    '水層': 2,
                    '乾層': 3,
                    '油氣層': 0,  # 歸為油層
                    '氣水層': 1,  # 歸為氣層
                    '油水層': 0   # 歸為油層
                }
                raw_labels = df['解釋結論'].values
                labels = np.array([label_mapping.get(str(label), 0) for label in raw_labels])
                print(f"✅ 找到標籤列: 解釋結論")
                print(f"✅ 標籤分布: {np.bincount(labels)}")
            elif 'fluid_type' in df.columns:
                labels = df['fluid_type'].values
                print(f"✅ 找到標籤列: fluid_type")
            elif 'label' in df.columns:
                labels = df['label'].values
                print(f"✅ 找到標籤列: label")
            else:
                # 如果沒有標籤，創建虛擬標籤
                print("⚠️ 未找到標籤列，使用虛擬標籤")
                labels = np.zeros(len(df), dtype=int)
            
            return data, labels
            
        except Exception as e:
            print(f"❌ 數據加載失敗: {e}")
            raise
    
    def _preprocess_data(self):
        """數據預處理"""
        # 處理缺失值
        self.data = self._handle_missing_values(self.data)
        
        # 標準化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 編碼標籤
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        # 創建序列數據
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """處理缺失值"""
        # 使用前向填充
        df = pd.DataFrame(data)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')  # 處理開頭的缺失值
        df = df.fillna(0)  # 如果還有缺失值，用0填充
        
        return df.values
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """創建序列數據"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(self.data) - self.sequence_length + 1):
            sequence = self.data[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length // 2]  # 使用序列中間的標籤
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取數據項"""
        sequence = self.sequences[idx]
        label = self.sequence_labels[idx]
        
        # 轉換為張量
        sequence_tensor = torch.FloatTensor(sequence).T  # (C, L)
        label_tensor = torch.LongTensor([label])
        
        # 重塑為2D圖像格式 (C, H, W)
        # 將一維測井曲線重塑為方形圖像
        h = w = int(np.sqrt(self.sequence_length))
        if h * w < self.sequence_length:
            h = w = int(np.ceil(np.sqrt(self.sequence_length)))
        
        # 填充到方形
        padded_length = h * w
        if sequence_tensor.shape[1] < padded_length:
            padding = torch.zeros(sequence_tensor.shape[0], padded_length - sequence_tensor.shape[1])
            sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
        
        # 重塑為圖像格式 - 每個通道保持一維，但整體為二維圖像
        # 這樣小波變換模組可以將每個通道轉換為時頻圖譜
        sequence_tensor = sequence_tensor.view(sequence_tensor.shape[0], h, w)
        
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor, label_tensor.squeeze()


class SyntheticWellLogDataset(Dataset):
    """
    合成測井曲線數據集
    用於生成模擬的測井數據進行測試
    """
    
    def __init__(self, num_samples: int = 1000, sequence_length: int = 100,
                 num_curves: int = 6, num_classes: int = 4):
        """
        初始化合成數據集
        Args:
            num_samples: 樣本數量
            sequence_length: 序列長度
            num_curves: 測井曲線數量
            num_classes: 流體類型數量
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_curves = num_curves
        self.num_classes = num_classes
        
        # 生成合成數據
        self.data, self.labels = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成數據"""
        data = []
        labels = []
        
        for i in range(self.num_samples):
            # 隨機選擇流體類型
            fluid_type = np.random.randint(0, self.num_classes)
            
            # 根據流體類型生成不同的測井曲線特徵
            if fluid_type == 0:  # 油層
                base_values = [50, -20, 100, 2.3, 0.25, 200]
                noise_level = 0.1
            elif fluid_type == 1:  # 氣層
                base_values = [30, -30, 200, 2.1, 0.15, 180]
                noise_level = 0.15
            elif fluid_type == 2:  # 水層
                base_values = [80, -10, 50, 2.5, 0.35, 220]
                noise_level = 0.08
            else:  # 乾層
                base_values = [20, -40, 300, 2.0, 0.10, 160]
                noise_level = 0.2
            
            # 生成序列數據
            sequence = []
            for j, base_value in enumerate(base_values):
                # 添加趨勢和噪聲
                trend = np.linspace(0, np.random.normal(0, 5), self.sequence_length)
                noise = np.random.normal(0, noise_level * abs(base_value), self.sequence_length)
                curve_data = base_value + trend + noise
                sequence.append(curve_data)
            
            data.append(np.array(sequence).T)
            labels.append(fluid_type)
        
        return np.array(data), np.array(labels)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取數據項"""
        sequence = self.data[idx]
        label = self.labels[idx]
        
        # 轉換為張量
        sequence_tensor = torch.FloatTensor(sequence).T  # (C, L)
        label_tensor = torch.LongTensor([label])
        
        # 重塑為2D圖像格式
        h = w = int(np.sqrt(self.sequence_length))
        if h * w < self.sequence_length:
            h = w = int(np.ceil(np.sqrt(self.sequence_length)))
        
        # 填充到方形
        padded_length = h * w
        if sequence_tensor.shape[1] < padded_length:
            padding = torch.zeros(sequence_tensor.shape[0], padded_length - sequence_tensor.shape[1])
            sequence_tensor = torch.cat([sequence_tensor, padding], dim=1)
        
        # 重塑為圖像格式
        sequence_tensor = sequence_tensor.view(sequence_tensor.shape[0], h, w)
        
        return sequence_tensor, label_tensor.squeeze()


class WaveletWellLogDataset(Dataset):
    """
    小波變換測井曲線數據集
    使用小波變換將測井曲線轉換為高清時頻圖譜
    只處理指定的6條測井曲線：GR、SP、AC、DEN、CNL、RT
    只加載有解釋結論標籤的數據行
    生成高清時頻圖譜並保存到指定文件夾
    """
    
    def __init__(self, data_path: str, sequence_length: int = 100, 
                 curve_names: List[str] = None, delimiter: str = None, 
                 has_header: bool = True, transform=None, 
                 require_labels: bool = True, wavelet_type: str = 'db4',
                 wavelet_level: int = 3, image_size: Tuple[int, int] = (256, 256),
                 save_images: bool = True, image_save_dir: str = './wavelet_images',
                 curve_specific_wavelets: Dict = None):
        """
        初始化小波變換數據集
        Args:
            data_path: txt數據文件路徑
            sequence_length: 序列長度
            curve_names: 測井曲線名稱列表
            delimiter: 分隔符（None為自動檢測）
            has_header: 是否有標題行
            transform: 數據變換
            require_labels: 是否要求必須有標籤（默認True）
            wavelet_type: 小波類型（默認'db4'）
            wavelet_level: 小波分解層數（默認3）
            image_size: 高清時頻圖譜尺寸 (H, W)，默認(256, 256)
            save_images: 是否保存時頻圖譜到文件
            image_save_dir: 時頻圖譜保存目錄
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.delimiter = delimiter
        self.has_header = has_header
        self.transform = transform
        self.require_labels = require_labels
        self.wavelet_type = wavelet_type
        self.wavelet_level = wavelet_level
        self.image_size = image_size
        self.save_images = save_images
        self.image_save_dir = image_save_dir
        self.curve_specific_wavelets = curve_specific_wavelets or {}
        
        # 使用您指定的6條測井曲線名稱
        if curve_names is None:
            curve_names = ['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT']
        self.curve_names = curve_names
        
        # 創建保存目錄
        if self.save_images:
            self._create_save_directories()
        
        # 加載數據
        self.data, self.labels = self._load_txt_data()
        
        # 數據預處理
        self._preprocess_data()
        
        # 生成高清小波時頻圖譜
        self._generate_high_quality_wavelet_images()
    
    def _create_save_directories(self):
        """創建時頻圖譜保存目錄"""
        import os
        
        # 創建主目錄
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        # 為每個測井曲線創建子目錄
        for curve_name in self.curve_names:
            curve_dir = os.path.join(self.image_save_dir, curve_name)
            os.makedirs(curve_dir, exist_ok=True)
        
        print(f"✅ 創建時頻圖譜保存目錄: {self.image_save_dir}")
        for curve_name in self.curve_names:
            print(f"   - {curve_name}: {os.path.join(self.image_save_dir, curve_name)}")
    
    def _detect_delimiter(self, file_path: str) -> str:
        """自動檢測分隔符"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # 常見分隔符
        delimiters = ['\t', ',', ';', '|', ' ']
        
        for delim in delimiters:
            if delim in first_line:
                parts = first_line.split(delim)
                if len(parts) >= 7:  # 至少7列（6列測井曲線數據 + 1列標籤）
                    return delim
        
        # 如果沒有檢測到明確的分隔符，使用空格
        return ' '
    
    def _load_txt_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加載TXT格式數據，只保留有解釋結論標籤的數據行"""
        try:
            # 自動檢測分隔符
            if self.delimiter is None:
                self.delimiter = self._detect_delimiter(self.data_path)
            
            print(f"✅ 使用分隔符: '{self.delimiter}'")
            
            # 嘗試不同的編碼方式讀取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            data_content = None
            
            for encoding in encodings:
                try:
                    with open(self.data_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    data_content = lines
                    print(f"✅ 成功使用編碼: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if data_content is None:
                raise ValueError("無法讀取文件，請檢查文件編碼")
            
            # 解析數據
            if self.has_header:
                header_line = data_content[0].strip()
                headers = header_line.split(self.delimiter)
                data_lines = data_content[1:]
                print(f"✅ 檢測到標題行: {headers}")
                
                # 找到指定的6條測井曲線的列索引
                curve_indices = []
                for curve_name in self.curve_names:
                    try:
                        idx = headers.index(curve_name)
                        curve_indices.append(idx)
                    except ValueError:
                        print(f"⚠️ 未找到測井曲線: {curve_name}")
                
                if len(curve_indices) != 6:
                    raise ValueError(f"只找到 {len(curve_indices)} 條測井曲線，需要6條")
                
                print(f"✅ 測井曲線列索引: {dict(zip(self.curve_names, curve_indices))}")
                
                # 找到解釋結論列的索引
                try:
                    label_index = headers.index('解释结论')
                    print(f"✅ 解釋結論列索引: {label_index}")
                except ValueError:
                    print("⚠️ 未找到'解释结论'列，嘗試其他可能的列名...")
                    possible_labels = ['解释结论', '结论', 'label', 'Label', 'LABEL']
                    label_index = None
                    for label_name in possible_labels:
                        try:
                            label_index = headers.index(label_name)
                            print(f"✅ 找到標籤列: {label_name} (索引: {label_index})")
                            break
                        except ValueError:
                            continue
                    
                    if label_index is None:
                        raise ValueError("未找到解釋結論列")
                
            else:
                data_lines = data_content
                # 生成默認標題
                headers = [f'Curve_{i}' for i in range(7)]  # 7列：6列曲線 + 1列標籤
                curve_indices = list(range(6))  # 前6列
                label_index = 6  # 第7列
                print(f"✅ 使用默認標題: {headers}")
            
            # 解析數值數據，只保留有解釋結論標籤的行
            parsed_data = []
            valid_labels = []
            skipped_count = 0
            
            for line_num, line in enumerate(data_lines, start=2 if self.has_header else 1):
                line = line.strip()
                if line and not line.startswith('#'):  # 跳過空行和註釋行
                    try:
                        values = [x.strip() for x in line.split(self.delimiter) if x.strip()]
                        if len(values) > max(max(curve_indices), label_index):  # 確保有足夠的列
                            # 提取指定的6條測井曲線數據
                            curve_values = []
                            for idx in curve_indices:
                                try:
                                    curve_values.append(float(values[idx]))
                                except ValueError:
                                    print(f"⚠️ 第{line_num}行測井曲線數據無效: {values[idx]}")
                                    break
                            else:
                                # 檢查解釋結論標籤
                                label_value = values[label_index]
                                # 簡單的標籤驗證：非空且不是純數字
                                if label_value and label_value.strip() and not label_value.strip().isdigit():
                                    parsed_data.append(curve_values)
                                    valid_labels.append(label_value)
                                else:
                                    skipped_count += 1
                                    if skipped_count <= 5:  # 只顯示前5個跳過的行
                                        print(f"⚠️ 第{line_num}行解釋結論無效，跳過: {label_value}")
                                    elif skipped_count == 6:
                                        print(f"⚠️ ... 還有更多無效解釋結論行被跳過")
                        else:
                            skipped_count += 1
                            if skipped_count <= 5:
                                print(f"⚠️ 第{line_num}行列數不足，跳過: {len(values)}列")
                    except Exception as e:
                        skipped_count += 1
                        if skipped_count <= 5:
                            print(f"⚠️ 第{line_num}行解析失敗: {e}")
                        continue
            
            if not parsed_data:
                raise ValueError("沒有找到有效的帶解釋結論標籤的數據行")
            
            print(f"✅ 成功加載有效數據: {len(parsed_data)}行")
            if skipped_count > 0:
                print(f"⚠️ 跳過了 {skipped_count} 行無效數據")
            
            # 轉換為numpy數組
            data_array = np.array(parsed_data)
            print(f"✅ 測井曲線數據形狀: {data_array.shape}")
            print(f"✅ 測井曲線: {self.curve_names}")
            
            # 檢查數據有效性
            data_array = self._validate_curve_data(data_array)
            
            # 轉換為標準格式 (N, 6)
            data = data_array
            
            # 處理標籤
            labels = self._process_labels(valid_labels)
            
            return data, labels
            
        except Exception as e:
            print(f"❌ TXT數據加載失敗: {e}")
            raise
    
    def _is_valid_label(self, label_value: str) -> bool:
        """檢查解釋結論標籤是否有效"""
        if not label_value or label_value.strip() == '':
            return False
        
        # 接受所有非空標籤，包括層位信息
        label_clean = label_value.strip()
        
        # 如果標籤包含層位信息（如"长4+5!2^(1_2)"），也認為是有效的
        if '长' in label_clean or '层' in label_clean or '!' in label_clean:
            return True
        
        # 常見的流體類型標籤
        valid_labels = {
            # 中文標籤
            '油層', '水層', '乾層', '干层', '差油層', '差油层', '油水同層', '油水同层',
            '油', '水', '乾', '干', '氣', '气',
            # 英文標籤
            'oil', 'water', 'dry', 'poor-oil', 'oil-water', 'gas',
            # 數字標籤
            '0', '1', '2', '3', '4', '5'
        }
        
        return label_clean in valid_labels
    
    def _validate_curve_data(self, curve_data: np.ndarray) -> np.ndarray:
        """驗證測井曲線數據的有效性"""
        # 檢查是否有無窮大或NaN值
        curve_data = np.nan_to_num(curve_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 檢查數值範圍 - 現在數據格式是 (N, 6)
        for i in range(curve_data.shape[1]):  # 遍歷6個測井曲線
            curve = curve_data[:, i]
            # 移除異常值（超過3個標準差）
            mean_val = np.mean(curve)
            std_val = np.std(curve)
            if std_val > 0:
                threshold = 3 * std_val
                curve_data[:, i] = np.clip(curve, mean_val - threshold, mean_val + threshold)
        
        return curve_data
    
    def _process_labels(self, raw_labels: List[str]) -> np.ndarray:
        """處理解釋結論標籤數據"""
        try:
            # 首先嘗試轉換為數值標籤
            numeric_labels = []
            for label in raw_labels:
                try:
                    numeric_labels.append(float(label))
                except ValueError:
                    # 如果不是數值，保持原樣
                    numeric_labels.append(label)
            
            # 檢查數值標籤的比例
            numeric_count = sum(1 for x in numeric_labels if isinstance(x, (int, float)))
            if numeric_count > len(numeric_labels) * 0.8:  # 80%以上是數值
                # 使用數值標籤
                labels = np.array(numeric_labels)
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 10:  # 如果標籤數量合理
                    # 創建標籤映射
                    label_mapping = {val: idx for idx, val in enumerate(sorted(unique_labels))}
                    labels = np.array([label_mapping[label] for label in labels])
                    print(f"✅ 數值標籤映射: {label_mapping}")
                else:
                    # 如果標籤數量過多，使用分箱
                    labels = np.digitize(labels, bins=np.percentile(labels, [25, 50, 75]))
                    print(f"✅ 使用分箱標籤，類別數: {len(np.unique(labels))}")
            else:
                # 使用字符串標籤
                label_mapping = {
                    '油層': 0, '氣層': 1, '水層': 2, '乾層': 3,
                    '油氣層': 0, '氣水層': 1, '油水層': 0,
                    'oil': 0, 'gas': 1, 'water': 2, 'dry': 3,
                    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5
                }
                labels = np.array([label_mapping.get(str(label), 0) for label in raw_labels])
                print(f"✅ 字符串標籤映射完成")
                print(f"✅ 標籤分布: {np.bincount(labels)}")
        
        except Exception as e:
            print(f"⚠️ 標籤處理失敗: {e}，使用默認標籤")
            labels = np.zeros(len(raw_labels), dtype=int)
        
        return labels
    
    def _preprocess_data(self):
        """數據預處理"""
        # 處理缺失值
        self.data = self._handle_missing_values(self.data)
        
        # 標準化
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        
        # 編碼標籤
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        
        # 創建序列數據
        self.sequences, self.sequence_labels = self._create_sequences()
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """處理缺失值"""
        # 使用前向填充
        df = pd.DataFrame(data)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')  # 處理開頭的缺失值
        df = df.fillna(0)  # 如果還有缺失值，用0填充
        
        return df.values
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """創建序列數據"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(self.data) - self.sequence_length + 1):
            sequence = self.data[i:i + self.sequence_length]
            label = self.labels[i + self.sequence_length // 2]  # 使用序列中間的標籤
            
            sequences.append(sequence)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def _generate_high_quality_wavelet_images(self):
        """使用小波變換生成高清時頻圖譜"""
        print(f"🔄 開始生成高清小波時頻圖譜...")
        print(f"   小波類型: {self.wavelet_type}")
        print(f"   分解層數: {self.wavelet_level}")
        print(f"   圖像尺寸: {self.image_size}")
        print(f"   保存目錄: {self.image_save_dir}")
        
        try:
            import pywt
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式後端
            
            self.wavelet_images = []
            
            for i, sequence in enumerate(self.sequences):
                if i % 50 == 0:
                    print(f"   進度: {i}/{len(self.sequences)}")
                
                # 為每個測井曲線生成高清小波時頻圖譜
                curve_images = []
                for curve_idx in range(sequence.shape[1]):  # 6個測井曲線
                    curve_signal = sequence[:, curve_idx]
                    curve_name = self.curve_names[curve_idx]
                    
                    # 根據配置選擇特定的小波基函數和分解層數
                    if curve_name in self.curve_specific_wavelets:
                        specific_config = self.curve_specific_wavelets[curve_name]
                        wavelet_type = specific_config['wavelet']
                        wavelet_level = specific_config['level']
                    else:
                        # 使用默認配置
                        wavelet_type = self.wavelet_type
                        wavelet_level = self.wavelet_level
                    
                    # 進行高質量小波變換
                    try:
                        # 使用離散小波變換生成高質量時頻圖譜
                        # 進行小波包變換
                        wp = pywt.WaveletPacket(data=curve_signal, wavelet=wavelet_type, mode='symmetric')
                        
                        # 提取指定層級的係數
                        coeffs = []
                        for node in wp.get_level(wavelet_level, 'natural'):
                            coeff = node.data
                            coeffs.append(coeff)
                        
                        if coeffs:
                            # 將係數轉換為時頻圖譜
                            coeff_array = np.array(coeffs)
                            
                            # 重塑為指定尺寸的高清圖像
                            h, w = self.image_size
                            
                            # 使用雙線性插值調整尺寸
                            from scipy.ndimage import zoom
                            if coeff_array.shape[0] != h or coeff_array.shape[1] != w:
                                zoom_factor_h = h / coeff_array.shape[0]
                                zoom_factor_w = w / coeff_array.shape[1]
                                coeff_array = zoom(coeff_array, (zoom_factor_h, zoom_factor_w), order=1)
                            
                            # 確保尺寸正確
                            if coeff_array.shape[0] > h:
                                coeff_array = coeff_array[:h, :]
                            if coeff_array.shape[1] > w:
                                coeff_array = coeff_array[:, :w]
                            if coeff_array.shape[0] < h or coeff_array.shape[1] < w:
                                # 如果尺寸不足，用零填充
                                temp = np.zeros((h, w))
                                temp[:coeff_array.shape[0], :coeff_array.shape[1]] = coeff_array
                                coeff_array = temp
                            
                            # 對數變換增強對比度
                            coeff_array = np.log10(np.abs(coeff_array) + 1e-10)
                            
                            # 標準化到[0, 1]範圍
                            if coeff_array.max() != coeff_array.min():
                                coeff_array = (coeff_array - coeff_array.min()) / (coeff_array.max() - coeff_array.min())
                            
                            curve_images.append(coeff_array)
                        else:
                            # 如果沒有係數，使用零圖像
                            curve_images.append(np.zeros(self.image_size))
                        
                        # 計算功率譜密度
                        power = np.abs(coeff_array) ** 2
                        
                        # 重塑為指定尺寸的高清圖像
                        h, w = self.image_size
                        
                        # 使用雙線性插值調整尺寸
                        from scipy.ndimage import zoom
                        if power.shape[0] != h or power.shape[1] != w:
                            zoom_factor_h = h / power.shape[0]
                            zoom_factor_w = w / power.shape[1]
                            power = zoom(power, (zoom_factor_h, zoom_factor_w), order=1)
                        
                        # 確保尺寸正確
                        if power.shape[0] > h:
                            power = power[:h, :]
                        if power.shape[1] > w:
                            power = power[:, :w]
                        if power.shape[0] < h or power.shape[1] < w:
                            # 如果尺寸不足，用零填充
                            temp = np.zeros((h, w))
                            temp[:power.shape[0], :power.shape[1]] = power
                            power = temp
                        
                        # 對數變換增強對比度
                        power = np.log10(power + 1e-10)
                        
                        # 標準化到[0, 1]範圍
                        if power.max() != power.min():
                            power = (power - power.min()) / (power.max() - power.min())
                        
                        curve_images.append(power)
                        
                        # 保存高清圖像
                        if self.save_images:
                            self._save_single_wavelet_image(power, i, curve_idx, curve_name)
                            
                    except Exception as e:
                        print(f"⚠️ 小波變換失敗 (序列{i}, 曲線{curve_idx}): {e}")
                        # 創建零圖像作為fallback
                        fallback_image = np.zeros(self.image_size)
                        curve_images.append(fallback_image)
                
                # 將6個曲線的時頻圖譜堆疊為一個多通道圖像
                if len(curve_images) == 6:
                    multi_channel_image = np.stack(curve_images, axis=0)  # (6, H, W)
                    self.wavelet_images.append(multi_channel_image)
                else:
                    print(f"⚠️ 序列{i}的曲線數量不正確: {len(curve_images)}")
                    # 創建零圖像作為fallback
                    fallback_image = np.zeros((6, self.image_size[0], self.image_size[1]))
                    self.wavelet_images.append(fallback_image)
            
            self.wavelet_images = np.array(self.wavelet_images)
            print(f"✅ 高清小波時頻圖譜生成完成: {self.wavelet_images.shape}")
            
        except ImportError as e:
            print(f"❌ 缺少必要庫: {e}")
            print("請安裝: pip install PyWavelets matplotlib scipy")
            # 創建零圖像作為fallback
            self.wavelet_images = np.zeros((len(self.sequences), 6, self.image_size[0], self.image_size[1]))
        except Exception as e:
            print(f"❌ 高清小波時頻圖譜生成失敗: {e}")
            # 創建零圖像作為fallback
            self.wavelet_images = np.zeros((len(self.sequences), 6, self.image_size[0], self.image_size[1]))
    
    def _save_single_wavelet_image(self, image_data, seq_idx, curve_idx, curve_name):
        """保存單個高清小波時頻圖譜"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # 創建高質量圖像
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
            
            # 使用高質量顏色映射
            im = ax.imshow(image_data, cmap='viridis', aspect='auto', interpolation='bilinear')
            
            # 添加顏色條
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power Spectrum Density (dB)', fontsize=12)
            
            # 設置標題和標籤
            ax.set_title(f'Wavelet Time-Frequency Spectrum\n{curve_name} - Sequence {seq_idx}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Frequency Scale', fontsize=12)
            
            # 設置網格
            ax.grid(True, alpha=0.3)
            
            # 保存高質量圖像
            image_path = os.path.join(self.image_save_dir, curve_name, 
                                     f"seq_{seq_idx:04d}_curve_{curve_name}.png")
            plt.savefig(image_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ 保存圖像失敗 (序列{seq_idx}, 曲線{curve_name}): {e}")
    
    def _save_wavelet_images(self):
        """保存高清小波時頻圖譜到文件"""
        print(f"💾 開始保存高清小波時頻圖譜到: {self.image_save_dir}")
        for i, sequence_images in enumerate(self.wavelet_images):
            for j, image in enumerate(sequence_images):
                curve_name = self.curve_names[j]
                image_path = os.path.join(self.image_save_dir, curve_name, f"seq_{i}_curve_{j}.png")
                try:
                    # 確保圖像數據是浮點類型，並且範圍在[0, 1]
                    image_data = (image * 255).astype(np.uint8)
                    # 使用PIL保存圖像
                    from PIL import Image
                    Image.fromarray(image_data).save(image_path)
                    print(f"   保存圖片: {image_path}")
                except Exception as e:
                    print(f"   保存圖片失敗 ({image_path}): {e}")
        print(f"✅ 高清小波時頻圖譜保存完成")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """獲取數據項"""
        # 返回小波時頻圖譜和標籤
        wavelet_image = torch.FloatTensor(self.wavelet_images[idx])  # (6, H, W)
        label = torch.LongTensor([self.sequence_labels[idx]])
        
        if self.transform:
            wavelet_image = self.transform(wavelet_image)
        
        return wavelet_image, label.squeeze()


def create_data_loaders(config, data_path: str = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建數據加載器
    Args:
        config: 配置對象
        data_path: 數據文件路徑
    Returns:
        訓練、驗證、測試數據加載器
    """
    if data_path and os.path.exists(data_path):
        # 使用真實數據
        print(f"加載真實數據: {data_path}")
        
        # 根據文件擴展名選擇合適的數據加載器
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.txt':
            print("📄 檢測到TXT格式，使用TXTWellLogDataset")
            train_dataset = TXTWellLogDataset(
                data_path=data_path,
                sequence_length=config.DATA_CONFIG['sequence_length'],
                curve_names=['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON'],
                require_labels=True  # 只加載有標籤的數據
            )
        elif file_ext in ['.csv', '.xlsx']:
            print("📊 檢測到CSV/Excel格式，使用WellLogDataset")
            train_dataset = WellLogDataset(
                data_path=data_path,
                sequence_length=config.DATA_CONFIG['sequence_length'],
                curve_names=['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
            )
        else:
            print(f"⚠️ 不支持的文件格式: {file_ext}，使用TXTWellLogDataset嘗試讀取")
            train_dataset = TXTWellLogDataset(
                data_path=data_path,
                sequence_length=config.DATA_CONFIG['sequence_length'],
                curve_names=['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
            )
        
        # 分割數據集
        train_size = int(config.DATA_CONFIG['train_ratio'] * len(train_dataset))
        val_size = int(config.DATA_CONFIG['val_ratio'] * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size]
        )
    else:
        # 使用合成數據
        train_dataset = SyntheticWellLogDataset(
            num_samples=1000,
            sequence_length=config.DATA_CONFIG['sequence_length'],
            num_curves=config.DATA_CONFIG['num_curves'],
            num_classes=config.DATA_CONFIG['num_classes']
        )
        
        val_dataset = SyntheticWellLogDataset(
            num_samples=200,
            sequence_length=config.DATA_CONFIG['sequence_length'],
            num_curves=config.DATA_CONFIG['num_curves'],
            num_classes=config.DATA_CONFIG['num_classes']
        )
        
        test_dataset = SyntheticWellLogDataset(
            num_samples=200,
            sequence_length=config.DATA_CONFIG['sequence_length'],
            num_curves=config.DATA_CONFIG['num_curves'],
            num_classes=config.DATA_CONFIG['num_classes']
        )
    
    # 創建數據加載器 - 優化GPU訓練
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.DATA_CONFIG['num_workers'],
        pin_memory=True,  # 加速GPU傳輸
        persistent_workers=True  # 保持worker進程
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.DATA_CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DATA_CONFIG['batch_size'],
        shuffle=False,
        num_workers=config.DATA_CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader 


def create_wavelet_data_loaders(config, data_path: str, 
                               wavelet_type: str = 'db4', wavelet_level: int = 3,
                               image_size: Tuple[int, int] = (256, 256),
                               save_images: bool = True, image_save_dir: str = './wavelet_images') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建高清小波變換數據加載器
    Args:
        config: 配置對象
        data_path: 數據文件路徑
        wavelet_type: 小波類型
        wavelet_level: 小波分解層數
        image_size: 高清時頻圖譜尺寸，默認(256, 256)
        save_images: 是否保存時頻圖譜到文件
        image_save_dir: 時頻圖譜保存目錄
    Returns:
        訓練、驗證、測試數據加載器
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"數據文件不存在: {data_path}")
    
    # 根據文件擴展名選擇合適的數據加載器
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.txt':
        print("📄 檢測到TXT格式，使用高清WaveletWellLogDataset")
        print(f"🔧 小波參數: 類型={wavelet_type}, 層數={wavelet_level}, 尺寸={image_size}")
        print(f"💾 保存設置: {save_images}, 目錄={image_save_dir}")
        
        # 創建高清小波變換數據集
        dataset = WaveletWellLogDataset(
            data_path=data_path,
            sequence_length=config.DATA_CONFIG['sequence_length'],
            curve_names=['GR', 'SP', 'AC', 'DEN', 'CNL', 'RT'],  # 使用您指定的6條測井曲線
            require_labels=True,  # 只加載有解釋結論標籤的數據
            wavelet_type=wavelet_type,
            wavelet_level=wavelet_level,
            image_size=image_size,
            save_images=save_images,
            image_save_dir=image_save_dir
        )
        
        print(f"✅ 小波變換數據集創建成功")
        print(f"📊 數據集大小: {len(dataset)}")
        print(f"📊 測井曲線: {dataset.curve_names}")
        
        # 數據分割
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, temp_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size + test_size]
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size]
        )
        
        print(f"📊 數據分割:")
        print(f"   - 訓練集: {len(train_dataset)} 樣本")
        print(f"   - 驗證集: {len(val_dataset)} 樣本")
        print(f"   - 測試集: {len(test_dataset)} 樣本")
        
        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.DATA_CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows環境下設為0
            pin_memory=True,
            persistent_workers=False  # Windows環境下設為False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.DATA_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.DATA_CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False
        )
        
        return train_loader, val_loader, test_loader
        
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}，請使用TXT格式文件") 