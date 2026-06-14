import torch
import torch.nn as nn
import pywt
import numpy as np
from typing import List, Tuple, Dict


class AdaptiveWaveletModule(nn.Module):
    """
    自適應小波變換模組
    根據不同測井曲線的特徵選擇最優的小波基和分解層數
    """
    
    def __init__(self, wavelet_families: List[str] = None, max_level: int = 5, 
                 entropy_threshold: float = 0.85, curve_specific_wavelets: Dict = None):
        super().__init__()
        
        if wavelet_families is None:
            wavelet_families = ['db1', 'db2', 'db3', 'db4', 'haar', 'coif1', 'coif2', 'sym2', 'sym3']
        
        self.wavelet_families = wavelet_families
        self.max_level = max_level
        self.entropy_threshold = entropy_threshold
        self.curve_specific_wavelets = curve_specific_wavelets or {}
        
        # 為每個測井曲線類型預定義最優小波基
        self.curve_wavelet_mapping = {
            'GR': 'db4',      # 自然伽馬曲線 - 適合db4
            'SP': 'sym3',     # 自然電位曲線 - 適合sym3
            'RES': 'coif2',   # 電阻率曲線 - 適合coif2
            'DEN': 'db3',     # 密度曲線 - 適合db3
            'NEU': 'haar',    # 中子曲線 - 適合haar
            'SON': 'db2'      # 聲波曲線 - 適合db2
        }
        
        # 緩存機制，避免重複計算
        self._wavelet_cache = {}
    
    def _calculate_entropy(self, coeffs: torch.Tensor) -> torch.Tensor:
        """計算小波係數的熵值"""
        energy = torch.sum(coeffs ** 2, dim=-1)
        prob = energy / (torch.sum(energy, dim=-1, keepdim=True) + 1e-8)
        return -torch.sum(prob * torch.log2(prob + 1e-8), dim=-1)
    
    def _select_optimal_wavelet(self, signal: np.ndarray, curve_type: str = None) -> Tuple[str, int]:
        """
        選擇最優小波基和分解層數
        優先使用配置中的特定參數，否則通過熵值選擇
        """
        # 創建緩存鍵
        signal_hash = hash(signal.tobytes())
        cache_key = f"{signal_hash}_{curve_type}"
        
        # 檢查緩存
        if cache_key in self._wavelet_cache:
            return self._wavelet_cache[cache_key]
        
        # 優先使用配置中的特定小波參數
        if curve_type and curve_type in self.curve_specific_wavelets:
            specific_config = self.curve_specific_wavelets[curve_type]
            wavelet = specific_config['wavelet']
            level = specific_config['level']
            result = (wavelet, level)
            self._wavelet_cache[cache_key] = result
            return result
        
        # 如果有預定義的曲線類型映射，直接使用
        if curve_type and curve_type in self.curve_wavelet_mapping:
            wavelet = self.curve_wavelet_mapping[curve_type]
        else:
            # 否則通過熵值選擇最優小波基
            min_entropy = float('inf')
            optimal_wavelet = self.wavelet_families[0]
            
            for wavelet in self.wavelet_families:
                try:
                    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
                    entropy_sum = 0
                    count = 0
                    
                    for level in range(1, min(3, self.max_level + 1)):
                        nodes = [node.path for node in wp.get_level(level, 'natural')]
                        for node in nodes:
                            coeff = torch.tensor(wp[node].data)
                            entropy_sum += self._calculate_entropy(coeff).mean().item()
                            count += 1
                    
                    avg_entropy = entropy_sum / count if count > 0 else float('inf')
                    if avg_entropy < min_entropy:
                        min_entropy = avg_entropy
                        optimal_wavelet = wavelet
                        
                except Exception:
                    continue
        
        # 確定最優分解層數
        optimal_level = 1
        try:
            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
            
            for level in range(1, self.max_level + 1):
                nodes = [node.path for node in wp.get_level(level, 'natural')]
                if not nodes:
                    break
                
                entropy_sum = 0
                for node in nodes:
                    coeff = torch.tensor(wp[node].data)
                    entropy_sum += self._calculate_entropy(coeff).mean().item()
                
                avg_entropy = entropy_sum / len(nodes)
                if avg_entropy < self.entropy_threshold:
                    optimal_level = level
                else:
                    break
                    
        except Exception:
            optimal_level = 1
        
        result = (wavelet, optimal_level)
        self._wavelet_cache[cache_key] = result
        return result
    
    def _extract_wavelet_features(self, signal: np.ndarray, wavelet: str, level: int) -> torch.Tensor:
        """提取小波特徵並轉換為二維時頻圖譜"""
        try:
            # 確保信號是一維的
            if len(signal.shape) > 1:
                signal = signal.flatten()
            
            # 進行小波包變換
            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
            
            # 提取指定層級的所有係數
            coeffs = []
            for node in wp.get_level(level, 'natural'):
                coeff = torch.tensor(node.data, dtype=torch.float32)
                coeffs.append(coeff)
            
            if coeffs:
                # 確保所有係數都有相同的長度
                max_length = max(coeff.shape[0] for coeff in coeffs)
                padded_coeffs = []
                
                for coeff in coeffs:
                    if coeff.shape[0] < max_length:
                        # 用零填充到相同長度
                        padding = torch.zeros(max_length - coeff.shape[0])
                        padded_coeff = torch.cat([coeff, padding])
                    else:
                        padded_coeff = coeff
                    padded_coeffs.append(padded_coeff)
                
                # 將係數重塑為二維時頻圖譜
                coeff_tensor = torch.stack(padded_coeffs, dim=0)  # (N_coeff, L)
                
                # 計算合適的圖像尺寸
                n_coeffs = coeff_tensor.shape[0]
                seq_len = coeff_tensor.shape[1]
                
                # 重塑為方形圖像
                h = w = int(np.sqrt(n_coeffs * seq_len))
                if h * w < n_coeffs * seq_len:
                    h = w = int(np.ceil(np.sqrt(n_coeffs * seq_len)))
                
                # 填充到方形
                padded_size = h * w
                if coeff_tensor.numel() < padded_size:
                    padding = torch.zeros(padded_size - coeff_tensor.numel())
                    coeff_tensor = torch.cat([coeff_tensor.flatten(), padding])
                else:
                    coeff_tensor = coeff_tensor.flatten()[:padded_size]
                
                # 重塑為二維時頻圖譜
                time_freq_image = coeff_tensor.view(h, w)
                
                return time_freq_image.unsqueeze(0)  # (1, H, W)
            else:
                # 如果沒有係數，返回零張量
                return torch.zeros((1, 64, 64))  # 默認64x64圖像
                
        except Exception as e:
            print(f"小波變換錯誤: {e}")
            # 返回零張量作為fallback
            return torch.zeros((1, 64, 64))  # 默認64x64圖像
    
    def forward(self, x: torch.Tensor, curve_types: List[str] = None) -> torch.Tensor:
        """
        前向傳播 - 將一維測井曲線轉換為二維時頻圖譜
        Args:
            x: 輸入張量 (B, C, H, W) - B批次大小, C通道數(測井曲線數), H高度, W寬度
            curve_types: 測井曲線類型列表
        Returns:
            二維時頻圖譜張量 (B, C, H, W)
        """
        batch_size, channels, height, width = x.shape
        results = []
        
        # 設置默認曲線類型
        if curve_types is None:
            curve_types = ['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
        
        for b in range(batch_size):
            batch_results = []
            
            for c in range(channels):
                # 提取一維測井曲線
                signal = x[b, c].cpu().numpy()
                
                # 如果輸入是二維的，轉換為一維
                if len(signal.shape) > 1:
                    signal = signal.flatten()
                
                curve_type = curve_types[c] if c < len(curve_types) else None
                
                # 選擇最優小波基和分解層數
                optimal_wavelet, optimal_level = self._select_optimal_wavelet(signal, curve_type)
                
                # 提取小波特徵並轉換為二維時頻圖譜
                time_freq_image = self._extract_wavelet_features(signal, optimal_wavelet, optimal_level)
                batch_results.append(time_freq_image)
            
            # 將該批次的所有通道特徵堆疊
            batch_tensor = torch.cat(batch_results, dim=0)  # (C, H, W)
            results.append(batch_tensor)
        
        # 將所有批次的結果堆疊
        return torch.stack(results).to(x.device)  # (B, C, H, W)


class MultiScaleWaveletModule(nn.Module):
    """
    多尺度小波變換模組
    同時使用多個小波基進行特徵提取
    """
    
    def __init__(self, wavelet_families: List[str] = None, max_level: int = 3):
        super().__init__()
        
        if wavelet_families is None:
            wavelet_families = ['db4', 'haar', 'coif2', 'sym3']
        
        self.wavelet_families = wavelet_families
        self.max_level = max_level
        
        # 為每個小波基創建特徵融合層
        self.feature_fusion = nn.ModuleList([
            nn.Conv2d(len(wavelet_families), 32, kernel_size=3, padding=1)
            for _ in range(max_level)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度小波變換前向傳播
        Args:
            x: 輸入張量 (B, C, H, W)
        Returns:
            融合後的小波特徵
        """
        batch_size, channels, height, width = x.shape
        multi_scale_features = []
        
        for level in range(1, self.max_level + 1):
            level_features = []
            
            for wavelet in self.wavelet_families:
                wavelet_features = []
                
                for b in range(batch_size):
                    for c in range(channels):
                        signal = x[b, c].cpu().numpy()
                        
                        try:
                            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric')
                            coeffs = []
                            
                            for node in wp.get_level(level, 'natural'):
                                coeff = torch.tensor(node.data, dtype=torch.float32)
                                coeffs.append(coeff)
                            
                            if coeffs:
                                wavelet_features.append(torch.stack(coeffs, dim=0))
                            else:
                                # 零張量作為fallback
                                wavelet_features.append(torch.zeros((1, height // (2 ** level), width // (2 ** level))))
                                
                        except Exception:
                            wavelet_features.append(torch.zeros((1, height // (2 ** level), width // (2 ** level))))
                
                # 將該小波基的特徵堆疊
                if wavelet_features:
                    wavelet_tensor = torch.stack(wavelet_features, dim=0)
                    level_features.append(wavelet_tensor)
            
            # 融合該層級的所有小波基特徵
            if level_features:
                level_tensor = torch.cat(level_features, dim=1)  # 在通道維度上拼接
                fused_features = self.feature_fusion[level - 1](level_tensor)
                multi_scale_features.append(fused_features)
        
        # 返回所有層級的融合特徵
        return torch.cat(multi_scale_features, dim=1) 