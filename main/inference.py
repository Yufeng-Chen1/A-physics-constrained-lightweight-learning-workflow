import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import List, Dict, Tuple
import argparse

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config
from models.fluid_identification_model import FluidIdentificationModel, MultiScaleFluidIdentificationModel, EnsembleFluidIdentificationModel
from data.data_loader import WellLogDataset, SyntheticWellLogDataset


class FluidIdentifier:
    """流體識別器"""
    
    def __init__(self, model_path: str, config: Config, model_type: str = 'standard'):
        self.config = config
        
        # 強制GPU檢查
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU不可用！此項目需要GPU進行推理。請檢查CUDA安裝和GPU驅動。")
        
        self.device = torch.device('cuda')
        
        # 創建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 加載模型權重
        self.load_model(model_path)
        
        # 流體類型映射
        self.fluid_types = ['Oil', 'Gas', 'Water', 'Dry']
        self.curve_types = ['GR', 'SP', 'RES', 'DEN', 'NEU', 'SON']
        
        print(f"✅ 模型已移至GPU: {next(self.model.parameters()).device}")
    
    def _create_model(self):
        """創建模型"""
        if self.model_type == 'multiscale':
            return MultiScaleFluidIdentificationModel(self.config)
        elif self.model_type == 'ensemble':
            return EnsembleFluidIdentificationModel(self.config)
        else:
            return FluidIdentificationModel(self.config)
    
    def load_model(self, model_path: str):
        """加載模型"""
        # 強制加載到GPU
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ 模型已從 {model_path} 加載到GPU")
    
    def preprocess_data(self, data: np.ndarray) -> torch.Tensor:
        """預處理數據"""
        # 確保數據形狀正確 (N, C) -> (1, C, H, W)
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        
        # 轉換為張量
        data_tensor = torch.FloatTensor(data)
        
        # 重塑為圖像格式
        batch_size, channels, sequence_length = data_tensor.shape
        h = w = int(np.sqrt(sequence_length))
        if h * w < sequence_length:
            h = w = int(np.ceil(np.sqrt(sequence_length)))
        
        # 填充到方形
        padded_length = h * w
        if data_tensor.shape[2] < padded_length:
            padding = torch.zeros(batch_size, channels, padded_length - data_tensor.shape[2])
            data_tensor = torch.cat([data_tensor, padding], dim=2)
        
        # 重塑為圖像格式 (B, C, H, W)
        data_tensor = data_tensor.view(batch_size, channels, h, w)
        
        return data_tensor
    
    def predict(self, data: np.ndarray, curve_types: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        進行預測
        Args:
            data: 輸入數據 (N, C) 或 (B, N, C)
            curve_types: 測井曲線類型列表
        Returns:
            預測結果和概率
        """
        # 預處理數據
        data_tensor = self.preprocess_data(data)
        data_tensor = data_tensor.to(self.device, non_blocking=True)
        
        # 設置曲線類型
        if curve_types is None:
            curve_types = self.curve_types[:data_tensor.shape[1]]
        
        # 進行預測
        with torch.no_grad():
            outputs = self.model(data_tensor, curve_types)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def predict_single_sequence(self, sequence: np.ndarray, curve_types: List[str] = None) -> Dict:
        """
        預測單個序列
        Args:
            sequence: 單個序列數據 (C, L)
            curve_types: 測井曲線類型列表
        Returns:
            預測結果字典
        """
        # 重塑為批次格式
        sequence_batch = sequence.T.reshape(1, -1, sequence.shape[0])
        
        # 進行預測
        predictions, probabilities = self.predict(sequence_batch, curve_types)
        
        # 構建結果字典
        result = {
            'predicted_fluid': self.fluid_types[predictions[0]],
            'confidence': float(np.max(probabilities[0])),
            'probabilities': {
                fluid_type: float(prob) 
                for fluid_type, prob in zip(self.fluid_types, probabilities[0])
            }
        }
        
        return result
    
    def predict_batch(self, data: np.ndarray, curve_types: List[str] = None) -> List[Dict]:
        """
        批量預測
        Args:
            data: 批量數據 (B, N, C)
            curve_types: 測井曲線類型列表
        Returns:
            預測結果列表
        """
        predictions, probabilities = self.predict(data, curve_types)
        
        results = []
        for i in range(len(predictions)):
            result = {
                'sample_id': i,
                'predicted_fluid': self.fluid_types[predictions[i]],
                'confidence': float(np.max(probabilities[i])),
                'probabilities': {
                    fluid_type: float(prob) 
                    for fluid_type, prob in zip(self.fluid_types, probabilities[i])
                }
            }
            results.append(result)
        
        return results
    
    def analyze_confidence(self, predictions: List[Dict]) -> Dict:
        """分析預測置信度"""
        confidences = [pred['confidence'] for pred in predictions]
        fluid_types = [pred['predicted_fluid'] for pred in predictions]
        
        analysis = {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'high_confidence_samples': sum(1 for c in confidences if c > 0.8),
            'low_confidence_samples': sum(1 for c in confidences if c < 0.5),
            'fluid_distribution': {fluid: fluid_types.count(fluid) for fluid in self.fluid_types}
        }
        
        return analysis
    
    def plot_probabilities(self, probabilities: np.ndarray, sample_indices: List[int] = None):
        """繪製概率分布"""
        if sample_indices is None:
            sample_indices = range(min(10, len(probabilities)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            if i >= 4:
                break
            
            ax = axes[i]
            probs = probabilities[idx]
            
            bars = ax.bar(self.fluid_types, probs, color=['red', 'orange', 'blue', 'gray'])
            ax.set_title(f'Sample {idx} - Probability Distribution')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            
            # 添加數值標籤
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('probability_distribution.png')
        plt.close()
    
    def plot_confidence_histogram(self, predictions: List[Dict]):
        """繪製置信度直方圖"""
        confidences = [pred['confidence'] for pred in predictions]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig('confidence_histogram.png')
        plt.close()
    
    def generate_report(self, predictions: List[Dict], output_path: str = 'prediction_report.txt'):
        """生成預測報告"""
        analysis = self.analyze_confidence(predictions)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("井下流體識別預測報告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 預測統計\n")
            f.write(f"總樣本數: {len(predictions)}\n")
            f.write(f"平均置信度: {analysis['mean_confidence']:.3f}\n")
            f.write(f"置信度標準差: {analysis['std_confidence']:.3f}\n")
            f.write(f"最高置信度: {analysis['max_confidence']:.3f}\n")
            f.write(f"最低置信度: {analysis['min_confidence']:.3f}\n\n")
            
            f.write("2. 置信度分析\n")
            f.write(f"高置信度樣本 (>0.8): {analysis['high_confidence_samples']}\n")
            f.write(f"低置信度樣本 (<0.5): {analysis['low_confidence_samples']}\n\n")
            
            f.write("3. 流體類型分布\n")
            for fluid_type, count in analysis['fluid_distribution'].items():
                percentage = count / len(predictions) * 100
                f.write(f"{fluid_type}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n4. 詳細預測結果\n")
            f.write("-" * 80 + "\n")
            for i, pred in enumerate(predictions[:20]):  # 只顯示前20個
                f.write(f"樣本 {pred['sample_id']}: {pred['predicted_fluid']} "
                       f"(置信度: {pred['confidence']:.3f})\n")
                for fluid, prob in pred['probabilities'].items():
                    f.write(f"  {fluid}: {prob:.3f}\n")
                f.write("\n")
        
        print(f"預測報告已保存到 {output_path}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='井下流體識別推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路徑')
    parser.add_argument('--data_path', type=str, help='數據文件路徑')
    parser.add_argument('--model_type', type=str, default='standard', 
                       choices=['standard', 'multiscale', 'ensemble'], help='模型類型')
    parser.add_argument('--output_dir', type=str, default='./results', help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建配置
    config = Config()
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 創建識別器
    identifier = FluidIdentifier(args.model_path, config, args.model_type)
    
    if args.data_path:
        # 使用真實數據
        print(f"加載數據: {args.data_path}")
        
        # 創建數據集
        dataset = WellLogDataset(
            data_path=args.data_path,
            sequence_length=config.DATA_CONFIG['sequence_length']
        )
        
        # 進行預測
        all_predictions = []
        for i in range(len(dataset)):
            data, _ = dataset[i]
            data = data.numpy()
            
            # 重塑為正確格式
            data = data.reshape(1, data.shape[0], -1)
            
            # 進行預測
            result = identifier.predict_single_sequence(data[0].T)
            result['sample_id'] = i
            all_predictions.append(result)
            
            if i % 100 == 0:
                print(f"已處理 {i}/{len(dataset)} 個樣本")
        
        # 生成報告
        identifier.generate_report(all_predictions, 
                                os.path.join(args.output_dir, 'prediction_report.txt'))
        
        # 繪製圖表
        probabilities = np.array([[pred['probabilities'][fluid] for fluid in identifier.fluid_types] 
                                for pred in all_predictions])
        identifier.plot_probabilities(probabilities)
        identifier.plot_confidence_histogram(all_predictions)
        
    else:
        # 使用合成數據進行演示
        print("使用合成數據進行演示...")
        
        # 創建合成數據集
        dataset = SyntheticWellLogDataset(
            num_samples=100,
            sequence_length=config.DATA_CONFIG['sequence_length'],
            num_curves=config.DATA_CONFIG['num_curves'],
            num_classes=config.DATA_CONFIG['num_classes']
        )
        
        # 進行預測
        all_predictions = []
        for i in range(len(dataset)):
            data, _ = dataset[i]
            data = data.numpy()
            
            # 重塑為正確格式
            data = data.reshape(1, data.shape[0], -1)
            
            # 進行預測
            result = identifier.predict_single_sequence(data[0].T)
            result['sample_id'] = i
            all_predictions.append(result)
        
        # 生成報告
        identifier.generate_report(all_predictions, 
                                os.path.join(args.output_dir, 'demo_report.txt'))
        
        # 繪製圖表
        probabilities = np.array([[pred['probabilities'][fluid] for fluid in identifier.fluid_types] 
                                for pred in all_predictions])
        identifier.plot_probabilities(probabilities)
        identifier.plot_confidence_histogram(all_predictions)
    
    print(f"推理完成！結果已保存到 {args.output_dir}")


if __name__ == '__main__':
    main() 