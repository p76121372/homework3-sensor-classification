import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from preprocessing import DataPreprocessor

class SensorClassifier(nn.Module):
    """神經網路模型，與sensor_model.py中的模型相同"""
    def __init__(self, input_size=3, hidden_size1=64, hidden_size2=32, num_classes=2):
        super(SensorClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

class DataQualityAgent:
    """
    Agent for detecting abnormal data in sensor data CSV files.
    Identifies which rows contain abnormal sensor readings using both rule-based and model-based approaches.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the agent with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file to analyze
        """
        self.csv_file_path = csv_file_path
        self.model_path = 'model_weight/sensor_classifier.pth'
        self.data = None
        self.model = None
        self.label_encoder = None
        self.preprocessor = DataPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.issues = {
            'temp_issues': [],
            'pressure_issues': [],
            'vibration_issues': [],
            'timestamp_issues': [],
            'label_consistency_issues': []
        }
        
        # Initialize statistics counters
        self.reset_statistics()
    
    def reset_statistics(self):
        """
        Reset all statistics counters to zero.
        """
        self.temp_abnormal_count = 0
        self.pressure_abnormal_count = 0
        self.vibration_abnormal_count = 0
        self.temp_null_count = 0
        self.pressure_null_count = 0
        self.vibration_null_count = 0
        self.rule_abnormal_count = 0
        self.model_abnormal_count = 0
        
    def load_data(self) -> bool:
        """
        Load the CSV data into the agent.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"✅ 成功載入數據: {len(self.data)} 行")
            return True
        except Exception as e:
            print(f"❌ 載入數據失敗: {e}")
            return False
    
    def load_model(self) -> bool:
        """
        Load the trained model for model-based checking.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Initialize model
            self.model = SensorClassifier()
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load label encoder
            self.label_encoder = checkpoint['label_encoder']
            
            print(f"✅ 成功載入模型: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            return False
    
    def preprocess_data_for_model(self) -> np.ndarray:
        """
        Preprocess data using DataPreprocessor in eval mode.
        
        Returns:
            np.ndarray: Preprocessed features
        """
        try:
            # Use eval mode for preprocessing
            df_processed = self.preprocessor.process(self.csv_file_path, mode='eval')
            
            # Extract features
            features = df_processed[['temp', 'pressure', 'vibration']].values
            return features
        except Exception as e:
            print(f"❌ 資料預處理失敗: {e}")
            return None
    
    def check_single_row_rule_based(self, row_data: pd.Series, row_idx: int) -> Dict:
        """
        Check a single row for rule-based abnormalities.
        
        Args:
            row_data (pd.Series): Single row data
            row_idx (int): Row index
            
        Returns:
            Dict: {'is_abnormal': bool, 'alert_message': str}
        """
        temp = row_data['temp']
        pressure = row_data['pressure']
        vibration = row_data['vibration']
        
        # Rule-based checking
        temp_abnormal = False
        pressure_abnormal = False
        vibration_abnormal = False
        temp_null = False
        pressure_null = False
        vibration_null = False
        
        # Check for null values
        if pd.isna(temp):
            temp_null = True
        else:
            temp_abnormal = temp > 52.0 or temp < 43.0
        
        if pd.isna(pressure):
            pressure_null = True
        else:
            pressure_abnormal = pressure > 1.08 or pressure < 0.97
        
        if pd.isna(vibration):
            vibration_null = True
        else:
            vibration_abnormal = vibration > 0.07
        
        rule_abnormal = temp_abnormal or pressure_abnormal or vibration_abnormal or temp_null or pressure_null or vibration_null
        
        if rule_abnormal:
            # Update statistics counters
            self.rule_abnormal_count += 1
            
            if temp_null:
                self.temp_null_count += 1
            elif temp_abnormal:
                self.temp_abnormal_count += 1
            
            if pressure_null:
                self.pressure_null_count += 1
            elif pressure_abnormal:
                self.pressure_abnormal_count += 1
            
            if vibration_null:
                self.vibration_null_count += 1
            elif vibration_abnormal:
                self.vibration_abnormal_count += 1
            
            # Identify which sensors are abnormal
            abnormal_sensors = []
            
            # Check null values first
            if temp_null:
                abnormal_sensors.append("溫度數據缺失")
            elif temp_abnormal:
                if temp > 52.0:
                    abnormal_sensors.append("溫度偏高 (正常範圍: 45-50°C)")
                else:
                    abnormal_sensors.append("溫度偏低 (正常範圍: 45-50°C)")
            
            if pressure_null:
                abnormal_sensors.append("壓力數據缺失")
            elif pressure_abnormal:
                if pressure > 1.08:
                    abnormal_sensors.append("壓力偏高 (正常範圍: 1.00-1.05)")
                else:
                    abnormal_sensors.append("壓力偏低 (正常範圍: 1.00-1.05)")
            
            if vibration_null:
                abnormal_sensors.append("振動數據缺失")
            elif vibration_abnormal:
                abnormal_sensors.append("振動偏高 (正常範圍: 0.02-0.04)")
            
            suggestions = self.generate_suggestions(temp_abnormal, pressure_abnormal, vibration_abnormal, temp_null, pressure_null, vibration_null)
            alert_message = f"[RULE-ALERT] {', '.join(abnormal_sensors)}"
            if suggestions:
                alert_message += f"\n建議: {', '.join(suggestions)}"
            
            return {
                'is_abnormal': True,
                'alert_message': alert_message
            }
        else:
            return {
                'is_abnormal': False,
                'alert_message': ''
            }
    
    def check_single_row_model_based(self, row_data: pd.Series, row_idx: int) -> Dict:
        """
        Check a single row for model-based abnormalities.
        Performs preprocessing (eval_mode) and model prediction for the specific row.
        
        Args:
            row_data (pd.Series): Single row data
            row_idx (int): Row index
            
        Returns:
            Dict: {'is_abnormal': bool, 'alert_message': str}
        """
        if self.model is None:
            return {
                'is_abnormal': False,
                'alert_message': ''
            }
        
        try:
            # Create a DataFrame with just this row for preprocessing
            row_df = pd.DataFrame([row_data])
            
            # Preprocess this single row using eval mode
            # We need to create a temporary CSV file for the preprocessor
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                row_df.to_csv(temp_file.name, index=False)
                temp_csv_path = temp_file.name
            
            try:
                # Preprocess the single row
                df_processed = self.preprocessor.process(temp_csv_path, mode='eval')
                
                # Extract features for this row
                features = df_processed[['temp', 'pressure', 'vibration']].values
                
                # Get model prediction for this single row
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(features).to(self.device)
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predictions = torch.max(outputs, 1)
                    
                    # Get results for this row
                    model_prediction = predictions[0].item()
                    model_probabilities = probabilities[0].cpu().numpy()
                
                # Convert prediction to label
                predicted_label = self.label_encoder.inverse_transform([model_prediction])[0]
                
                # Get the probability for the predicted class
                predicted_probability = model_probabilities[model_prediction]
                
                # Check if model predicts abnormal
                model_abnormal = predicted_label == 'abnormal'
                
                if model_abnormal:
                    # Update model statistics counter
                    self.model_abnormal_count += 1
                    
                    suggestions = ["檢查所有感測器及產品狀況", "進行設備維護檢查"]
                    alert_message = f"[MODEL-ALERT] {predicted_label} (分數: {predicted_probability:.3f})\n建議: {', '.join(suggestions)}"
                    
                    return {
                        'is_abnormal': True,
                        'alert_message': alert_message
                    }
                else:
                    return {
                        'is_abnormal': False,
                        'alert_message': ''
                    }
                    
            finally:
                # Clean up temporary file
                if os.path.exists(temp_csv_path):
                    os.unlink(temp_csv_path)
                    
        except Exception as e:
            print(f"❌ 單行模型檢查失敗 (行 {row_idx + 1}): {e}")
            return {
                'is_abnormal': False,
                'alert_message': ''
            }
    
    def check_all_rows_combined(self):
        """
        Check all rows using both rule-based and model-based approaches.
        Process one row at a time and print results immediately.
        
        Returns:
            Dict: Summary statistics
        """
        print("="*80)
        print("🚨 開始逐行異常檢測 (Rule-based + Model-based)")
        print("="*80)
        
        # Reset statistics before starting
        self.reset_statistics()
        
        # Try to load model for model-based checking
        model_available = False
        
        if self.model is None:
            # Try to load model
            if self.load_model():
                model_available = True
            else:
                print("⚠️  模型載入失敗，將只進行rule-based檢查")
        
        # Process each row
        for idx, row in self.data.iterrows():
            temp = row['temp']
            pressure = row['pressure']
            vibration = row['vibration']
            timestamp = row['timestamp']
            
            # Rule-based check
            rule_result = self.check_single_row_rule_based(row, idx)
            
            # Model-based check (if available)
            model_result = {'is_abnormal': False, 'alert_message': ''}
            if model_available:
                model_result = self.check_single_row_model_based(row, idx)
            
            # Determine overall abnormal status
            has_abnormal = rule_result['is_abnormal'] or model_result['is_abnormal']
            
            if has_abnormal:
                # Format timestamp
                try:
                    dt = pd.to_datetime(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_time = timestamp
                
                # Print row information only once
                print(f"行 {idx + 1} | {formatted_time} | TEMP={temp}°C | PRESSURE={pressure} | VIBRATION={vibration}")
                
                # Print rule-based alert if abnormal
                if rule_result['is_abnormal']:
                    print(rule_result['alert_message'])
                
                # Print model-based alert if abnormal
                if model_result['is_abnormal']:
                    print(model_result['alert_message'])
                
                print("-" * 80)  # Separator between abnormal rows
        
        
        # Return summary statistics
        return {
            'total_rows': len(self.data),
            'check_type': 'combined',
            'combined': {
                'temperature_abnormal': self.temp_abnormal_count,
                'pressure_abnormal': self.pressure_abnormal_count,
                'vibration_abnormal': self.vibration_abnormal_count,
                'temp_null': self.temp_null_count,
                'pressure_null': self.pressure_null_count,
                'vibration_null': self.vibration_null_count,
                'total_abnormal_rows': self.rule_abnormal_count,
                'model_abnormal_rows': self.model_abnormal_count
            }
        }

    def generate_suggestions(self, temp_abnormal: bool, pressure_abnormal: bool, vibration_abnormal: bool, 
                           temp_null: bool = False, pressure_null: bool = False, vibration_null: bool = False) -> List[str]:
        """
        Generate suggestions based on detected abnormalities.
        
        Args:
            temp_abnormal (bool): Whether temperature is abnormal
            pressure_abnormal (bool): Whether pressure is abnormal
            vibration_abnormal (bool): Whether vibration is abnormal
            temp_null (bool): Whether temperature data is missing
            pressure_null (bool): Whether pressure data is missing
            vibration_null (bool): Whether vibration data is missing
            
        Returns:
            List[str]: List of suggestions
        """
        suggestions = []
        
        if temp_null:
            suggestions.extend([
                "檢查溫度感測器狀況",
            ])
        elif temp_abnormal:
            suggestions.extend([
                "檢查冷卻系統",
            ])
        
        if pressure_null:
            suggestions.extend([
                "檢查壓力感測器狀況",
            ])
        elif pressure_abnormal:
            suggestions.extend([
                "檢查產品壓力狀況",
            ])
        
        if vibration_null:
            suggestions.extend([
                "檢查振動感測器狀況",
            ])
        elif vibration_abnormal:
            suggestions.extend([
                "檢查設備固定",
            ])
        
        return suggestions

def main():
    """
    Main function to run the data quality agent.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='異常數據檢測器')
    parser.add_argument('csv_file', help='要分析的CSV檔案路徑')
    args = parser.parse_args()
    
    # Initialize the agent
    agent = DataQualityAgent(args.csv_file)
    
    # Load data first
    if not agent.load_data():
        return
    
    # Run full analysis (results are printed directly during processing)
    summary = agent.check_all_rows_combined()
    
    # Print final summary
    print(f"\n📈 分析完成！總共處理了 {summary['total_rows']} 行數據")
    print(f"🔍 檢查類型: {summary['check_type']}")
    
    combined_summary = summary['combined']
    print(f"📊 統計結果:")
    print(f"   溫度異常: {combined_summary['temperature_abnormal']} 個")
    print(f"   壓力異常: {combined_summary['pressure_abnormal']} 個")
    print(f"   振動異常: {combined_summary['vibration_abnormal']} 個")
    print(f"   溫度空值: {combined_summary['temp_null']} 個")
    print(f"   壓力空值: {combined_summary['pressure_null']} 個")
    print(f"   振動空值: {combined_summary['vibration_null']} 個")
    print(f"   總異常行數[Rule]: {combined_summary['total_abnormal_rows']} 個")
    print(f"   總異常行數[Model]: {combined_summary['model_abnormal_rows']} 個")

if __name__ == "__main__":
    main()