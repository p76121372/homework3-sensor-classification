# 感測器異常檢測系統

這是一個完整的感測器異常檢測系統，包含數據生成、預處理、模型訓練和異常檢測功能。

## 📁 項目結構

```
和碩作業3/
├── generate_data.py              # 數據生成器
├── preprocessing.py              # 數據預處理器
├── training_model.py             # 模型訓練腳本
├── checking_agent.py             # 異常檢測代理
├── training_history.png          # 訓練歷史圖表
├── requirements.txt              # Python依賴套件
├── model_weight/
│   ├── sensor_classifier.pth     # 訓練好的模型
│   └── preprocessing_stats.pkl   # 預處理統計信息
├── Data/
│   ├── training.csv              # 訓練數據
│   ├── testing.csv               # 測試數據
│   └── testing_50.csv            # 小型測試數據
└── README.md                     # 本說明文件
```

## 🚀 快速開始

### 1. 生成測試數據
```bash
python generate_data.py -n 500 -o Data/testing.csv
```

### 2. 訓練模型
```bash
python training_model.py
```

### 3. 進行異常檢測
```bash
python checking_agent.py Data/testing.csv
```

## 📊 感測器參數

| 感測器 | 正常範圍 | 異常範圍 | 單位 |
|--------|----------|----------|------|
| 溫度 (temp) | 45-50°C | >52°C 或 <43°C | °C |
| 壓力 (pressure) | 1.00-1.05 | >1.08 或 <0.97 | - |
| 振動 (vibration) | 0.02-0.04 | >0.07 | - |

## 🔧 詳細功能說明

### 1. 數據生成器 (`generate_data.py`)

生成模擬的感測器數據，用於測試和驗證異常檢測算法。

#### 功能特點
- **獨立決定**: 每個感測器的值域都是獨立決定的
- **可調機率**: 可調整正常/異常標籤的機率分布
- **空值生成**: 有一定機率產生空值（可調整）

#### 使用方法
```bash
# 基本用法
python generate_data.py

# 進階參數
python generate_data.py -n 500 -o my_data.csv --normal_prob 0.3 --abnormal_prob 0.7 --null_prob 0.05
```

#### 參數說明
- `-n, --num_rows`: 生成數據行數 (默認: 300)
- `-o, --output`: 輸出檔案名稱 (默認: testing.csv)
- `--normal_prob`: normal label時每個感測器在正常值域的機率 (默認: 0.95)
- `--abnormal_prob`: abnormal label時每個感測器在正常值域的機率 (默認: 0.3)
- `--null_prob`: 產生空值的機率 (默認: 0.05)

#### 輸出格式
生成的CSV文件包含以下欄位：
- `timestamp`: 時間戳記 (YYYY-MM-DD HH:MM:SS)
- `temp`: 溫度值 (可能為空值)
- `pressure`: 壓力值 (可能為空值)
- `vibration`: 振動值 (可能為空值)
- `label`: 標籤 (normal/abnormal)

### 2. 數據預處理器 (`preprocessing.py`)

`DataPreprocessor` 類別用於處理CSV檔案的預處理，支援training和eval兩種模式。

#### 功能特點
- **缺失值填補**: 使用平均值填補缺失值
- **Z-score標準化**: 使用平均值和標準差進行標準化
- **統計資訊管理**: 自動儲存和載入統計資訊
- **雙模式支援**: training模式和eval模式

#### 使用方法
```python
from preprocessing import DataPreprocessor

# 創建預處理器
preprocessor = DataPreprocessor()

# Training模式 - 計算統計資訊並處理數據
training_data = preprocessor.process("Data/training.csv", mode='training')

# Eval模式 - 使用已保存的統計資訊處理數據
eval_data = preprocessor.process("Data/testing.csv", mode='eval')
```

#### 模式說明
- **Training模式**: 計算各數值欄位的平均值、標準差和樣本數，並儲存統計資訊
- **Eval模式**: 載入之前training模式儲存的統計資訊進行處理

#### 統計資訊格式
```python
{
    'column_name': {
        'mean': 平均值,
        'std': 標準差,
        'sample_count': 有效樣本數,
        'total_count': 總樣本數
    }
}
```

### 3. 模型訓練 (`training_model.py`)

使用PyTorch實現的三層神經網路模型，用於檢測感測器數據的異常狀態。

#### 模型架構
```
輸入層 (3) → 隱藏層1 (64) → 隱藏層2 (32) → 輸出層 (2)
```

- **輸入**: temp, pressure, vibration (3個特徵)
- **隱藏層1**: 64個神經元，ReLU激活函數，Dropout(0.2)
- **隱藏層2**: 32個神經元，ReLU激活函數，Dropout(0.2)
- **輸出層**: 2個神經元 (normal/abnormal)

#### 使用方法
```bash
python training_model.py
```

#### 訓練參數
- **優化器**: Adam (learning_rate=0.001)
- **損失函數**: CrossEntropyLoss
- **批次大小**: 16
- **訓練輪數**: 100 epochs
- **數據分割**: 80% 訓練, 20% 測試

#### 模型性能
- **準確率**: ~92%
- **支援的標籤**: normal, abnormal
- **輸入格式**: 3個數值 (temp, pressure, vibration)

#### 設備支援
- 自動檢測GPU/CPU
- 支援CUDA加速
- 自動優化設備使用

### 4. 異常檢測代理 (`checking_agent.py`)

綜合的異常檢測系統，結合規則基礎和模型基礎的檢測方法。

#### 功能特點
- **雙重檢測**: 規則基礎 + 模型基礎檢測
- **逐行處理**: 逐行檢查並即時輸出結果
- **詳細統計**: 提供完整的異常統計信息
- **智能建議**: 根據異常類型提供具體建議
- **空值檢測**: 自動檢測並標記空值異常

#### 使用方法
```bash
python checking_agent.py Data/testing.csv
```

#### 檢測類型

##### 規則基礎檢測
- **溫度異常**: 超出正常範圍 (45-50°C)
- **壓力異常**: 超出正常範圍 (1.00-1.05)
- **振動異常**: 超出正常範圍 (0.02-0.04)
- **空值檢測**: 檢測缺失的感測器數據

##### 模型基礎檢測
- 使用訓練好的神經網路模型
- 基於機器學習的異常檢測
- 提供預測信心度

#### 輸出格式
```
行 1 | 2024-06-03 19:00:00 | TEMP=52.5°C | PRESSURE=1.10 | VIBRATION=0.08
[RULE-ALERT] 溫度偏高 (正常範圍: 45-50°C), 壓力偏高 (正常範圍: 1.00-1.05), 振動偏高 (正常範圍: 0.02-0.04)
建議: 檢查冷卻系統, 檢查產品壓力狀況, 檢查設備固定
[MODEL-ALERT] abnormal (分數: 0.875)
建議: 檢查所有感測器及產品狀況, 進行設備維護檢查
```

#### 統計報告
```
📈 分析完成！總共處理了 500 行數據
🔍 檢查類型: combined
📊 統計結果:
   溫度異常: 15 個
   壓力異常: 8 個
   振動異常: 12 個
   溫度空值: 3 個
   壓力空值: 2 個
   振動空值: 1 個
   總異常行數[Rule]: 25 個
   總異常行數[Model]: 18 個
```

## 🛠️ 安裝依賴

```bash
pip install -r requirements.txt
```

## 📋 使用範例

### 完整工作流程

1. **生成測試數據**
```bash
python generate_data.py -n 500 -o Data/testing.csv
```

2. **訓練模型**
```bash
python training_model.py
```

3. **進行異常檢測**
```bash
python checking_agent.py Data/testing.csv
```

### 自定義參數範例

```bash
# 生成高空值率的數據
python generate_data.py -n 500 --null_prob 0.15

# 生成反轉邏輯的數據
python generate_data.py -n 500 --normal_prob 0.3 --abnormal_prob 0.8
```