import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from preprocessing import DataPreprocessor

class SensorClassifier(nn.Module):
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

def load_and_preprocess_data(csv_file):
    """載入並預處理數據"""
    # 使用DataPreprocessor進行預處理
    preprocessor = DataPreprocessor()
    
    # 使用training模式進行預處理
    df_processed = preprocessor.process(csv_file, mode='training')
    
    # 準備特徵和標籤
    X = df_processed[['temp', 'pressure', 'vibration']].values
    y = df_processed['label'].values
    
    # 編碼標籤
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # 由於DataPreprocessor已經進行了Z-score標準化，所以不需要額外的scaler
    # 但為了保持與原代碼的兼容性，我們創建一個虛擬的scaler
    class DummyScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit_transform(self, X):
            # 直接返回已經標準化的數據
            return X
        
        def transform(self, X):
            # 直接返回已經標準化的數據
            return X
        
        def inverse_transform(self, X):
            # 使用preprocessor的統計資訊進行反轉換
            stats = preprocessor.get_stats()
            if stats is None:
                return X
            
            X_original = X.copy()
            for i, col in enumerate(['temp', 'pressure', 'vibration']):
                if col in stats:
                    X_original[:, i] = X[:, i] * stats[col]['std'] + stats[col]['mean']
            return X_original
    
    scaler = DummyScaler()
    
    return X, y_encoded, label_encoder, scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    """訓練模型"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 計算平均損失和準確率
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, '
                  f'Train Acc: {train_accuracy:.2f}%, '
                  f'Val Acc: {val_accuracy:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 繪製損失
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 繪製準確率
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(train_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 檢測設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用設備: {device}")
    if device.type == 'cuda':
        print(f"   GPU型號: {torch.cuda.get_device_name(0)}")
        print(f"   GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("   CPU模式")
    
    # 載入和預處理數據
    print("📊 載入數據...")
    X, y, label_encoder, scaler = load_and_preprocess_data('Data/training.csv')
    
    # 分割數據
    # strtify 為確保比例
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 轉換為PyTorch張量並移至設備
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 創建數據加載器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 創建模型並移至設備
    model = SensorClassifier(input_size=3, hidden_size1=64, hidden_size2=32, num_classes=2)
    model = model.to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🚀 開始訓練...")
    print(f"📈 訓練數據: {len(X_train)} 樣本")
    print(f"🧪 測試數據: {len(X_test)} 樣本")
    print(f"🏗️  模型架構: 3 -> 64 -> 32 -> 2")
    
    # 訓練模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=100
    )
    
    # 評估模型
    model.eval()
    test_predictions = []
    test_true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
    
    # 計算準確率
    accuracy = accuracy_score(test_true_labels, test_predictions)
    
    print(f"\n🎯 最終測試準確率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 分類報告
    class_names = label_encoder.classes_
    print(f"\n📋 分類報告:")
    print(classification_report(test_true_labels, test_predictions, 
                              target_names=class_names))
    
    # 繪製訓練歷史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'scaler': scaler,
        'accuracy': accuracy
    }, 'sensor_classifier.pth')
    
    print(f"\n💾 模型已保存為 'sensor_classifier.pth'")
    print(f"📊 訓練歷史圖已保存為 'training_history.png'")
    
    # 顯示一些預測範例
    print(f"\n🔍 預測範例:")
    model.eval()
    with torch.no_grad():
        sample_inputs = X_test_tensor[:5]
        sample_outputs = model(sample_inputs)
        _, sample_predicted = torch.max(sample_outputs, 1)
        
        for i in range(5):
            true_label = label_encoder.inverse_transform([y_test[i]])[0]
            pred_label = label_encoder.inverse_transform([sample_predicted[i].item()])[0]
            features = scaler.inverse_transform(X_test[i].reshape(1, -1))[0]
            print(f"  樣本 {i+1}: temp={features[0]:.1f}°C, pressure={features[1]:.2f}, "
                  f"vibration={features[2]:.2f} | 真實: {true_label} | 預測: {pred_label}")

if __name__ == "__main__":
    main()