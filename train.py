import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from model import *
# 数据预处理
train_df = pd.read_csv('/home/liuzilong/data/liuzilong/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/home/liuzilong/data/liuzilong/playground-series-s5e3/test.csv')

# 处理周期特征
def process_periodic_features(df):
    df = df.copy()
    # 处理day
    df['sin_day'] = np.sin(2 * math.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * math.pi * df['day'] / 365)
    df.drop('day', axis=1, inplace=True)
    
    # 处理winddirection
    df['sin_wind'] = np.sin(2 * math.pi * df['winddirection'] / 360)
    df['cos_wind'] = np.cos(2 * math.pi * df['winddirection'] / 360)
    df.drop('winddirection', axis=1, inplace=True)
    return df

train_df = process_periodic_features(train_df)
test_df = process_periodic_features(test_df)

# 分离特征和标签
X_train = train_df.drop(['id', 'rainfall'], axis=1)
y_train = train_df['rainfall'].values
test_ids = test_df['id']
X_test = test_df.drop('id', axis=1)

# 标准化数值特征
numeric_features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 
                   'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 转换为张量
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

# 划分训练集和验证集
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

# 创建Dataset
class RainfallDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

train_dataset = RainfallDataset(X_train_split, y_train_split)
val_dataset = RainfallDataset(X_val_split, y_val_split)
test_dataset = RainfallDataset(X_test_tensor)

# 参数设置
batch_size = 128
input_dim = X_train.shape[1]
hidden_dim = 256
num_blocks = 101
dropout_rate = 0.3
num_classes = 2

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetClassifier(input_dim, hidden_dim, num_blocks, dropout_rate, num_classes).to(device)

# 类别权重处理（处理不平衡数据）
class_counts = np.bincount(y_train)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 优化器和学习率调度
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# 数据加载
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练循环
best_val_acc = 0.0
no_improve = 0

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    train_loss = total_loss / len(train_dataset)
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    val_loss = val_loss / len(val_dataset)
    
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    scheduler.step(val_acc)
    
    # 早停和保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/home/liuzilong/data/liuzilong/checkpoints/rainfall/best_model.pth')
        no_improve = 0
    else:
        no_improve += 1

# 预测测试集
model.load_state_dict(torch.load('/home/liuzilong/data/liuzilong/checkpoints/rainfall/best_model.pth'))
model.eval()
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        predictions.extend(pred.cpu().numpy())

# 生成提交文件
submission = pd.DataFrame({'id': test_ids, 'rainfall': predictions})
submission.to_csv('/home/liuzilong/data/liuzilong/playground-series-s5e3/submission.csv', index=False)