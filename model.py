from torch import nn
# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

# 定义模型
class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, dropout_rate, num_classes):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.fc(x)
        return x
