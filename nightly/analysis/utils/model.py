"""
神经网络模型（精简版）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FFNN(nn.Module):
    """5层前馈神经网络"""

    def __init__(self, input_dim, width=4):
        super(FFNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, width * 16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(width * 16, width * 8),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(width * 8, width * 4),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(width * 2, width * 1),
            nn.ReLU(),

            nn.Linear(width * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train(self, X_train, y_train, epochs=20, batch_size=128, verbose=False):
        """
        训练模型

        Args:
            X_train: 训练特征，shape (n_samples, n_features)
            y_train: 训练标签，shape (n_samples,)
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息

        Raises:
            ValueError: 如果输入数据无效
        """
        # 输入验证
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train cannot be None or empty")
        if y_train is None or len(y_train) == 0:
            raise ValueError("y_train cannot be None or empty")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Shape mismatch: X_train has {len(X_train)} samples, "
                f"y_train has {len(y_train)} samples"
            )
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).view(-1, 1)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.4f}')

    def predict(self, X):
        """预测标签"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).float()
        return predictions.cpu().numpy().flatten()

    def predict_proba(self, X):
        """预测概率"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy().flatten()
