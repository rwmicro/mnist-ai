import torch.nn.functional as F
import torch.nn as nn

class SubNetwork(nn.Module):
    def __init__(self, input_size):
        super(SubNetwork, self).__init__()
        # Define fully connected layers for the subnetwork
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for MNIST/Fashion-MNIST
        # Define batch normalization layer
        self.bn = nn.BatchNorm1d(512)
        # Define dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Forward pass through fully connected layers
        x = F.relu(self.bn(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x