import torch.nn.functional as F
import torch.nn as nn

class BaseCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(p=0.35)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(p=0.35)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(p=0.35)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout4 = nn.Dropout2d(p=0.35)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.dropout5 = nn.Dropout2d(p=0.35)
        
        self.conv6 = nn.Conv2d(1024, 2000, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(2000)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout6 = nn.Dropout2d(p=0.35)
        
        self.fc1 = nn.Linear(2000 * 7 * 7, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout6(x)
        
        x = x.view(-1, 2000 * 7 * 7)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout7(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
