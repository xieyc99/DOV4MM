import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleNet, self).__init__()
        
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_channels, hidden_channels)
        
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(hidden_channels, hidden_channels)
        
        # 定义输出层
        self.output = nn.Linear(hidden_channels, out_channels)
        
        # 定义激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.sigmoid(self.output(x))
        return x