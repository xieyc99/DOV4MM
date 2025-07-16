import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        
        # 使用global mean pooling汇总图的信息
        x = global_mean_pool(x, data.batch)
        
        return F.log_softmax(x, dim=1)
