from torch.nn import Linear
from torch.nn import functional as Func
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch
import torch_geometric
torch_geometric.set_debug(True)


class GCN(torch.nn.Module):
    def __init__(self, features_num, hidden_channels, seed=1):
        super(GCN, self).__init__()
        torch.manual_seed(12345)  # 12345
        self.conv1 = GCNConv(features_num, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//4)
        self.conv3 = GCNConv(hidden_channels//4, hidden_channels // 4)
        self.lin = Linear(hidden_channels//4, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index, relevant_batch_indices, labels):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.lin(x[relevant_batch_indices])
        softmax_vals = Func.softmax(x, dim=1)
        lsm = self.logsoftmax(x)
        if lsm.shape[0] > labels.shape[0]:
            lsm = lsm[:labels.shape[0]]
        return lsm, softmax_vals
