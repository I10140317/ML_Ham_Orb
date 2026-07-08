import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Parameter, Sigmoid
from torch_geometric.nn import MessagePassing

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class GatedGCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.lin_u = Linear(in_channels, out_channels)
        self.att_mlp = Linear(2 * in_channels + edge_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_u.reset_parameters()
        self.att_mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x_u = self.lin_u(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        return ReLU()(out + x_u)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        gate_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = self.att_mlp(gate_input).sigmoid()
        return gate * x_j

class GatedGCNMultiTask(nn.Module):
    def __init__(self, node_input_dim = 3, edge_input_dim = 3, hidden_dim =  32, out_dim = 1, num_layers = 8, dropout = 0.1):
        super().__init__()

        self.node_encoder = MLP(node_input_dim, hidden_dim, hidden_dim)
        self.edge_encoder = MLP(edge_input_dim, hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        for _ in range(num_layers):
            self.convs.append(GatedGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.node_head = MLP(hidden_dim, hidden_dim, out_dim)
        self.edge_head = MLP(hidden_dim * 3, hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, e)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = norm(h + h_new)

        node_pred = self.node_head(h)

        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], e], dim=-1)
        edge_pred = self.edge_head(edge_input)

        return node_pred, edge_pred


