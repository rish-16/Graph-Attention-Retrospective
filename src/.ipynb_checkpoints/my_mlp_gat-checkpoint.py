from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax, get_laplacian, to_dense_adj

class Eigen(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, edge_idx):
        lap_idx, lap_wt = get_laplacian(edge_idx, normalization="sym")
        lap_adj = to_dense_adj(lap_idx)
        eigenvals, eigenvecs = torch.linalg.eig(lap_adj)
        top_eig = eigenvecs.squeeze(0)[:, 1:self.k+1]
        top_eig = torch.real(top_eig)
        new_edge_features = torch.Tensor(edge_idx.size(1), 2 * self.k).to(edge_idx.device)
        new_edge_idx = edge_idx.T

        for idx, pair in enumerate(new_edge_idx):
            i, j = pair
            x_i_prime = top_eig[i]
            x_j_prime = top_eig[j]
            new_feat = torch.cat([x_i_prime, x_j_prime], dim=0)
            new_edge_features[idx] = new_feat

        return new_edge_features

class my_MLP_GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        att_in_channels: int,
        att_out_channels: int,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_in_channels = att_in_channels
        self.att_out_channels = att_out_channels
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        self.att_in = Linear(att_in_channels, att_out_channels, bias=bias,
                            weight_initializer='glorot') 

        self.att_out = Linear(att_out_channels, 1, bias=False,
                            weight_initializer='glorot') 

        self.lin = Linear(in_channels, out_channels, bias=bias,
                            weight_initializer='glorot')

        self._alpha = None
        self._pair_pred = None

        self.reset_parameters()

    def reset_parameters(self):
        self.att_in.reset_parameters()
        self.att_out.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_info: bool = None):
            
        x = self.lin(x)

        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        alpha = self._alpha
        pair_pred = self._pair_pred
        self._alpha = None
        self._pair_pred = None

        out = out.mean(dim=1)

        if isinstance(return_attention_info, bool):
            assert alpha is not None
            assert pair_pred is not None
            return out, (edge_index, alpha), pair_pred
        else:
            return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
    
        cat = torch.cat([x_i, x_j], dim=1)
        
        # alpha and beta are set to 1 
        node_attr = self.att_in(cat) # [E, d]
        
        temp = F.leaky_relu(node_attr + edge_attr) # [E, d]
        project = self.att_out(temp)
        self._pair_pred = project
        gamma = tg.utils.softmax(project, index, ptr, size_i) # [E, d]
        msg = gamma * x_j # [E, d]
        
        self._alpha = gamma # edge-wise score
        
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

class GATv3(nn.Module):
    def __init__(self, k):
        super().__init__()

        self.eigen = Eigen(k)
        self.gat1 = my_MLP_GATConv(
                        in_channels=1, # w_in
                        out_channels=1, # w_out
                        att_in_channels=2,
                        att_out_channels=2,
                        add_self_loops=True
                    )

    def forward(self, x, edge_idx):
        eigen_x = self.eigen(edge_idx)
        out = self.gat1(x, edge_idx, edge_attr=eigen_x)

        return out
    
# g = GATv3(1)
# x = torch.rand(200, 1)
# e = torch.randint(0, 200, size=(2, 356))
# y = g(x, e)
# print (y.shape)