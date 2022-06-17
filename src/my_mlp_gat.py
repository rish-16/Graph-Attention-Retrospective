from typing import Optional, Tuple, Union
import numpy as np

import torch, math
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
from torch_geometric.datasets import Planetoid


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
    
class Eigen2(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        
    def forward(self, edge_idx, n, sigma):
        # lap_idx, lap_wt = get_laplacian(edge_idx, normalization="sym")
        edge_adj = to_dense_adj(edge_idx)
        eigenvals, eigenvecs = torch.linalg.eig(edge_adj)
        top_eig = eigenvecs.squeeze(0)[:, 1:self.k+1]
        top_eig = torch.real(top_eig)
        
        new_edge_features = torch.Tensor(edge_idx.size(1), 1).to(edge_idx.device)
        new_edge_idx = edge_idx.T
        
        for idx, pair in enumerate(new_edge_idx):
            i, j = pair
            x_i_prime = top_eig[i]
            x_j_prime = top_eig[j]
            dot = torch.dot(x_i_prime, x_j_prime)
            final = 8 * torch.sqrt(torch.log(torch.tensor(n))) * sigma * torch.sign(dot)
            new_edge_features[idx] = final
            
        return new_edge_features.view(-1, 1), eigenvals, top_eig

class GATv3Layer(MessagePassing):
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

        self.att_in = Linear(att_in_channels, att_out_channels, bias=bias, weight_initializer='glorot') 

        self.att_out = Linear(att_out_channels, 1, bias=False, weight_initializer='glorot') 

        self.lin = Linear(in_channels, out_channels, bias=bias, weight_initializer='glorot')

        self._alpha = None
        self._pair_pred = None
        self._phi_attention_score = None
        self._psi_attention_score = None       
        self._indic = None       

        self.reset_parameters()

    def reset_parameters(self):
        self.att_in.reset_parameters()
        self.att_out.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, cur_mu=0, sigma=0, return_attention_info=None):
    
        # print (x.shape, edge_index.shape, edge_attr.shape)
            
        x = self.lin(x)

        # num_nodes = x.size(0)
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None, cur_mu=cur_mu, sigma=sigma)

        alpha = self._alpha
        pair_pred = self._pair_pred
        phi_attention_score = self._phi_attention_score
        psi_attention_score = self._psi_attention_score
        indic = self._indic

        self._alpha = None
        self._phi_attention_score = None
        self._psi_attention_score = None
        self._pair_pred = None
        self._indic = None

        out = out.mean(dim=1)

        if isinstance(return_attention_info, bool):
            assert alpha is not None
            assert pair_pred is not None
            assert phi_attention_score is not None
            assert psi_attention_score is not None            
            return out, (edge_index, alpha, phi_attention_score, psi_attention_score, indic), pair_pred
        else:
            return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i, cur_mu, sigma):
        def indicator(n, sigma, mu):
            if np.linalg.norm(mu) >= 0.5 * sigma * np.sqrt(2 * np.log(n)):
                return 1
            else:
                return 0
        
        print ("with indicator", cur_mu, sigma)
        
        """
        if distance between means (mu) is not that big, we use edge_attr
        """
        
        cat = torch.cat([x_i, x_j], dim=1)
        # [E, 1] -> r(LRelu(S(Wx)))
        node_attr = self.att_out(F.leaky_relu(self.att_in(cat), 0.2)) 
        
        indic = indicator(x_i.size(0), sigma, cur_mu)
        if indic == 1: # distance is big
            print ("Activating Phi")
            attn = node_attr # phi
        else: # distance is small
            print ("Activating Psi")
            attn = edge_attr # psi
            
        # attn = (1 - indic)*node_attr + indic*edge_attr
        
        print ("Indicator:", indic)
        
        self._phi_attention_score = node_attr
        self._psi_attention_score = edge_attr
        self._pair_pred = attn
        self._indic = indic
        
        # attn = n
        
        gamma = tg.utils.softmax(attn, index, ptr, size_i) # [E, 1]
        msg = gamma * x_j # [E, d]
        
        self._alpha = gamma # edge-wise score
        
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
    
# class GATv3(nn.Module):
#     def __init__(self, indim, k):
#         super().__init__()

#         self.eigen = Eigen2(k)
#         self.gat1 = my_MLP_GATConv(
#                         in_channels=indim, # w_in
#                         out_channels=1, # w_out
#                         att_in_channels=2,
#                         att_out_channels=2,
#                         add_self_loops=True
#                     )

#     def forward(self, x, edge_idx, p, q, sigma):
#         eigen_x = self.eigen(edge_idx, p, q, x.size(0), sigma)
#         out = self.gat1(x, edge_idx, edge_attr=eigen_x, p=p, q=q)

#         return out

# datasets  = [Planetoid(root='data/CiteSeer/', name='CiteSeer')]

# def get_graph_stats(edge_idx, y):
#     new_edge_idx = edge_idx.T
#     total_edges = new_edge_idx.size(0)
#     p = 0
#     q = 0
#     for idx, pair in enumerate(new_edge_idx):
#         i, j = pair
#         if y[i] == y[j]:
#             p += 1
#         else:
#             q += 1
    
#     p = p/total_edges
#     q = q/total_edges
        
#     return p, q
    
# for dt in datasets:
#     print (dt)
#     x = dt.data.x
#     e = dt.data.edge_index
#     y = dt.data.y
#     n = x.size(0)

#     p, q = get_graph_stats(e, y)
#     sigma = torch.std(x)
    
#     g = GATv3(dt.num_features, 1)
#     pred = g(x, e, p=p, q=q, sigma=sigma)
#     print (pred.shape)