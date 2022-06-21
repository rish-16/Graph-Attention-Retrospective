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
    
class OGGATLayer(MessagePassing):
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
        
        C = self.out_channels

        x_: OptTensor = None
            
        assert x.dim() == 2
        x_ = self.lin(x).view(-1, C)

        assert x_ is not None

        if self.add_self_loops:
            num_nodes = x_.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x_, size=None)

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


    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        tmp = torch.cat([x_i, x_j], dim=1)
        tmp = self.att_in(tmp)
        tmp = F.leaky_relu(tmp, self.negative_slope)
        tmp = self.att_out(tmp)
        self._pair_pred = tmp
        alpha = softmax(tmp, index, ptr, size_i)
        self._alpha = alpha

        return x_j * alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')

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

        self.reset_parameters()

    def reset_parameters(self):
        self.att_in.reset_parameters()
        self.att_out.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_feat, cur_mu, return_attention_info=None):
        sigma = torch.std(x)
            
        x = self.lin(x)

        num_nodes = x.size(0)
        edge_index, edge_feat = remove_self_loops(edge_index, edge_attr=edge_feat)
        edge_index, edge_feat = add_self_loops(edge_index, edge_attr=edge_feat, num_nodes=num_nodes)
        
        print (f"# nodes: {x.size(0)}")
        out = self.propagate(edge_index=edge_index, x=x, size=None, sigma=sigma, cur_mu=cur_mu, numnodes=x.size(0), edge_feat=edge_feat)

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

    def message(self, x_j, x_i, index, ptr, size_i, sigma, cur_mu, numnodes, edge_feat):
        
        def indicator_fn(nnodes, cursigma, curmu):
            return np.linalg.norm(curmu) >= 1 * cursigma * np.sqrt(2 * np.log(nnodes))

        # [E, 1] -> r(LRelu(S(Wx)))
        cat = torch.cat([x_i, x_j], dim=1)
        node_attr = self.att_out(F.leaky_relu(self.att_in(cat), 0.2)) 
        
        indic = indicator_fn(numnodes, sigma, cur_mu)
        print ("Indicator Preference:", "Phi" if indic else "Psi")
         
        attn = node_attr if indic else edge_feat
        self._pair_pred = attn
        
        gamma = tg.utils.softmax(attn, index, ptr, size_i) # [E, 1]
        msg = x_j * gamma
        
        self._alpha = gamma # edge-wise score
        
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
    
