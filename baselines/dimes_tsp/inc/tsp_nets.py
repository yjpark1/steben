import torch
import torch.nn as nn
from baselines.dimes_tsp.inc.tsp_utils import *
from copy import deepcopy
# from baselines.cherrypick.networks import VertexEmbedding
import torch_geometric.nn as gnn


# GNN for edge embeddings
class EmbNet(nn.Module):
    @classmethod
    def make(cls, args):
        return cls(args.emb_depth, args.graph_emb_dim, args.net_units, args.net_act_fn, args.emb_agg_fn)

    def __init__(self, depth, feats, units, act_fn, agg_fn):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x  # dim: (nodes, node_feature)
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)  # dim: (edges, gnn_feature)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = act_fn
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def reset_parameters(self):
        for layer in self.lins:
            layer.reset_parameters()

    @staticmethod
    def is_trainable(par):
        return par.requires_grad

    def trainables(self):
        for par in self.parameters():
            if self.is_trainable(par):
                yield par

    def named_trainables(self):
        for name, par in self.named_parameters():
            if self.is_trainable(par):
                yield name, par

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth, units, preds, act_fn):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)

    # copy architecture but reinitialize weights
    def twin(self):
        return type(self)(depth=self.depth, units=self.units, preds=self.preds, act_fn=self.act_fn).to(self.device)

    # copy architecture and weights
    def clone(self):
        return deepcopy(self)

    @classmethod
    def make(cls, args):
        return cls(args.par_depth, args.net_units, 1, args.net_act_fn).to(args.device)


class DimesNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_net = EmbNet.make(args)
        self.par_net = ParNet.make(args)

    def forward(self, x, edge_index, edge_attr, emb_net=None, par_net=None):
        return self.infer(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            emb_net=self.emb_net if emb_net is None else emb_net,
            par_net=self.par_net if par_net is None else par_net,
        )

    @staticmethod
    def infer(x, edge_index, edge_attr, emb_net, par_net):
        emb = emb_net(x, edge_index, edge_attr)
        par = par_net(emb)
        return par
