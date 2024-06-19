import torch
import torch.nn as nn

from baselines.cherrypick.networks import GraphEmbedding
from baselines.dimes_tsp.inc.tsp_nets import EmbNet, ParNet


class DimesNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_emb = GraphEmbedding(p=args.graph_emb_dim, k=2)
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
