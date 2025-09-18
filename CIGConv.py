import torch as th
import torch.nn.functional as F
from torch import nn
from dgl.utils import expand_as_pair
import dgl.function as fn
import torch

class DualGatedEquivariantCIGConv(nn.Module):
    def __init__(self, input_dim, output_dim, edge_feat_dim, drop=0.1, chem_dim=32):
        super(DualGatedEquivariantCIGConv, self).__init__()
        self.chem_dim = chem_dim
        self.geom_dim = edge_feat_dim - chem_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.chem_gate = nn.Sequential(
            nn.Linear(self.chem_dim + 2*self.input_dim, self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        )
        self.geom_gate = nn.Sequential(
            nn.Linear(self.geom_dim + 2*self.input_dim, self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, self.input_dim)
        )
        self.fuse = nn.Linear(self.input_dim, 2*self.input_dim)  # => (2,d)

        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def message(self, edges):
        src_h = edges.src['hn']
        dst_h = edges.dst['hn']
        e = edges.data['he']  # shape [E, edge_feat_dim]

        chem_part = e[:, :self.chem_dim]
        geom_part = e[:, self.chem_dim:]

        chem_in = torch.cat([src_h, dst_h, chem_part], dim=-1)  # => [E, 2*d + chem_dim]
        geom_in = torch.cat([src_h, dst_h, geom_part], dim=-1)  # => [E, 2*d + geom_dim]

        w_chem = self.chem_gate(chem_in)    # => [E, d]
        w_geom = self.geom_gate(geom_in)    # => [E, d]

        w_total = w_chem + w_geom          # => [E, d]

        gate_logits = self.fuse(w_total).view(-1, 2, self.input_dim)
        gate_weights = F.softmax(gate_logits, dim=1)  # => [E, 2, d]

        gate_self, gate_neigh = gate_weights[:, 0, :], gate_weights[:, 1, :]  # => [E, d]
        w_edge = edges.data['edge_weight']  # => [E, 1], scalar

        m = gate_neigh * src_h + gate_self * dst_h + e
        return {'m': m}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.dstdata['hn'] = feat_dst
            graph.edata['he'] = edge_feat  # [E, chem_emb_dim + geom_emb_dim]

            graph.edata['edge_weight'] = self.edge_weight_mlp(edge_feat)

            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = feat_dst + graph.dstdata['neigh']
            rst = self.mlp(rst)
            return rst


