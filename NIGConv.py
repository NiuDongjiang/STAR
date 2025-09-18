import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair

class GatedGeometricAwareNIGConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_feat_dim,
                 feat_drop=0.1,
                 bias=True):
        super().__init__()

        self.feat_drop = nn.Dropout(feat_drop)

        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, in_feats),
            nn.LeakyReLU(),
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

        # 修改为输出2倍维度以用于Softmax竞争比例融合
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(2 * in_feats + edge_feat_dim, in_feats),
            nn.LeakyReLU(),
            nn.Linear(in_feats, 2 * in_feats)  # 输出两个竞争通道
        )
        self._out_feats = out_feats
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=True)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def message(self, edges):
        src_h = edges.src['h']
        dst_h = edges.dst['h']
        edge_feat = edges.data['he']

        weight = edges.data['_edge_weight']

        gate_input = torch.cat([src_h, dst_h, edge_feat], dim=-1)
        gate_logits = self.edge_gate_mlp(gate_input).view(-1, 2, src_h.shape[-1])

        gate_weights = F.softmax(gate_logits, dim=1)
        gate_self, gate_neigh = gate_weights[:, 0, :], gate_weights[:, 1, :]

        gated_message = gate_neigh * src_h + gate_self * dst_h

        return {'m': gated_message}

    def forward(self, graph, feat, edge_feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)

            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst
            graph.edata['he'] = edge_feat
            graph.edata['_edge_weight'] = self.edge_weight_mlp(edge_feat)

            graph.update_all(self.message, fn.mean('m', 'neigh'))

            h_neigh = graph.dstdata['neigh']
            rst = self.fc_self(feat_dst) + self.fc_neigh(h_neigh)
            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            return rst
