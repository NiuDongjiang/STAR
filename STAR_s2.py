import torch.nn as nn
import dgl
from dgl.nn.pytorch import edge_softmax
import torch
from CIGConv import CIGConv, EquivariantCIGConv, GatedEquivariantCIGConv, DualGatedEquivariantCIGConv
from NIGConv import NIGConv, GeometricAwareNIGConv, GatedGeometricAwareNIGConv
from HGC import HeteroGraphConv
from torch_geometric.utils import to_dense_batch
from util import *
from noise import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
from typing import Union


class DTIPredictor(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, tau, layer_num=3):
        super(DTIPredictor, self).__init__()

        self.convs = nn.ModuleList()

        for _ in range(layer_num):
            convl = DualGatedEquivariantCIGConv(hidden_feat_size, hidden_feat_size, hidden_feat_size, chem_dim=hidden_feat_size // 2)
            convp = DualGatedEquivariantCIGConv(hidden_feat_size, hidden_feat_size, hidden_feat_size, chem_dim=hidden_feat_size // 2)
            convlp = GatedGeometricAwareNIGConv(hidden_feat_size, hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            convpl = GatedGeometricAwareNIGConv(hidden_feat_size, hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            conv = HeteroGraphConv(
                {
                    'intra_l': convl,
                    'intra_p': convp,
                    'inter_l2p': convlp,
                    'inter_p2l': convpl
                }
            )
            self.convs.append(conv)

        self.lin_node_l = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_node_p = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_edge_ll = nn.Linear(edge_feat_size, hidden_feat_size)
        self.lin_edge_pp = nn.Linear(edge_feat_size, hidden_feat_size)
        self.chem_dim_raw = 6
        self.geom_dim_raw = 11
        self.chem_emb_dim = hidden_feat_size // 2
        self.geom_emb_dim = hidden_feat_size // 2

        self.lin_edge_ll_chem = nn.Linear(self.chem_dim_raw, self.chem_emb_dim)
        self.lin_edge_ll_geom = nn.Linear(self.geom_dim_raw, self.geom_emb_dim)

        self.lin_edge_pp_chem = nn.Linear(self.chem_dim_raw, self.chem_emb_dim)
        self.lin_edge_pp_geom = nn.Linear(self.geom_dim_raw, self.chem_emb_dim)
        self.lin_edge_lp = nn.Linear(11, hidden_feat_size)
        self.lin_edge_pl = nn.Linear(11, hidden_feat_size)

        self.s2 = S2(hidden_feat_size, hidden_feat_size, hidden_feat_size)

        self.bias_ligandpocket = BiasCorrectionLigandPocket(hidden_feat_size, hidden_feat_size, hidden_feat_size)
        self.bias_pocketligand = BiasCorrectionPocketLigand(hidden_feat_size, hidden_feat_size, hidden_feat_size)
        self.tau = tau
        self.tau_raw_l2p = nn.Parameter(torch.tensor(self.tau))
        self.tau_raw_p2l = nn.Parameter(torch.tensor(self.tau))
        self.tau_raw_pos = nn.Parameter(torch.tensor(self.tau))
        self.tau_raw_neg = nn.Parameter(torch.tensor(self.tau))

    def forward(self, bg, device):
        atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']
        atom_feats = {
            'ligand': self.lin_node_l(atom_feats['ligand'][:, :-3]),
            'pocket': self.lin_node_p(atom_feats['pocket'][:, :-3])
        }
        bond_ll = bond_feats[('ligand', 'intra_l', 'ligand')]  # [E_ll, 17]
        bond_pp = bond_feats[('pocket', 'intra_p', 'pocket')]  # [E_pp, 17]
        bond_lp = bond_feats[('ligand', 'inter_l2p', 'pocket')]  # [E_lp, 11]
        bond_pl = bond_feats[('pocket', 'inter_p2l', 'ligand')]  # [E_pl, 11]
        angle_cols_intra = [6, 7, 8]
        angle_cols_inter = [0, 1, 2]
        #bond_ll = periodic_transform(bond_ll, angle_cols_intra)
        #bond_pp = periodic_transform(bond_pp, angle_cols_intra)
        #bond_lp = periodic_transform(bond_lp, angle_cols_inter)
        #bond_pl = periodic_transform(bond_pl, angle_cols_inter)

        chem_ll_raw = bond_ll[:, :self.chem_dim_raw]  # [E_ll, 6]
        geom_ll_raw = bond_ll[:, self.chem_dim_raw:]
        chem_ll_emb = self.lin_edge_ll_chem(chem_ll_raw)  # -> [E_ll, chem_emb_dim]
        geom_ll_emb = self.lin_edge_ll_geom(geom_ll_raw)  # -> [E_ll, geom_emb_dim]
        bond_ll_fused = torch.cat([chem_ll_emb, geom_ll_emb], dim=-1)
        chem_pp_raw = bond_pp[:, :self.chem_dim_raw]
        geom_pp_raw = bond_pp[:, self.chem_dim_raw:]
        chem_pp_emb = self.lin_edge_pp_chem(chem_pp_raw)
        geom_pp_emb = self.lin_edge_pp_geom(geom_pp_raw)
        bond_pp_fused = torch.cat([chem_pp_emb, geom_pp_emb], dim=-1)

        bond_feats = {
            ('ligand', 'intra_l', 'ligand'): bond_ll_fused,
            ('pocket', 'intra_p', 'pocket'): bond_pp_fused,
            ('ligand', 'inter_l2p', 'pocket'): self.lin_edge_lp(bond_lp),
            ('pocket', 'inter_p2l', 'ligand'): self.lin_edge_pl(bond_pl),
        }

        bg.edata['e'] = bond_feats

        rsts = atom_feats
        for conv in self.convs:
            rsts = conv(bg, rsts)

        bg.nodes['ligand'].data['h'] = rsts['ligand']
        bg.nodes['pocket'].data['h'] = rsts['pocket']

        atompairs_lp, atompairs_pl = self.s2(bg)

        bias_lp = self.bias_ligandpocket(bg)
        bias_pl = self.bias_pocketligand(bg)

        return (atompairs_lp - bias_lp).view(-1), (atompairs_pl - bias_pl).view(-1)

def periodic_transform(x, angle_cols):
    x = x.clone()
    x[:, angle_cols] = torch.sin(x[:, angle_cols])
    return x

import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class S2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(S2, self).__init__()

        self.prj_lp_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.prj_pl_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.att_lp = nn.Linear(hidden_feat_size * 3, 1)
        self.att_pl = nn.Linear(hidden_feat_size * 3, 1)

        self.fc_lp = nn.Linear(hidden_feat_size, 1)
        self.fc_pl = nn.Linear(hidden_feat_size, 1)

    def apply_attention_edges(self, edges, attn_layer):
        z = torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=-1)
        attn_score = attn_layer(z)
        interaction = edges.data['e'] * edges.src['h'] * edges.dst['h']
        return {'attn_score': attn_score, 'interaction': interaction}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']

            g.nodes['ligand'].data['h'] = self.prj_lp_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_lp_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_lp_edge(g.edges['inter_l2p'].data['e'])

            g.apply_edges(lambda edges: self.apply_attention_edges(edges, self.att_lp), etype='inter_l2p')
            attn_lp = dgl.softmax_edges(g, 'attn_score', etype='inter_l2p')
            g.edges['inter_l2p'].data['interaction'] *= attn_lp
            logit_lp = self.fc_lp(g.edges['inter_l2p'].data['interaction'])
            g.edges['inter_l2p'].data['logit_lp'] = logit_lp
            logit_lp_sum = dgl.sum_edges(g, 'logit_lp', etype='inter_l2p')

            g.nodes['ligand'].data['h'] = self.prj_pl_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_pl_dst(node_pocket_feats)

            g.apply_edges(lambda edges: self.apply_attention_edges(edges, self.att_pl), etype='inter_p2l')
            attn_pl = dgl.softmax_edges(g, 'attn_score', etype='inter_p2l')
            g.edges['inter_p2l'].data['interaction'] *= attn_pl
            logit_pl = self.fc_pl(g.edges['inter_p2l'].data['interaction'])
            g.edges['inter_p2l'].data['logit_pl'] = logit_pl
            logit_pl_sum = dgl.sum_edges(g, 'logit_pl', etype='inter_p2l')

            return logit_lp_sum, logit_pl_sum


class BiasCorrectionLigandPocket(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(BiasCorrectionLigandPocket, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w': w}

    def apply_scores(self, edges):
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_l2p'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_l2p')
            scores = edge_softmax(g['inter_l2p'], g.edges['inter_l2p'].data['w'])

            g.edges['inter_l2p'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_l2p')

            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_l2p'))

            return bias


class BiasCorrectionPocketLigand(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(BiasCorrectionPocketLigand, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w': w}

    def apply_scores(self, edges):
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_p2l'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_p2l')
            scores = edge_softmax(g['inter_p2l'], g.edges['inter_p2l'].data['w'])

            g.edges['inter_p2l'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_p2l')

            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_p2l'))

            return bias


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
