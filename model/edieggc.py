import dgl
import dgl.function as fn
import torch
from dgl.nn import SumPooling, AvgPooling

from model.utils import RBFExpansion, BentIdentity
import torch as th
import torch.nn.functional as F
from torch import nn
from dgl.utils import expand_as_pair


class GINEConv(nn.Module):
    def __init__(self,
                 apply_func=None,
                 init_eps=0,
                 learn_eps=False):
        super(GINEConv, self).__init__()
        self.apply_func = apply_func
        if learn_eps:
            self.eps = nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def message(self, edges):
        r"""User-defined Message Function"""
        return {'m': F.relu(edges.src['hn'] + edges.data['he'])}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.edata['he'] = edge_feat
            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst


class GINConv(nn.Module):
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        self.edge_update = nn.Sequential(
            nn.Linear(input_features, output_features),
            BentIdentity()
        )
        self.apply_fun = nn.Sequential(
            nn.Linear(input_features, output_features),
            BentIdentity(),
            nn.Linear(input_features, output_features),
            BentIdentity(),
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        g = g.local_var()

        g.ndata["e_src"] = node_feats
        g.apply_edges(lambda edges: {'y': self.edge_update(edges.src['e_src'] + edge_feats)/2})
        g.update_all(fn.copy_e('y', 'm'), fn.sum('m', 'x'))
        x = node_feats + g.dstdata['x']/10
        x = self.apply_fun(x/2)
        y = g.edata.pop("y")

        return x, y


class EGGConv(nn.Module):

    def __init__(
        self, input_features: int, output_features: int):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.
        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        return x, m


class Conv(nn.Module):
    def __init__(self, in_feat, out_feat, residual=True):
        super().__init__()
        self.residual = residual
        self.gnn = EGGConv(in_feat, out_feat)
        self.bn_nodes = nn.BatchNorm1d(out_feat)
        self.bn_edges = nn.BatchNorm1d(out_feat)

    def forward(self, g, x_in, y_in):
        x, y = self.gnn(g, x_in, y_in)
        x, y = F.silu(self.bn_nodes(x)), F.silu(self.bn_edges(y))

        if self.residual:
            x = x + x_in
            y = y + y_in

        return x, y


class EquivBlock(nn.Module):
    def __init__(self, args, residual=True):
        super().__init__()
        self.residual = residual
        self.W_h = nn.Linear(args.hidden_features, args.hidden_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        v: torch.Tensor,
        x: torch.Tensor,
    ):
        g = g.local_var()

        g.ndata['x_i'] = self.W_h(x)/256
        g.ndata['x_j'] = -self.W_h(x)/256
        g.apply_edges(fn.u_add_v("x_i", "x_j", "x_nodes"))
        phi = g.edata.pop("x_nodes")
        g.edata['v'] = g.edata['u'].unsqueeze(2) * phi.unsqueeze(1)

        if self.residual:
            g.edata['v'] = (v + g.edata['v']) / 2

        return g.edata['v']


class EDiEGGCConv(nn.Module):
    """Line graph update."""

    def __init__(self, args, line_graph: bool, residual: bool = True):
        super().__init__()
        self.line_graph = line_graph
        if self.line_graph:
            self.edge_update = Conv(args.hidden_features, args.hidden_features, residual)

        self.node_update = Conv(args.hidden_features, args.hidden_features, residual)

        self.equiv_block = EquivBlock(args, residual)

    def forward(self, g, lg, v_in, x_in, y_in, z_in):
        g = g.local_var()

        x, y = self.node_update(g, x_in, y_in)
        v = self.equiv_block(g, v_in, x)
        z = z_in
        if self.line_graph:
            lg = lg.local_var()
            # Edge-gated graph convolution update on crystal graph
            y, z = self.edge_update(lg, y, z)

        return v, x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""
    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = args.hidden_features
        self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8, bins=args.edge_input_features),
            MLPLayer(args.edge_input_features, args.embedding_features),
            MLPLayer(args.embedding_features, args.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=args.triplet_input_features),
            MLPLayer(args.triplet_input_features, args.embedding_features),
            MLPLayer(args.embedding_features, args.hidden_features),
        )
        self.module_layers = nn.ModuleList([EDiEGGCConv(args, True) for idx in range(args.alignn_layers)]
                                           + [EDiEGGCConv(args, False) for idx in range(args.gcn_layers-2)])

    def forward(self, g, lg):
        g = g.local_var()
        lg = lg.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        y = self.edge_embedding(g.edata['d'])

        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        # initial vector features
        v = torch.zeros_like(g.edata['r']).view(-1, 3, 1).expand(-1, -1, self.hidden_features)

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            v, x, y, z = module(g, lg, v, x, y, z)

        return v, x, y, z


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.target = args.target

        self.module_layers = nn.ModuleList([EDiEGGCConv(args, False) for idx in range(2)])
        self.in_readout = nn.Sequential(
            nn.Linear(args.hidden_features*2, args.hidden_features),
            nn.SiLU(),
        )
        self.eq_readout = nn.Sequential(
            nn.Linear(args.hidden_features*2, args.hidden_features),
            nn.SiLU(),
        )
        if args.task == 'regression':
            self.readout = nn.Linear(args.hidden_features, 1)
        elif args.task == 'classification':
            self.readout = nn.Linear(args.hidden_features, len(args.label_list))
        self.sumpooling = SumPooling()
        self.avgpooling = AvgPooling()
        # self.readout = nn.Linear(args.hidden_features, args.output_features)
        # self.pooling = AvgPooling()  # SumPooling()
        
        # self.q = nn.Sequential(
        #     MLPLayer(args.hidden_features, args.hidden_features),
        #     nn.Linear(args.hidden_features, 1),
        # )

    def forward(self, g, lg, v, x, y, z):
        g = g.local_var()

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            v, x, y, z = module(g, lg, v, x, y, z)

        g.edata['v'] = v
        g.update_all(fn.copy_e('v', 'v'), fn.sum('v', 'v'))
        v = g.ndata.pop('v')
        # final node features: average, sum pooling
        in_sum = self.sumpooling(g, x)
        in_avg = self.avgpooling(g, x)
        eq_sum = self.sumpooling(g, v)  # (batch, 3, hidden)
        in_out = self.in_readout(torch.cat([in_sum, in_avg], dim=-1))  # (batch, hidden)
        out_sum = in_out + torch.linalg.norm(eq_sum, dim=1)  # (batch, hidden)
        out = self.readout(out_sum)  # (batch, output)

        return torch.squeeze(out)


class EDiEGGC(nn.Module):

    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.target = args.target

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, g, lg):
        g = g.local_var()
        lg = lg.local_var()

        v, x, y, z = self.encoder(g, lg)
        out = self.decoder(g, lg, v, x, y, z)

        return out, v, x, y, z

