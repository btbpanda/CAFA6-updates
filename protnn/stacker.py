import torch
from torch import nn

try:
    from protlib.metric import get_depths
except ImportError:
    get_depths = None

def get_dag_dense(G, direction='all', self_loop=True):
    dst, src = [], []

    for i, node in enumerate(G.terms_list):
        if self_loop:
            dst.append(i)
            src.append(i)

        if direction in ['all', 'fwd']:
            for j in node['adj']:
                dst.append(i)
                src.append(j)

        if direction in ['all', 'bwd']:
            for j in node['children']:
                dst.append(i)
                src.append(j)

    dst, src = torch.LongTensor(dst), torch.LongTensor(src)

    return dst, src

class GCNLayer(nn.Module):

    def __init__(self, in_features):
        super().__init__()

        self.w0 = nn.Linear(in_features * 2, in_features)
        self.act0 = nn.ReLU()
        self.w1 = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x0, dst, src):
        x = torch.cat([
            x0.index_reduce(1, dst, x0[:, src], reduce='mean', include_self=False),
            x0.index_reduce(1, dst, x0[:, src], reduce='amax', include_self=False)
        ], dim=2)

        x = self.w0(x)
        x = self.act0(x)
        x = self.dropout(x)
        x = self.w1(x)

        x = x0 + x

        return x


class GCNStacker(nn.Module):

    def __init__(self, in_models, in_goa, graph, hidden_size=16, n_layers=8, embed_size=16):

        super().__init__()

        for direction in ['all', 'fwd', 'bwd']:
            dst, src = get_dag_dense(graph, direction=direction, self_loop=False)
            self.register_buffer(f'{direction}_dst', dst, )
            self.register_buffer(f'{direction}_src', src, )

        self.in_models = in_models
        self.in_goa = in_goa
        # self.in_features = in_models * 4 + in_goa

        self.nout = graph.idxs
        self.n_layers = n_layers
        self.embed_size = embed_size
        # node embedding

        if embed_size > 0:
            self.node_embed = nn.Embedding(self.nout, self.embed_size)

        node_feats = hidden_size + embed_size

        self.input = nn.Linear(self.in_models * 4 + in_goa, hidden_size)
        self.act = nn.ReLU()
        self.bn0 = nn.LayerNorm(in_models * 4)
        # self.dropout = nn.Dropout1d(0.5)
        # self.dropout = nn.Dropout1d(0.5)

        self.gcn_alldir = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_alldir.append(GCNLayer(node_feats))

        self.gcn_fwd = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_fwd.append(GCNLayer(node_feats))

        self.gcn_bwd = nn.ModuleList()
        for i in range(self.n_layers):
            self.gcn_bwd.append(GCNLayer(node_feats))

        self.clf = nn.Linear(node_feats * 4, 1)
        # self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)

    def forward(self, batch):

        l = batch['x'].shape[0]

        x = self.bn0(batch['x'])
        goa = batch['goa']
        # goa = self.dropout(batch['goa'])
        # goa = batch['goa']
        x = torch.cat([x, goa], dim=2)
        x = self.input(x)
        x = self.act(x)

        if self.embed_size > 0:
            emb = self.node_embed(torch.tile(torch.arange(self.nout).cuda(), (l, 1)))
            x = torch.cat([x, emb], dim=2)

        # x = self.bn0(x)
        # n_samples * n_nodes * n_features

        x0 = x  # n_samples * n_features * n_nodes

        layers = [x0]

        for gcns, direction in zip(
                [self.gcn_alldir, self.gcn_bwd, self.gcn_fwd],
                ['all', 'bwd', 'fwd']
        ):
            x = x0
            for gcn in gcns:
                x = gcn(x, dst=getattr(self, f'{direction}_dst'), src=getattr(self, f'{direction}_src'))
            layers.append(x)

        x = torch.cat(layers, dim=2)  # n_samples * n_features * n_nodes

        x = self.clf(x)[..., 0]  # + self.bias  # n_samples * n_nodes * n_features
        return x


class PropLevel(nn.Module):

    def __init__(self, idxs, G):
        super().__init__()
        src, dst_lvl, dst = [], [], []

        for n, i in enumerate(idxs):
            adj = G.terms_list[i]['adj']
            src.extend(adj)
            dst_lvl.extend([n] * len(adj))
            dst.append(i)

        self.register_buffer('src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('dst_lvl', torch.tensor(dst_lvl, dtype=torch.long))
        self.register_buffer('dst', torch.tensor(dst, dtype=torch.long))

    def forward(self, p_cond, p_raw, gt=None):

        p_par = torch.empty((len(p_cond), len(self.dst)), dtype=p_cond.dtype, device=p_cond.device)
        p_src = p_raw[:, self.src]

        if gt is not None:
            p_src = torch.maximum(p_src, gt[:, self.src])

        p_par = 1 - p_par.index_reduce(1, self.dst_lvl, 1 - p_src, reduce='prod', include_self=False)

        p_raw = p_raw.clone()
        p_raw[:, self.dst] = p_par * p_cond[:, self.dst]

        return p_raw


class GraphProp(nn.Module):

    def __init__(self, G):
        super().__init__()

        D = get_depths(G)
        self.levels = nn.ModuleList()

        for i in range(len(D)):
            self.levels.append(
                PropLevel(D[i], G)
            )

    def forward(self, p_cond, gt=None):

        p_raw = torch.ones_like(p_cond)
        for level in self.levels:
            p_raw = level(p_cond, p_raw, gt)

        return p_raw

class GCNStackerCND(nn.Module):
    def __init__(self, in_models, in_goa, graph, hidden_size=16, n_layers=8, embed_size=16):
        super().__init__()

        self.stacker = GCNStacker(in_models, in_goa, graph, hidden_size, n_layers, embed_size)
        self.act = nn.Sigmoid()
        self.cond_prop = GraphProp(graph)

    def forward(self, batch):
        p_cond = self.stacker(batch)
        p_cond = self.act(p_cond)

        p_raw = self.cond_prop(p_cond)

        return p_cond, p_raw


class ComposedBCELoss(nn.Module):

    def __init__(self, cond_rate=0.3):

        super().__init__()
        self.cond_rate = cond_rate
        self.loss_cond = nn.BCELoss(reduction='none')
        self.loss_raw = nn.BCELoss(reduction='mean')

    def forward(self, p_cond, p_raw, gt):

        cond_mask = ~torch.isnan(gt)
        gt = torch.where(cond_mask, gt, 0)
        cond_mask = cond_mask.type(p_cond.dtype)

        loss_cond = self.loss_cond(p_cond, gt)
        loss_cond = (loss_cond * cond_mask).sum() / cond_mask.sum()
        loss_raw = self.loss_raw(p_raw, gt)

        # print(loss_cond, loss_raw)

        return loss_cond * self.cond_rate + loss_raw


