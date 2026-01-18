import numpy as np
import torch
from numba import jit
from torch.utils.data import Dataset, DataLoader

try:
    from protlib.metric import get_depths
    import cupy as cp
except ImportError:
    cp, get_depths = [None] * 2

@jit(nopython=True)
def propagate(arr, cnd_mask, col, n_mod, adj):
    for j in range(n_mod):

        if not cnd_mask[j, col]:
            continue

        acc_p = 1.0
        acc_1mp = 1.0

        for k in adj:
            acc_1mp *= 1 - arr[j * 4 + 3, k]
            acc_p *= arr[j * 4 + 2, k]

        arr[j * 4 + 3, col] *= 1 - acc_1mp
        arr[j * 4 + 2, col] *= acc_p

    return


class Propagator:

    def __init__(self, G, preds):

        self.G = G
        self.D = get_depths(G)
        cnd_mask = np.ones((len(preds), G.idxs), dtype=np.bool_)

        for n, pred in enumerate(preds):
            if pred.cond:
                continue
            cnd_mask[n, pred.ns_idx] = False

        self.adj = [np.asarray(x['adj']) for x in G.terms_list]

        self.cnd_mask = np.asarray(cnd_mask)
        self.n_mod = len(preds)

    def __call__(self, batch):

        arr = batch['x'].numpy()

        for i in range(len(self.D)):
            for k in self.D[i]:
                # for k in self.G.order:
                adj = self.adj[k]
                if len(adj) == 0:
                    continue

                propagate(arr, self.cnd_mask, k, self.n_mod, adj)

        return batch


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


class StackDataset(Dataset):

    def __init__(self, preds, G, goa_list, p_goa=1, targets=None, gt=None, p_gt=0.3):

        self.preds = preds
        self.G = G
        self.nout = len(G.terms_list)
        self.goa = [x.tolist() for x in goa_list]
        self.p_goa = p_goa

        self.gt = None
        if gt is not None:
            self.gt = gt.tolist()
        self.p_gt = p_gt

        self.targets = targets
        self.adj = [np.array(x['adj'], dtype=np.int64) for x in G.terms_list]
        self.prop = Propagator(G, preds)

    def __getitem__(self, index):

        batch = {}
        x = []

        for prediction in self.preds:
            x.append(
                torch.from_numpy(prediction[index])
            )

        batch['x'] = torch.cat(x, dim=0)
        self.prop(batch)

        x = batch['x'].swapaxes(0, 1)
        # apply logit to prob part only
        nout, _ = x.shape
        x = x.reshape((nout, -1, 4))
        x0, x = x[..., :1], x[..., 1:]

        x = torch.clamp(x, 1e-6, 1 - 1e-6)
        x = torch.log(x / (1 - x))
        batch['x'] = torch.cat([x0, x], dim=2).view(nout, -1)

        # add go annotations
        x_goa = []

        goa = np.zeros((len(self.goa), self.nout), dtype=np.float32)
        if np.random.rand() < self.p_goa:
            for n, ann in enumerate(self.goa):
                ann = ann[index]
                if len(ann) > 0:
                    goa[n, ann] = 1
        x_goa.append(torch.from_numpy(goa))

        batch['goa'] = torch.cat(x_goa, dim=0).swapaxes(0, 1)

        # self.prop(batch)

        if self.targets is not None:
            batch['y'] = torch.from_numpy(self.targets[index])

        gt = None
        if self.gt is not None:
            gt = torch.zeros(self.nout, dtype=torch.float32)
            ann = self.gt[index]
            gt[ann] = 1
        else:
            if np.random.rand() < self.p_gt:
                gt = batch['y'].nan_to_num(nan=0)

        if gt is not None:
            batch['gt'] = gt

        return batch

    def __len__(self, ):

        return len(self.preds[0])

    def __len__(self, ):

        return len(self.preds[0])
