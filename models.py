# -*- coding: utf-8 -*-
"""
BenchPCNP 论文中的 9 个基线模型（论文：所有基线均为 2 层卷积 + 2 层 MLP）。

普通图（BenchPCNP-General，团展开边）：
    GCN, ResGCN, JKNet, APPNP, GraphSAGE, GAT
超图（BenchPCNP-Hyper，每个网络一条超边）：
    HyperGCN（Yadati et al., NeurIPS 2019，带 mediator 的超图拉普拉斯近似）
    HyperResGCN（HyperGCN 卷积 + 残差）
    HyperJKNet（HyperGCN 卷积 + Jumping Knowledge）
额外提供（论文未列出，便于对比）：HGNN（Feng et al., AAAI 2019 的超图卷积），MLP（不使用结构）

所有模型 forward(data) -> logits [N, C]，全图（full-batch）训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPProp
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def mlp_head(in_dim, hidden, out_dim, dropout):
    """2 层 MLP 分类头。"""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, out_dim))


# ============================================================================ 普通图模型

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, **kw):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden), GCNConv(hidden, hidden)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, ei))
        return self.head(F.dropout(x, self.dropout, self.training))


class ResGCN(nn.Module):
    """GCN + 残差连接：先线性投影到隐层，再 x = x + GCNConv(x)。"""

    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, **kw):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden), GCNConv(hidden, hidden)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.inp(F.dropout(data.x, self.dropout, self.training)))
        for conv, norm in zip(self.convs, self.norms):
            h = conv(F.dropout(x, self.dropout, self.training), data.edge_index)
            x = F.relu(norm(x + h))
        return self.head(F.dropout(x, self.dropout, self.training))


class JKNet(nn.Module):
    """Jumping Knowledge（Xu et al., ICML 2018），默认按原文 concat 聚合各层表示。"""

    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, jk="cat", **kw):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden), GCNConv(hidden, hidden)])
        self.jk = jk
        out = hidden * 2 if jk == "cat" else hidden
        self.head = mlp_head(out, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        hs = []
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, ei))
            hs.append(x)
        x = torch.cat(hs, -1) if self.jk == "cat" else torch.stack(hs, 0).max(0)[0]
        return self.head(F.dropout(x, self.dropout, self.training))


class APPNP(nn.Module):
    """Predict then Propagate：2 层 MLP 预测 + 个性化 PageRank 传播（原文默认 K=10, alpha=0.1）。"""

    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, K=10, alpha=0.1, **kw):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.prop = APPNPProp(K=K, alpha=alpha)
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = F.dropout(data.x, self.dropout, self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = self.lin2(x)
        x = F.relu(self.prop(x, data.edge_index))
        return self.head(F.dropout(x, self.dropout, self.training))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, aggr="mean", **kw):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden, aggr=aggr), SAGEConv(hidden, hidden, aggr=aggr)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, ei))
        return self.head(F.dropout(x, self.dropout, self.training))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, heads=8, **kw):
        super().__init__()
        per_head = max(hidden // heads, 1)
        self.convs = nn.ModuleList([
            GATConv(in_dim, per_head, heads=heads, dropout=dropout),
            GATConv(per_head * heads, per_head, heads=heads, dropout=dropout),
        ])
        self.head = mlp_head(per_head * heads, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.elu(conv(x, ei))
        return self.head(F.dropout(x, self.dropout, self.training))


class MLP(nn.Module):
    """不使用任何连接关系的参照模型。"""

    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, **kw):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        return self.head(F.dropout(self.body(F.dropout(data.x, self.dropout, self.training)),
                                   self.dropout, self.training))


# ============================================================================ 超图工具

def sym_norm_propagate(x, row, col, w, n):
    """out = D^-1/2 (A + I) D^-1/2 x，A 由 (row, col, w) 给出（要求已对称）。"""
    loop = torch.arange(n, device=x.device)
    row = torch.cat([row, loop])
    col = torch.cat([col, loop])
    w = torch.cat([w, torch.ones(n, device=x.device, dtype=w.dtype)])
    deg = torch.zeros(n, device=x.device, dtype=w.dtype).index_add_(0, row, w)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0
    norm = dinv[row] * w * dinv[col]
    return torch.zeros_like(x).index_add_(0, row, x[col] * norm.unsqueeze(-1))


@torch.no_grad()
def hypergcn_structure(signal, he_node, he_id, num_he, mediators=True):
    """
    HyperGCN 的超图拉普拉斯近似（与官方实现 hypergcn/model/utils.py::laplacian 等价，向量化版本）：
      对每条超边 e，把节点信号投影到随机方向 rv 上，取投影最大 S_e、最小 I_e 两个节点；
      mediators=True ：连 S_e-I_e，并把其余节点都分别连到 S_e 和 I_e，权重均为 1/(2|e|-3)
      mediators=False：只连 S_e-I_e，权重 1/|e|
    返回对称的 (row, col, weight)，重复边权重累加。
    """
    dev = signal.device
    L = he_node.numel()
    rv = torch.rand(signal.size(1), device=dev)
    p = signal[he_node] @ rv                                    # [L]
    pos = torch.arange(L, device=dev)
    big = torch.full((num_he,), L, device=dev, dtype=torch.long)

    pmax = torch.full((num_he,), -float("inf"), device=dev).scatter_reduce(0, he_id, p, "amax")
    cand = torch.where(p == pmax[he_id], pos, torch.full_like(pos, L))
    s_pos = big.scatter_reduce(0, he_id, cand, "amin")

    pmin = torch.full((num_he,), float("inf"), device=dev).scatter_reduce(0, he_id, p, "amin")
    not_s = pos != s_pos[he_id]
    cand = torch.where((p == pmin[he_id]) & not_s, pos, torch.full_like(pos, L))
    i_pos = big.scatter_reduce(0, he_id, cand, "amin")
    # 所有投影相同（例如特征完全一样）时，取第一个非 S_e 的节点作 I_e
    fallback = big.scatter_reduce(0, he_id, torch.where(not_s, pos, torch.full_like(pos, L)), "amin")
    i_pos = torch.where(i_pos >= L, fallback, i_pos)

    size = torch.zeros(num_he, device=dev).index_add_(0, he_id, torch.ones(L, device=dev))
    S, I = he_node[s_pos], he_node[i_pos]
    if mediators:
        w_e = 1.0 / (2 * size - 3)
        rows, cols, ws = [S, I], [I, S], [w_e, w_e]
        med = (pos != s_pos[he_id]) & (pos != i_pos[he_id])
        k, e = he_node[med], he_id[med]
        wk = w_e[e]
        rows += [S[e], k, I[e], k]
        cols += [k, S[e], k, I[e]]
        ws += [wk, wk, wk, wk]
    else:
        w_e = 1.0 / size
        rows, cols, ws = [S, I], [I, S], [w_e, w_e]
    row, col, w = torch.cat(rows), torch.cat(cols), torch.cat(ws)
    n = int(signal.size(0))
    key = row * n + col
    uniq, inv = torch.unique(key, return_inverse=True)
    w = torch.zeros(uniq.numel(), device=dev).index_add_(0, inv, w)
    return uniq // n, uniq % n, w


class HyperGCNConv(nn.Module):
    """HyperGCN 卷积层：H' = Â(H W)，Â 每次前向按 HW 重新计算（fast=True 时只用输入特征算一次）。"""

    def __init__(self, in_dim, out_dim, mediators=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.mediators = mediators

    def forward(self, x, data, structure=None):
        hw = self.lin(x)
        if structure is None:
            structure = hypergcn_structure(hw.detach(), data.he_node, data.he_id,
                                           data.num_hyperedges, self.mediators)
        row, col, w = structure
        return sym_norm_propagate(hw, row, col, w, x.size(0))


class _HyperBase(nn.Module):
    def __init__(self, fast=False):
        super().__init__()
        self.fast = fast
        self._fast_struct = None

    def fast_structure(self, data):
        # FastHyperGCN：用输入特征算一次结构并缓存
        if self._fast_struct is None:
            self._fast_struct = hypergcn_structure(data.x, data.he_node, data.he_id,
                                                   data.num_hyperedges, True)
        return self._fast_struct


class HyperGCN(_HyperBase):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, mediators=True, fast=False, **kw):
        super().__init__(fast)
        self.convs = nn.ModuleList([HyperGCNConv(in_dim, hidden, mediators),
                                    HyperGCNConv(hidden, hidden, mediators)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        st = self.fast_structure(data) if self.fast else None
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, data, st))
        return self.head(F.dropout(x, self.dropout, self.training))


class HyperResGCN(_HyperBase):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, mediators=True, fast=False, **kw):
        super().__init__(fast)
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([HyperGCNConv(hidden, hidden, mediators),
                                    HyperGCNConv(hidden, hidden, mediators)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.inp(F.dropout(data.x, self.dropout, self.training)))
        st = self.fast_structure(data) if self.fast else None
        for conv, norm in zip(self.convs, self.norms):
            h = conv(F.dropout(x, self.dropout, self.training), data, st)
            x = F.relu(norm(x + h))
        return self.head(F.dropout(x, self.dropout, self.training))


class HyperJKNet(_HyperBase):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, mediators=True, fast=False, jk="cat", **kw):
        super().__init__(fast)
        self.convs = nn.ModuleList([HyperGCNConv(in_dim, hidden, mediators),
                                    HyperGCNConv(hidden, hidden, mediators)])
        self.jk = jk
        out = hidden * 2 if jk == "cat" else hidden
        self.head = mlp_head(out, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        st = self.fast_structure(data) if self.fast else None
        hs = []
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, data, st))
            hs.append(x)
        x = torch.cat(hs, -1) if self.jk == "cat" else torch.stack(hs, 0).max(0)[0]
        return self.head(F.dropout(x, self.dropout, self.training))


class HGNNConv(nn.Module):
    """HGNN 超图卷积：X' = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Θ（超边权重 W=I）。"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, data):
        hn, he, m, n = data.he_node, data.he_id, data.num_hyperedges, x.size(0)
        x = self.lin(x)
        ones = torch.ones(hn.numel(), device=x.device)
        dv = torch.zeros(n, device=x.device).index_add_(0, hn, ones).clamp(min=1)
        de = torch.zeros(m, device=x.device).index_add_(0, he, ones).clamp(min=1)
        xs = x * dv.pow(-0.5).unsqueeze(-1)
        e = torch.zeros(m, x.size(1), device=x.device).index_add_(0, he, xs[hn]) / de.unsqueeze(-1)
        out = torch.zeros_like(x).index_add_(0, hn, e[he])
        return out * dv.pow(-0.5).unsqueeze(-1)


class HGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.5, **kw):
        super().__init__()
        self.convs = nn.ModuleList([HGNNConv(in_dim, hidden), HGNNConv(hidden, hidden)])
        self.head = mlp_head(hidden, hidden, n_cls, dropout)
        self.dropout = dropout

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.dropout(x, self.dropout, self.training)
            x = F.relu(conv(x, data))
        return self.head(F.dropout(x, self.dropout, self.training))


# ============================================================================ 注册表

MODELS = {
    "GCN": (GCN, "graph"),
    "ResGCN": (ResGCN, "graph"),
    "JKNet": (JKNet, "graph"),
    "APPNP": (APPNP, "graph"),
    "GraphSAGE": (GraphSAGE, "graph"),
    "GAT": (GAT, "graph"),
    "HyperGCN": (HyperGCN, "hyper"),
    "HyperJKNet": (HyperJKNet, "hyper"),
    "HyperResGCN": (HyperResGCN, "hyper"),
    # 以下为论文之外的参照
    "HGNN": (HGNN, "hyper"),
    "MLP": (MLP, "graph"),
}

PAPER_MODELS = ["GCN", "ResGCN", "JKNet", "APPNP", "GraphSAGE", "GAT",
                "HyperGCN", "HyperJKNet", "HyperResGCN"]


def build_model(name, in_dim, n_cls, **kw):
    cls, _ = MODELS[name]
    return cls(in_dim, n_cls=n_cls, **kw)


def model_format(name):
    return MODELS[name][1]
