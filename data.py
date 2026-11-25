# -*- coding: utf-8 -*-
"""
BenchPCNP 数据加载（dataloader）。

直接读取官方仓库 Data/Graphs/GraphN/graph.json 与 Data/HyperGraphs/GraphN/hypergraph.json，
按论文方法构造节点特征 r = [r1 r2 r3]：
    r1 = 器件类别 one-hot，r2 = 封装 one-hot，r3 = 引脚数二进制编码（官方代码为 8 位）
图结构：
    General：网络两两连边（团展开），即 graph.json 里的 edges
    Hyper  ：每个网络一条超边，即 hypergraph.json 里的 hyperedges
数据划分：
    official : 复用官方 MergeGraphs / MergeHyperGraphs 下 .pt 文件里的 train/val/test 掩码
               （论文 Table 5 的超图测试集 715 个节点与之完全一致）
    random   : 按 6:2:2（可用 train_rate 调整）随机划分，两种图格式按 (板子, 位号) 共用同一划分
    stratified: 按类别分层的 6:2:2 划分
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "BenchPCNP_src", "Data")

OFFICIAL_PT = {
    "graph": os.path.join("MergeGraphs", "graph_feature_mode_111_train_rate_0.6.pt"),
    "hyper": os.path.join("MergeHyperGraphs", "hyper_feature_mode_111_train_rate_0.6.pt"),
}


@dataclass
class NetlistData:
    """合并后的整图数据（所有板子拼成一张大图，板子之间没有边）。"""
    fmt: str                       # "graph" 或 "hyper"
    x: torch.Tensor                # [N, F] 节点特征
    y: torch.Tensor                # [N] 模块类别
    edge_index: torch.Tensor       # [2, E] 团展开后的无向边（双向，已去重、去自环）
    he_node: torch.Tensor          # [L] 超边关联：节点下标
    he_id: torch.Tensor            # [L] 超边关联：超边下标
    num_hyperedges: int
    keys: List[Tuple[str, str]]    # 每个节点的 (板子目录名, 位号)
    graph_id: torch.Tensor         # [N] 节点所属板子编号
    num_classes: int
    label_names: List[str]
    train_mask: Optional[torch.Tensor] = None
    val_mask: Optional[torch.Tensor] = None
    test_mask: Optional[torch.Tensor] = None
    info: Dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    def to(self, device):
        for name in ["x", "y", "edge_index", "he_node", "he_id", "graph_id",
                     "train_mask", "val_mask", "test_mask"]:
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, v.to(device))
        return self


def one_hot_encoding(label: int, length: int) -> List[int]:
    return [1 if i == label else 0 for i in range(length)]


def binary_encoding(label: int, length: int = 8) -> List[int]:
    """与官方 GraphConstruction.binary_encoding 完全一致（高位在前，超出长度时截断低位以外部分）。"""
    rt = [0] * length
    b_rt = bin(label)[2:]
    for i in list(b_rt)[::-1]:
        length -= 1
        if i == "1":
            rt[length] = 1
    return rt


def _load_json(path: str):
    # 官方 JSON 为 GBK 编码（中文标签），Linux 下也必须显式指定
    with open(path, "r", encoding="gbk") as f:
        return json.load(f)


def _graph_dirs(sub_root: str) -> List[str]:
    # 官方 read_graph 用 os.listdir 的顺序（Windows 下即字典序），这里固定为 sorted 保证跨平台一致
    return sorted(d for d in os.listdir(sub_root) if not d.endswith(".json"))


def _label_maps(root: str, fmt: str):
    if fmt == "graph":
        tab = _load_json(os.path.join(root, "Graphs", "Graph_labels.json"))
        return tab["labels"], tab["labels map"], tab["cls map"], tab["footprints map"]
    tab = _load_json(os.path.join(root, "HyperGraphs", "Graph_labels.json"))
    return tab["labels"], tab["labels_map"], tab["cls_map"], tab["footprints_map"]


def vocab_hit_flags(info, cls_map, fp_map) -> Dict[str, bool]:
    """node_info 三元组 [footprint, pin_n, cls] 是否在官方词表内。"""
    footprint, _, cls = info[0], info[1], info[-1]
    return {"footprint": footprint in fp_map, "cls": cls in cls_map}


def build_features(
    info,
    cls_map,
    fp_map,
    feature_mode: str,
    *,
    oov_zero: bool = False,
) -> List[int]:
    """
    feature_mode 每一位对应 [类别, 封装, 引脚]：'1' 真实特征，'2' 全 1，'0' 全 0（与官方一致）。

    oov_zero（推理用）：
        封装 / 类别不在官方词表时，对应 one-hot 置为全 0（论文未规定 OOV 策略，便于先看拓扑效果）；
        训练集加载默认 oov_zero=False，词表外会 KeyError 以便及早发现脏数据。
    """
    footprint, pin_n, cls = info[0], info[1], info[-1]
    pin_n = int(pin_n)

    if cls in cls_map:
        cls_oh = one_hot_encoding(cls_map[cls], len(cls_map))
    elif oov_zero:
        cls_oh = [0] * len(cls_map)
    else:
        raise KeyError(f"器件类别不在词表中: cls={cls!r}（推理请设 oov_zero=True）")

    if footprint in fp_map:
        fp_oh = one_hot_encoding(fp_map[footprint], len(fp_map))
    elif oov_zero:
        fp_oh = [0] * len(fp_map)
    else:
        raise KeyError(f"封装不在词表中: footprint={footprint!r}（推理请设 oov_zero=True）")

    options = [cls_oh, fp_oh, binary_encoding(pin_n)]
    feat: List[int] = []
    for i, flag in enumerate(feature_mode):
        if flag == "1":
            feat += options[i]
        elif flag == "2":
            feat += [1] * len(options[i])
        else:
            feat += [0] * len(options[i])
    return feat


def load_benchpcnp(fmt: str = "hyper", root: str = DEFAULT_ROOT, feature_mode: str = "111",
                   verbose: bool = True) -> NetlistData:
    """读取全部 50 块板子并合并成一张大图；同时构造团展开边与超边，两类模型都能用。"""
    assert fmt in ("graph", "hyper")
    label_names, labels_map, cls_map, fp_map = _label_maps(root, fmt)
    sub = os.path.join(root, "Graphs" if fmt == "graph" else "HyperGraphs")
    gsub = os.path.join(root, "Graphs")
    hsub = os.path.join(root, "HyperGraphs")

    xs, ys, keys, gids = [], [], [], []
    src, dst = [], []
    he_node, he_id = [], []
    n_he = 0
    for gi, gdir in enumerate(_graph_dirs(sub)):
        g = _load_json(os.path.join(gsub, gdir, "graph.json"))
        h = _load_json(os.path.join(hsub, gdir, "hypergraph.json"))
        # 节点顺序跟随当前格式的官方 JSON，保证 official 掩码逐行对齐
        nodes = g["nodes"] if fmt == "graph" else h["nodes"]
        node_info = g["node info"] if fmt == "graph" else h["node_info"]
        y_dict = g["y"] if fmt == "graph" else h["y"]
        base = len(keys)
        nid = {n: base + i for i, n in enumerate(nodes)}
        label_of = {n: labels_map[k] for k in y_dict for n in y_dict[k]}
        for n in nodes:
            xs.append(build_features(node_info[n], cls_map, fp_map, feature_mode))
            ys.append(label_of.get(n, -1))
            keys.append((gdir, n))
            gids.append(gi)
        # 团展开边（graph.json 已是双向列表）
        for a, b in zip(g["edges"][0], g["edges"][1]):
            if a in nid and b in nid and a != b:
                src.append(nid[a])
                dst.append(nid[b])
        # 超边：同一网络内去重，只保留 ≥2 个器件的网络
        for hedge in h["hyperedges"]:
            members = sorted({nid[n] for n in hedge if n in nid})
            if len(members) < 2:
                continue
            he_node += members
            he_id += [n_he] * len(members)
            n_he += 1

    x = torch.tensor(np.asarray(xs, dtype=np.float32))
    y = torch.tensor(ys, dtype=torch.long)
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    ei = torch.unique(ei, dim=1)  # 去重并保证双向
    data = NetlistData(
        fmt=fmt, x=x, y=y, edge_index=ei,
        he_node=torch.tensor(he_node, dtype=torch.long),
        he_id=torch.tensor(he_id, dtype=torch.long),
        num_hyperedges=n_he, keys=keys,
        graph_id=torch.tensor(gids, dtype=torch.long),
        num_classes=len(label_names), label_names=label_names,
    )
    data.info = {"feature_mode": feature_mode, "num_graphs": len(set(gids))}
    if verbose:
        print(f"[data] fmt={fmt} 节点={data.num_nodes} 特征维={x.size(1)} 类别={data.num_classes} "
              f"无向边={ei.size(1) // 2} 超边={n_he} 未标注={int((y < 0).sum())}")
    return data


# ----------------------------------------------------------------------------- 划分

def _masks_from_index(n, tr, va, te):
    masks = []
    for idx in (tr, va, te):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(idx, dtype=torch.long)] = True
        masks.append(m)
    return masks


def split_random(data: NetlistData, train_rate=0.6, val_rate=0.2, seed=0, stratified=False):
    """6:2:2 随机划分；按 (板子, 位号) 排序后再打乱，使 graph/hyper 两种格式得到同一批测试节点。"""
    n = data.num_nodes
    order = sorted(range(n), key=lambda i: data.keys[i])
    rng = np.random.RandomState(seed)
    labeled = [i for i in order if int(data.y[i]) >= 0]
    tr, va, te = [], [], []
    if stratified:
        by_cls: Dict[int, List[int]] = {}
        for i in labeled:
            by_cls.setdefault(int(data.y[i]), []).append(i)
        for c in sorted(by_cls):
            idx = by_cls[c]
            rng.shuffle(idx)
            k_tr = int(round(len(idx) * train_rate))
            k_va = int(round(len(idx) * val_rate))
            tr += idx[:k_tr]
            va += idx[k_tr:k_tr + k_va]
            te += idx[k_tr + k_va:]
    else:
        idx = list(labeled)
        rng.shuffle(idx)
        k_tr = int(round(len(idx) * train_rate))
        k_va = int(round(len(idx) * val_rate))
        tr, va, te = idx[:k_tr], idx[k_tr:k_tr + k_va], idx[k_tr + k_va:]
    data.train_mask, data.val_mask, data.test_mask = _masks_from_index(n, tr, va, te)
    return data


def split_official(data: NetlistData, root: str = DEFAULT_ROOT, check: bool = True):
    """复用官方 .pt 的掩码；check=True 时逐行比对特征和标签，确认节点顺序一致。"""
    path = os.path.join(root, OFFICIAL_PT[data.fmt])
    off = torch.load(path, map_location="cpu", weights_only=False)  # 需要安装 torch_geometric
    n = data.num_nodes
    if check:
        # 官方 .pt 固定在 CPU；data 可能已在 GPU（多 trial 复用同一对象），比对前统一到 CPU
        ref_x, ref_y = off.x[:n].float(), off.y[:n].long()
        cur_x, cur_y = data.x.detach().cpu().float(), data.y.detach().cpu().long()
        if data.info.get("feature_mode") == "111":
            same_x = torch.equal(ref_x, cur_x)
        else:
            same_x = True  # 特征消融时特征本来就不同，只比对标签
        same_y = torch.equal(ref_y, cur_y)
        if not (same_x and same_y):
            raise RuntimeError(f"官方 .pt 与 JSON 重建的节点不一致 x={same_x} y={same_y}，"
                               f"请改用 --split random")
    data.train_mask = off.train_mask[:n].bool().clone()
    data.val_mask = off.val_mask[:n].bool().clone()
    data.test_mask = off.test_mask[:n].bool().clone()
    return data


def make_split(data: NetlistData, split: str, seed=0, train_rate=0.6, root=DEFAULT_ROOT):
    if split == "official":
        return split_official(data, root)
    if split == "random":
        return split_random(data, train_rate, (1 - train_rate) / 2, seed, stratified=False)
    if split == "stratified":
        return split_random(data, train_rate, (1 - train_rate) / 2, seed, stratified=True)
    raise ValueError(split)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for f in ("graph", "hyper"):
        d = load_benchpcnp(f)
        split_official(d)
        print(f"  official 划分 train/val/test = {int(d.train_mask.sum())}/"
              f"{int(d.val_mask.sum())}/{int(d.test_mask.sum())}（已与官方 .pt 逐行比对一致）")
        split_random(d, seed=0)
        print(f"  random 6:2:2 train/val/test = {int(d.train_mask.sum())}/"
              f"{int(d.val_mask.sum())}/{int(d.test_mask.sum())}")
