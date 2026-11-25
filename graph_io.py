# -*- coding: utf-8 -*-
"""单张网表图 ↔ NetlistData，供推理使用。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from data import NetlistData, build_features, vocab_hit_flags, _label_maps
from data import DEFAULT_ROOT


def load_vocab(root: str = DEFAULT_ROOT, fmt: str = "hyper", pack_dir: Optional[str | Path] = None):
    """
    加载 cls / footprint / 模块标签词表。
    优先使用 inference_pack/vocab_{fmt}.json（服务器无需再带 BenchPCNP_src）。
    """
    if pack_dir is not None:
        p = Path(pack_dir) / f"vocab_{fmt}.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                tab = json.load(f)
            return {
                "label_names": tab["label_names"],
                "labels_map": tab["labels_map"],
                "cls_map": tab["cls_map"],
                "footprints_map": tab["footprints_map"],
                "num_classes": tab["num_classes"],
            }
    label_names, labels_map, cls_map, fp_map = _label_maps(root, fmt)
    return {
        "label_names": label_names,
        "labels_map": labels_map,
        "cls_map": cls_map,
        "footprints_map": fp_map,
        "num_classes": len(label_names),
    }


def _clique_edges(members: List[str], nid: Dict[str, int]) -> Tuple[List[int], List[int]]:
    idx = [nid[r] for r in members]
    src, dst = [], []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            src += [idx[i], idx[j]]
            dst += [idx[j], idx[i]]
    return src, dst


def is_place_xfer(obj: Dict[str, Any]) -> bool:
    """是否为 placeXfer 布局 JSON（components + symbol/device），而非已转换的 graph record。"""
    return bool(obj.get("components")) and "node_info" not in obj


def place_xfer_to_record(
    xfer: Dict[str, Any],
    graph_id: Optional[str] = None,
    hint_y_name: Optional[Dict[str, str]] = None,
    description: str = "",
) -> Dict[str, Any]:
    """
    将 placeXfer JSON 转为推理 record（不做词表对齐）：
      node_info[ref] = [symbol, pin_n, device]  即 [footprint, 引脚数, devicetype]
      超边：同一 net 上 ≥2 个器件
    """
    main = xfer.get("mainComp") or "CHIP"
    gid = graph_id or main
    comps = xfer.get("components") or []
    nodes: List[str] = []
    node_info: Dict[str, List[Any]] = {}
    net_refs: Dict[str, set] = {}

    for c in comps:
        ref = c["ref"]
        nodes.append(ref)
        pins = c.get("pins") or []
        pin_n = int(c.get("pinCount") or len(pins) or 0)
        # 原味：封装 = symbol，类别 = device 字符串
        node_info[ref] = [str(c.get("symbol") or ""), pin_n, str(c.get("device") or "")]
        for p in pins:
            net = (p.get("net") or "").strip()
            if not net:
                continue
            net_refs.setdefault(net, set()).add(ref)

    hyperedges = [sorted(refs) for refs in net_refs.values() if len(refs) >= 2]
    desc = description or (
        f"placeXfer main={main} units={xfer.get('units', '?')}; "
        f"footprint=symbol, cls=device（未对齐官方词表）"
    )
    return {
        "id": gid,
        "board_id": gid,
        "source": "placeXfer",
        "description": desc,
        "nodes": nodes,
        "node_info": node_info,
        "hyperedges": hyperedges,
        "hint_y_name": hint_y_name or {},
        "eval_mask": [True] * len(nodes),
        "place_xfer_meta": {
            "stamp": xfer.get("stamp"),
            "design": xfer.get("design"),
            "mainComp": main,
        },
    }


def normalize_graph_record(obj: Dict[str, Any], path: Optional[Path] = None) -> Dict[str, Any]:
    """若为 placeXfer 则现场转换，否则原样返回 graph record。"""
    if is_place_xfer(obj):
        gid = Path(path).stem if path else obj.get("mainComp", "custom")
        return place_xfer_to_record(obj)
    return obj


def record_to_netlist(
    record: Dict[str, Any],
    vocab: Dict[str, Any],
    feature_mode: str = "111",
    oov_zero: Optional[bool] = None,
    fmt: str = "hyper",
) -> NetlistData:
    """record：nodes, node_info{ref:[footprint,pin_n,cls]}, hyperedges, y_true, eval_mask, oov_zero"""
    cls_map, fp_map = vocab["cls_map"], vocab["footprints_map"]
    labels_map = vocab["labels_map"]
    label_names = vocab["label_names"]
    nodes = list(record["nodes"])
    node_info = record["node_info"]
    nid = {n: i for i, n in enumerate(nodes)}

    vf = {n: vocab_hit_flags(node_info[n], cls_map, fp_map) for n in nodes}
    if oov_zero is None:
        use_oov = bool(record.get("oov_zero")) or any(
            not vf[n]["cls"] or not vf[n]["footprint"] for n in nodes
        )
    else:
        use_oov = bool(oov_zero)

    xs = [
        build_features(node_info[n], cls_map, fp_map, feature_mode, oov_zero=use_oov)
        for n in nodes
    ]
    y_true = record.get("y_true") or record.get("y_name") or {}

    def _label_id(ref: str) -> int:
        if ref not in y_true:
            return -1
        v = y_true[ref]
        if isinstance(v, int):
            return v if 0 <= v < len(label_names) else -1
        return labels_map.get(v, -1)

    ys = [_label_id(n) for n in nodes]

    src, dst = [], []
    he_node, he_id = [], []
    n_he = 0
    for hedge in record.get("hyperedges", []):
        mem = sorted({n for n in hedge if n in nid})
        if len(mem) < 2:
            continue
        s, d = _clique_edges(mem, nid)
        src += s
        dst += d
        he_node += [nid[n] for n in mem]
        he_id += [n_he] * len(mem)
        n_he += 1

    if not src and record.get("edges"):
        e0, e1 = record["edges"]
        for a, b in zip(e0, e1):
            if a in nid and b in nid and a != b:
                src.append(nid[a])
                dst.append(nid[b])

    x = torch.tensor(np.asarray(xs, dtype=np.float32))
    y = torch.tensor(ys, dtype=torch.long)
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    if ei.numel():
        ei = torch.unique(ei, dim=1)

    eval_mask = record.get("eval_mask")
    test_mask = torch.tensor(eval_mask, dtype=torch.bool) if eval_mask is not None else None

    data = NetlistData(
        fmt=fmt,
        x=x,
        y=y,
        edge_index=ei,
        he_node=torch.tensor(he_node, dtype=torch.long),
        he_id=torch.tensor(he_id, dtype=torch.long),
        num_hyperedges=n_he,
        keys=[(record.get("board_id", record.get("id", "custom")), n) for n in nodes],
        graph_id=torch.zeros(len(nodes), dtype=torch.long),
        num_classes=len(label_names),
        label_names=label_names,
        test_mask=test_mask,
    )
    cls_hit = sum(1 for n in nodes if vf[n]["cls"])
    fp_hit = sum(1 for n in nodes if vf[n]["footprint"])
    data.info = {
        "feature_mode": feature_mode,
        "oov_zero": use_oov,
        "graph_id": record.get("id"),
        "hint_y_name": record.get("hint_y_name", {}),
        "description": record.get("description", ""),
        "vocab_flags": vf,
        "vocab_hit_summary": {
            "nodes": len(nodes),
            "cls_hit": cls_hit,
            "footprint_hit": fp_hit,
        },
    }
    return data


def load_graph_record(path: Union[str, Path], vocab: Optional[Dict] = None, **kw) -> NetlistData:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    record = normalize_graph_record(raw, path)
    if vocab is None:
        vocab = load_vocab()
    return record_to_netlist(record, vocab, **kw)