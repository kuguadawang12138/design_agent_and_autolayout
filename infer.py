# -*- coding: utf-8 -*-
"""
BenchPCNP 推理：官方测试图 + U34 自定义图。

服务器最少文件：
  infer.py, train.py(仅 build_model 依赖 models/data), graph_io.py, data.py, models.py,
  inference_pack/（manifest、vocab_*.json、bench/*.json、custom/U34_Comps.json）

查看数据包（无需权重）：
  python infer.py --inspect --pack inference_pack

推理：
  python infer.py --checkpoint results/checkpoints/.../HyperResGCN_trial1_seed0.pt --device cuda
  python infer.py --checkpoint xxx.pt --graphs U44,U1,U34
  python infer.py --checkpoint xxx.pt --place_xfer ../data_layout/data/U44_Group_20261110_105339.json
  python infer.py --checkpoint xxx.pt --graphs U34 --refs Y2,C186

指标说明：
  prob / p1（CSV 列 top1_prob）= softmax 后对「当前预测的模块类」的概率，即 Top-1 置信度，不是 ACC。
  封装/cls：在官方 one-hot 词表内才编码，否则该段特征为 0（不强行改成词表内字符串）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import torch

from graph_io import load_graph_record, load_vocab
from models import build_model, model_format

HERE = Path(__file__).resolve().parent

U34_CRYSTAL_REFS = {"Y2", "C186", "C188", "R168"}
U1_CRYSTAL_REFS = {"X1", "X2"}
CUSTOM_HIGHLIGHT = {
    "U34": U34_CRYSTAL_REFS | {"U34"},
    "U44": {"U44"},
    "U1": U1_CRYSTAL_REFS | {"U1"},
}

# manifest 未登记时，按 id 在 custom/ 下猜文件名
_CUSTOM_GUESS = {
    "U34": "custom/U34_Comps.json",
    "U44": "custom/U44_Group.json",
    "U1": "custom/U1_Group.json",
}


def _resolve_custom_id(pack: Path, gid: str) -> dict | None:
    """在 inference_pack/custom 下按 id 找图，不依赖 manifest 是否更新。"""
    rel = _CUSTOM_GUESS.get(gid)
    if rel and (pack / rel).is_file():
        return {"id": gid, "path": rel}
    custom = pack / "custom"
    if not custom.is_dir():
        return None
    for p in sorted(custom.glob(f"{gid}*.json")):
        return {"id": gid, "path": f"custom/{p.name}"}
    return None


def _select_graphs(pack: Path, manifest: dict, graphs_arg: str, extra_xfer: List[dict]) -> List[dict]:
    graphs = list(manifest.get("graphs") or [])
    graphs = extra_xfer + graphs
    if graphs_arg == "bench":
        return [g for g in graphs if str(g.get("path", "")).startswith("bench/")]
    if graphs_arg == "custom":
        out = [g for g in graphs if str(g.get("path", "")).startswith("custom/")]
        if out:
            return out
        # 旧 manifest 无 custom 项：扫 custom 目录
        custom = pack / "custom"
        if custom.is_dir():
            return [
                {"id": p.stem.replace("_Group", "").replace("_Comps", ""), "path": f"custom/{p.name}"}
                for p in sorted(custom.glob("*.json"))
            ]
        return []
    if graphs_arg == "all":
        return graphs
    want = {x.strip() for x in graphs_arg.split(",") if x.strip()}
    picked = [g for g in graphs if g.get("id") in want]
    found = {g["id"] for g in picked}
    for gid in sorted(want - found):
        g = _resolve_custom_id(pack, gid)
        if g:
            picked.append(g)
            found.add(gid)
    missing = want - found
    if missing:
        ids = [g.get("id") for g in manifest.get("graphs", [])]
        custom_ids = [p.stem for p in (pack / "custom").glob("*.json")] if (pack / "custom").is_dir() else []
        raise SystemExit(
            f"未找到图 id={sorted(missing)}。\n"
            f"  manifest 中 id 共 {len(ids)} 个（是否缺 U44/U1：看 inference_pack/manifest.json 末尾）。\n"
            f"  custom/*.json 现有：{custom_ids or '无（请拷 inference_pack/custom）'}\n"
            f"  可先跑：python infer.py --inspect --pack {pack}"
        )
    return picked


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    data = data.to(device)
    logits = model(data)
    prob = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)
    return pred.cpu(), prob.cpu()


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt.get("model_name")
    if not name:
        raise SystemExit("checkpoint 缺少 model_name 字段")
    cfg = ckpt.get("train_config") or {}
    model = build_model(
        name,
        ckpt.get("in_dim"),
        ckpt.get("num_classes"),
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.2),
        heads=cfg.get("heads", 8),
        mediators=not cfg.get("no_mediators", False),
        fast=cfg.get("fast", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model, name, ckpt


def inspect_pack(pack: Path):
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺少 {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"[inspect] 任务={manifest.get('task')} 划分={manifest.get('split')} "
          f"feature_mode={manifest.get('feature_mode')}")
    print(f"[inspect] 官方测试节点={manifest.get('official_test_nodes')}")
    print(f"[inspect] 图数量={len(manifest.get('graphs', []))}")

    jsonl = pack / "bench" / "test_nodes.jsonl"
    if jsonl.is_file():
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_board = Counter(r["board"] for r in rows)
        by_cls = Counter(r["y_true_name"] for r in rows)
        print(f"\n--- test_nodes.jsonl 共 {len(rows)} 条 ---")
        print(f"按板（前 8）: {dict(by_board.most_common(8))}")
        print(f"按模块（前 12）: {dict(by_cls.most_common(12))}")
        crystal = [r for r in rows if r["y_true_name"] == "晶振"]
        if crystal:
            print(f"测试集中「晶振」标注 {len(crystal)} 个")

    from graph_io import load_vocab as _lv
    from data import vocab_hit_flags

    vocab = _lv(pack_dir=pack, fmt="hyper")
    custom_paths = {
        "U34": "custom/U34_Comps.json",
        "U44": "custom/U44_Group.json",
        "U1": "custom/U1_Group.json",
    }
    for cid, rel in custom_paths.items():
        p = pack / rel
        if not p.is_file():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        ni = rec.get("node_info", {})
        hits = [vocab_hit_flags(ni[r], vocab["cls_map"], vocab["footprints_map"]) for r in rec["nodes"]]
        n_cls = sum(h["cls"] for h in hits)
        n_fp = sum(h["footprint"] for h in hits)
        print(f"\n--- {cid} nodes={len(rec['nodes'])} hedges={len(rec.get('hyperedges', []))} "
              f"词表命中 cls={n_cls}/{len(hits)} fp={n_fp}/{len(hits)} ---")
        refs = sorted(
            U34_CRYSTAL_REFS if cid == "U34"
            else (U1_CRYSTAL_REFS | {"U1"} if cid == "U1" else [rec.get("mainComp", "U44")])
        )
        for ref in refs:
            if ref not in ni:
                continue
            info = ni[ref]
            hint = (rec.get("hint_y_name") or {}).get(ref, "-")
            vf = vocab_hit_flags(info, vocab["cls_map"], vocab["footprints_map"])
            print(f"    {ref}: hint={hint}  fp={info[0]!r} cls={info[2]!r} pins={info[1]} "
                  f"({'+cls' if vf['cls'] else '-cls'}{' +fp' if vf['footprint'] else ' -fp'})")


def run_graph(model, model_name, record_path, vocab, feature_mode, device, topk):
    record_path = Path(record_path)
    with open(record_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    data = load_graph_record(
        record_path, vocab, feature_mode=feature_mode, fmt=model_format(model_name),
    )
    pred, prob = predict(model, data, device)
    names = data.label_names
    rows = []
    eval_mask = record.get("eval_mask", [True] * len(record["nodes"]))
    for i, ref in enumerate(record["nodes"]):
        yid = int(data.y[i])
        pid = int(pred[i])
        topv, topi = torch.topk(prob[i], k=min(topk, len(names)))
        vf = (data.info.get("vocab_flags") or {}).get(ref, {})
        rows.append({
            "ref": ref,
            "pred_id": pid,
            "pred_name": names[pid],
            "y_true_id": yid,
            "y_true_name": names[yid] if yid >= 0 else "",
            "hint_name": (record.get("hint_y_name") or {}).get(ref, ""),
            "eval": bool(eval_mask[i]),
            "topk": [{"name": names[int(j)], "prob": float(topv[k])} for k, j in enumerate(topi)],
            "node_info": record["node_info"].get(ref),
            "cls_in_vocab": vf.get("cls"),
            "fp_in_vocab": vf.get("footprint"),
            "correct": yid >= 0 and pid == yid,
        })
    return rows, data, record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--pack", default=str(HERE / "inference_pack"))
    ap.add_argument("--feature_mode", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--graphs", default="all", help="all / bench / custom / Graph0,U34")
    ap.add_argument("--max_graphs", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--show_errors", action="store_true")
    ap.add_argument("--refs", default="")
    ap.add_argument(
        "--place_xfer",
        default="",
        help="逗号分隔的 placeXfer JSON 路径（现场转图，symbol/device 原味）",
    )
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pack = Path(args.pack)
    if args.inspect:
        inspect_pack(pack)
        return

    if not args.checkpoint:
        raise SystemExit("请指定 --checkpoint，或使用 --inspect")

    with open(pack / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    feature_mode = args.feature_mode or manifest.get("feature_mode", "111")

    dev = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, model_name, ckpt = load_checkpoint(Path(args.checkpoint), dev)
    model.to(dev)
    vocab = load_vocab(fmt=model_format(model_name), pack_dir=pack)

    extra_xfer: List[dict] = []
    if args.place_xfer.strip():
        for p in [x.strip() for x in args.place_xfer.split(",") if x.strip()]:
            pid = Path(p).stem.split("_Group")[0].split("_")[0]
            if not pid or pid.isdigit():
                pid = Path(p).stem
            extra_xfer.append({"id": pid, "path": p, "_abs": True})

    graphs = _select_graphs(pack, manifest, args.graphs, extra_xfer)
    if not graphs:
        raise SystemExit(
            f"没有可推理的图（graphs=0）。--graphs={args.graphs!r}\n"
            f"  常见原因：manifest 太旧未含 U44/U1，或未拷 inference_pack/custom/。\n"
            f"  请同步整包 inference_pack，或：python infer.py --inspect --pack {pack}"
        )
    if args.max_graphs:
        graphs = graphs[: args.max_graphs]

    out_dir = Path(args.out) / f"infer_{time.strftime('%Y%m%d_%H%M%S')}"
    per_dir = out_dir / "per_graph"
    per_dir.mkdir(parents=True, exist_ok=True)

    all_preds, n_eval, n_correct = [], 0, 0
    filter_refs = {r.strip() for r in args.refs.split(",") if r.strip()}

    print(f"[infer] model={model_name} device={dev} graphs={len(graphs)} feature_mode={feature_mode}")
    print(f"[infer] checkpoint seed={ckpt.get('seed')} metrics={ckpt.get('metrics')}")

    for g in graphs:
        gpath = Path(g["path"]) if g.get("_abs") else (pack / g["path"])
        if not gpath.is_file():
            raise SystemExit(f"图文件不存在：{gpath}（id={g.get('id')}）请检查 --pack 与 inference_pack/custom")
        rows, data, record = run_graph(model, model_name, gpath, vocab, feature_mode, dev, args.topk)
        all_preds.append({"graph": g["id"], "path": g["path"], "rows": rows})

        with open(per_dir / f"{g['id']}.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ref", "pred_name", "y_true_name", "hint_name", "eval", "correct",
                        "top1_prob", "cls", "footprint", "pin_n", "cls_in_vocab", "fp_in_vocab"])
            for r in rows:
                info = r["node_info"] or ["", 0, ""]
                w.writerow([r["ref"], r["pred_name"], r["y_true_name"], r["hint_name"], r["eval"],
                            r["correct"], r["topk"][0]["prob"] if r["topk"] else "",
                            info[2], info[0], info[1], r.get("cls_in_vocab"), r.get("fp_in_vocab")])

        te = [r for r in rows if r["eval"] and r["y_true_id"] >= 0]
        if te:
            n_eval += len(te)
            n_correct += sum(r["correct"] for r in te)
            acc = 100 * sum(r["correct"] for r in te) / len(te)
        else:
            acc = float("nan")

        print(f"\n=== {g['id']} === {record.get('description', '')[:70]}")
        vs = data.info.get("vocab_hit_summary") or {}
        if vs:
            print(f"  词表命中 cls {vs.get('cls_hit')}/{vs.get('nodes')}  "
                  f"footprint {vs.get('footprint_hit')}/{vs.get('nodes')}")
        hi = CUSTOM_HIGHLIGHT.get(g["id"], set())
        if hi:
            for r in rows:
                if r["ref"] in hi:
                    tk = r["topk"][0]["prob"] if r["topk"] else 0.0
                    print(f"  {r['ref']:6s} pred={r['pred_name']:12s} hint={r['hint_name'] or '-':8s} p1={tk:.3f}")

        if filter_refs or args.show_errors or g["id"] in CUSTOM_HIGHLIGHT:
            for r in rows:
                if filter_refs and r["ref"] not in filter_refs:
                    continue
                if not r["eval"]:
                    continue
                if args.show_errors and r["y_true_id"] >= 0 and r["correct"]:
                    continue
                if r["y_true_id"] < 0 and not r["hint_name"]:
                    continue
                tag = "OK" if r["correct"] else ("ERR" if r["y_true_id"] >= 0 else "HINT")
                yt = r["y_true_name"] or r["hint_name"] or "-"
                if filter_refs or tag != "OK" or g["id"] in CUSTOM_HIGHLIGHT:
                    print(f"  [{tag}] {r['ref']:8s} pred={r['pred_name']} true={yt}")

        if te:
            print(f"  → 本图测试 ACC={acc:.2f}% ({sum(r['correct'] for r in te)}/{len(te)})")

    summary = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "device": str(dev),
        "n_graphs": len(graphs),
        "eval_nodes": n_eval,
        "eval_correct": n_correct,
        "eval_acc": 100 * n_correct / n_eval if n_eval else None,
        "feature_mode": feature_mode,
        "ckpt_seed": ckpt.get("seed"),
    }
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": all_preds}, f, ensure_ascii=False, indent=2)
    if n_eval:
        print(f"\n[完成] 官方测试集合并 ACC={summary['eval_acc']:.2f}% ({n_correct}/{n_eval})")
    print(f"[完成] 输出目录 {out_dir}")


if __name__ == "__main__":
    main()