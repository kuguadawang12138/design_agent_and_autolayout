# -*- coding: utf-8 -*-
"""
BenchPCNP 基线训练 / 评估脚本（全图训练，节点分类）。

示例：
  # CPU 冒烟测试（每个模型 1 次实验、少量 epoch）
  python train.py --models all --trials 1 --epochs 5
  # 复现论文 Table 5（5 次随机实验，官方划分）
  python train.py --models all --trials 5 --device cuda
  # 真正的 6:2:2 随机划分
  python train.py --models HyperGCN --split random
  # 特征消融（论文 Fig.6）：Full=222，只用类别=100，只用封装=010，只用引脚=001
  python train.py --models HyperGCN --feature_mode 100
  # 不保存权重
  python train.py --models GCN --trials 1 --no_save_model

评价指标：ACC、宏平均 F1 / Precision / Recall（sklearn，zero_division=0），以验证集 ACC 选最优 epoch。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from data import DEFAULT_ROOT, load_benchpcnp, make_split
from losses import build_loss
from models import MODELS, PAPER_MODELS, build_model, model_format

# 论文 Table 5 / Table 6 的均值，用于对照
PAPER = {
    "GCN": (56.78, 45.70), "ResGCN": (64.34, 53.92), "JKNet": (56.41, 44.82),
    "APPNP": (56.63, 41.21), "GraphSAGE": (67.94, 54.60), "GAT": (66.59, 53.49),
    "HyperGCN": (76.49, 69.78), "HyperJKNet": (78.34, 72.32), "HyperResGCN": (79.07, 70.77),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    y_true, y_pred = y_true.cpu().numpy(), y_pred.cpu().numpy()
    return {
        "acc": 100 * accuracy_score(y_true, y_pred),
        "f1": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": 100 * precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": 100 * recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": 100 * f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    pred = model(data).argmax(-1)
    return metrics(data.y[mask], pred[mask])


def _run_config(args):
    """写入 checkpoint 的超参快照，便于日后按相同配置复现。"""
    keys = ("feature_mode", "split", "train_rate", "hidden", "dropout", "lr", "weight_decay",
            "heads", "loss", "label_smoothing", "no_mediators", "fast", "epochs", "patience")
    return {k: getattr(args, k) for k in keys}


def run_one(name, data, args, seed, trial_idx, split_seed, ckpt_dir, log=print):
    set_seed(seed)
    dev = data.x.device
    model = build_model(name, data.x.size(1), data.num_classes, hidden=args.hidden,
                        dropout=args.dropout, heads=args.heads, mediators=not args.no_mediators,
                        fast=args.fast).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    y_tr = data.y[data.train_mask]
    crit = build_loss(args.loss, y_tr, data.num_classes, args.label_smoothing).to(dev)

    best_val, best_state, best_ep, bad = -1.0, None, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(data)
        loss = crit(out[data.train_mask], y_tr)
        loss.backward()
        opt.step()
        if ep % args.eval_every and ep != args.epochs:
            continue
        val = evaluate(model, data, data.val_mask)
        score = val[args.select]
        if score > best_val:
            best_val, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += args.eval_every
        if args.verbose and (ep % args.log_every == 0 or ep == 1):
            log(f"    ep{ep:4d} loss={loss.item():.4f} val_acc={val['acc']:.2f} val_f1={val['f1']:.2f}")
        if bad >= args.patience:
            break
    train_time = time.time() - t0
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    test = evaluate(model, data, data.test_mask)
    test.update({
        "best_epoch": best_ep,
        "best_val": best_val,
        "time_s": train_time,
        "epochs_run": ep,
        "params": sum(p.numel() for p in model.parameters()),
        "seed": seed,
        "split_seed": split_seed,
        "trial": trial_idx,
    })

    if args.save_model and best_state is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{name}_trial{trial_idx + 1}_seed{seed}.pt")
        torch.save({
            "model_name": name,
            "seed": seed,
            "split_seed": split_seed,
            "trial": trial_idx,
            "state_dict": best_state,
            "best_epoch": best_ep,
            "num_classes": data.num_classes,
            "in_dim": data.x.size(1),
            "metrics": {k: test[k] for k in test if k not in ("seed", "split_seed", "trial")},
            "train_config": _run_config(args),
        }, ckpt_path)
        test["checkpoint"] = ckpt_path
        log(f"    权重已保存: {ckpt_path}")
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all", help="all / paper / extra / 逗号分隔的模型名")
    ap.add_argument("--data_root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default="official", choices=["official", "random", "stratified"])
    ap.add_argument("--split_seed", type=int, default=0, help="random/stratified 划分的随机种子")
    ap.add_argument("--resplit", action="store_true", help="每次实验都用不同划分（split_seed+trial）")
    ap.add_argument("--train_rate", type=float, default=0.6)
    ap.add_argument("--feature_mode", default="111")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="训练随机种子基值，第 t 次 trial 使用 seed+t")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--select", default="acc", choices=["acc", "f1"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--loss", default="ce", choices=["ce", "wce", "focal", "cb_focal"])
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--no_mediators", action="store_true", help="HyperGCN 不使用 mediator")
    ap.add_argument("--fast", action="store_true", help="FastHyperGCN：结构只按输入特征算一次")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数，0 表示默认")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True,
                    help="是否保存验证集最优权重（默认保存，可用 --no-save-model 关闭）")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.models == "all" or args.models == "paper":
        names = list(PAPER_MODELS)
    elif args.models == "extra":
        names = list(PAPER_MODELS) + ["HGNN", "MLP"]
    else:
        names = [m.strip() for m in args.models.split(",") if m.strip()]
    for n in names:
        if n not in MODELS:
            raise SystemExit(f"未知模型 {n}，可选：{list(MODELS)}")
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    print(f"[env] torch={torch.__version__} device={dev} split={args.split} feature_mode={args.feature_mode}")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or f"{args.split}_fm{args.feature_mode}_tr{args.train_rate}_{args.loss}"
    ckpt_root = os.path.join(args.out, "checkpoints", f"{tag}_{run_stamp}")

    cache = {}
    results = {}
    for name in names:
        fmt = model_format(name)
        if fmt not in cache:
            cache[fmt] = load_benchpcnp(fmt, args.data_root, args.feature_mode)
        data = cache[fmt]
        runs = []
        for t in range(args.trials):
            train_seed = args.seed + t
            sseed = args.split_seed + (t if args.resplit else 0)
            # 划分在 CPU 上做即可；多 trial 时 data 可能仍在上次 trial 的 GPU 上，data.py 内比对已处理
            make_split(data, args.split, seed=sseed, train_rate=args.train_rate, root=args.data_root)
            data.to(dev)
            if t == 0:
                print(f"[{name}] fmt={fmt} train/val/test={int(data.train_mask.sum())}/"
                      f"{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")
            r = run_one(name, data, args, seed=train_seed, trial_idx=t, split_seed=sseed,
                        ckpt_dir=os.path.join(ckpt_root, name), log=print)
            runs.append(r)
            print(f"  trial{t + 1} (seed={train_seed}, split_seed={sseed}): ACC={r['acc']:.2f} F1={r['f1']:.2f} "
                  f"P={r['precision']:.2f} R={r['recall']:.2f} best_ep={r['best_epoch']} time={r['time_s']:.1f}s")
        agg = {}
        for k in ("acc", "f1", "precision", "recall", "f1_weighted", "time_s"):
            v = np.array([r[k] for r in runs])
            agg[k] = (float(v.mean()), float(v.std()))
        results[name] = {
            "format": fmt,
            "runs": runs,
            "mean_std": agg,
            "seeds": [args.seed + t for t in range(args.trials)],
            "split_seeds": [args.split_seed + (t if args.resplit else 0) for t in range(args.trials)],
        }
        paper = PAPER.get(name)
        ref = f" | 论文 ACC={paper[0]:.2f} F1={paper[1]:.2f}" if paper else ""
        print(f"[{name}] ACC={agg['acc'][0]:.2f}±{agg['acc'][1]:.2f} F1={agg['f1'][0]:.2f}±{agg['f1'][1]:.2f}"
              f" P={agg['precision'][0]:.2f} R={agg['recall'][0]:.2f}{ref}")

    print("\n| 模型 | 图格式 | ACC(%) | F1(%) | Precision(%) | Recall(%) | 单次耗时(s) | 论文 ACC | 论文 F1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        a = r["mean_std"]
        p = PAPER.get(name, (float("nan"), float("nan")))
        print(f"| {name} | {r['format']} | {a['acc'][0]:.2f}±{a['acc'][1]:.2f} | {a['f1'][0]:.2f}±{a['f1'][1]:.2f} | "
              f"{a['precision'][0]:.2f} | {a['recall'][0]:.2f} | {a['time_s'][0]:.1f} | {p[0]:.2f} | {p[1]:.2f} |")

    os.makedirs(args.out, exist_ok=True)
    summary = {
        "args": vars(args),
        "checkpoint_dir": ckpt_root if args.save_model else None,
        "base_seed": args.seed,
        "split_seed_base": args.split_seed,
        "results": results,
    }
    path = os.path.join(args.out, f"results_{tag}_{run_stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存：{path}")
    if args.save_model:
        print(f"权重目录：{ckpt_root}")


if __name__ == "__main__":
    main()