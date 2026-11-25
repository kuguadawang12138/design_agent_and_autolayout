# -*- coding: utf-8 -*-
"""
超参数小网格搜索（论文未给出学习率 / 隐层 / dropout 等，需要自行搜索）。
示例：python sweep.py --models HyperResGCN --trials 1
结果按验证集 ACC 排序输出，并写入 results/sweep_*.json。
"""
import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from data import DEFAULT_ROOT, load_benchpcnp, make_split
from models import model_format
from train import run_one

GRID = {
    "lr": [0.01, 0.005],
    "weight_decay": [5e-4, 5e-5],
    "dropout": [0.5, 0.2, 0.0],
    "hidden": [128, 256],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="HyperResGCN")
    ap.add_argument("--split", default="official")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--loss", default="ce")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_configs", type=int, default=0, help="只跑前 N 组配置，0 表示全部")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))

    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    if args.max_configs:
        configs = configs[:args.max_configs]
    rows = []
    for name in [m.strip() for m in args.models.split(",")]:
        data = load_benchpcnp(model_format(name), DEFAULT_ROOT, "111", verbose=False)
        make_split(data, args.split)
        data.to(dev)
        for cfg in configs:
            ns = argparse.Namespace(heads=8, no_mediators=False, fast=False, loss=args.loss,
                                    label_smoothing=0.0, epochs=args.epochs, patience=args.patience,
                                    eval_every=args.eval_every, select="acc", verbose=False,
                                    log_every=50, **cfg)
            runs = [run_one(name, data, ns, seed=t) for t in range(args.trials)]
            r = {k: float(np.mean([x[k] for x in runs])) for k in ("best_val", "acc", "f1", "time_s")}
            rows.append({"model": name, **cfg, **r})
            print(f"{name} {cfg} val={r['best_val']:.2f} test_acc={r['acc']:.2f} "
                  f"test_f1={r['f1']:.2f} {r['time_s']:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["model"], -r["best_val"]))
    print("\n按验证集 ACC 排序：")
    for r in rows:
        print(r)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"sweep_{args.models.replace(',', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"已保存：{path}")


if __name__ == "__main__":
    main()
