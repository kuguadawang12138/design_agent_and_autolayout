# -*- coding: utf-8 -*-
"""
节点分类损失函数。论文只给出分类任务，默认使用交叉熵（ce）；
另外提供针对长尾类别分布（论文 Fig.4a）的可选损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(y_train: torch.Tensor, n_cls: int, power: float = 0.5) -> torch.Tensor:
    """按训练集频次的 -power 次方计算类别权重，并归一化到均值为 1；训练集中缺失的类别权重为 0。"""
    cnt = torch.bincount(y_train, minlength=n_cls).float()
    w = torch.where(cnt > 0, cnt.clamp(min=1).pow(-power), torch.zeros_like(cnt))
    return w * (cnt > 0).sum() / w.sum().clamp(min=1e-12)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, -1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def build_loss(name: str, y_train: torch.Tensor, n_cls: int, label_smoothing: float = 0.0):
    """
    name:
      ce     标准交叉熵（论文默认）
      wce    类别加权交叉熵（频次^-0.5）
      focal  Focal Loss（gamma=2）
      cb_focal 类别加权 Focal Loss
    """
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    w = class_weights(y_train, n_cls).to(y_train.device)
    if name == "wce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(2.0)
    if name == "cb_focal":
        return FocalLoss(2.0, w)
    raise ValueError(name)
