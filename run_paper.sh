#!/usr/bin/env bash
# BenchPCNP 论文实验一键复现（Linux / GPU 服务器）
# 用法：bash run_paper.sh            # 默认 cuda
#       DEVICE=cpu bash run_paper.sh
set -e
cd "$(dirname "$0")"
DEVICE=${DEVICE:-cuda}
PY=${PY:-python}

# 数据：官方仓库自带处理好的图（如目录不存在则克隆）
if [ ! -d BenchPCNP_src/Data/Graphs ]; then
  git clone --depth 1 https://github.com/MrBade/BenchPCNP.git BenchPCNP_src
fi

# 1) Table 5 / Table 6：9 个基线 × 5 次随机实验，官方划分
$PY train.py --models paper --trials 5 --device $DEVICE --tag table5_official

# 2) 真正的 6:2:2 随机划分（同一划分，5 个随机种子）
$PY train.py --models paper --trials 5 --split random --device $DEVICE --tag table5_random622

# 3) Fig.6 特征消融（超图模型）：Full(全 1)=222，只用类别=100，只用封装=010，只用引脚=001，全部=111
for FM in 222 100 010 001 111; do
  $PY train.py --models HyperGCN,HyperJKNet,HyperResGCN --trials 5 --feature_mode $FM \
      --split random --device $DEVICE --tag fig6_fm$FM
done

# 4) Fig.7 训练集比例（超图模型）
for TR in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
  $PY train.py --models HyperGCN,HyperJKNet,HyperResGCN --trials 5 --train_rate $TR \
      --split random --device $DEVICE --tag fig7_tr$TR
done
