#!/usr/bin/env bash
cd /qcraft-vepfs-01/Perception/perception-users/liangkaiwen/mmdet3.x/
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/qcraft-vepfs-01/Perception/perception-users/liangkaiwen/mmdet3.x/

CONFIG=$1
GPUS=$2

NNODES=${MLP_WORKER_NUM-:1}
NODE_RANK=${MLP_ROLE_INDEX:-0}
PORT=${MLP_WORKER_0_PORT:-29500}
MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
