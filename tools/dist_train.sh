#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
# NNODES=${NNODES:-4}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-12345}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NNODES=${MLP_WORKER_NUM:-1}
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
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# torchrun \
#     --standalone \
#     --nnodes=$NNODES \
#     --nproc_per_node=$GPUS \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:3}
