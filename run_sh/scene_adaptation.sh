#!/usr/bin/env bash
GPUNO=$1
ITER=$2
CKPT=$3
CONFIG=$4
((PORT=$1+29500))
echo $PORT


CUDA_VISIBLE_DEVICES=$GPUNO  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=$PORT train/train_ddp_sceneflow_fast_rendering_semantic.py \
 -c $CONFIG \
  --launcher="pytorch" --ft_path=$CKPT \
  --fast_render_iter=$ITER
