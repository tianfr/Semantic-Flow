#!/usr/bin/env bash
function rand(){
    min=$1
    max=$(($2 - $min + 1))
    num=$(($RANDOM+1000000000)) # 增加一个10位的数再求余
    echo $(($num%$max + $min))
}
RAND_NO=$(rand 10 200)
GPUNO=$1
CONFIG=$2
((PORT=$1+39500+$RAND_NO))
echo $PORT

TORCH_DISTRIBUTED_DEBUG=DETAIL CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$GPUNO  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 \
--master_addr=127.0.0.1 --master_port=$PORT train/train_ddp_sceneflow_semantic_completion.py \
 -c $CONFIG \
  --launcher="pytorch"  --N_static_iters=60001  --N_iters=80001