#!/usr/bin/env bash
# Launch training on one or more nodes with torchrun.
#
#   scripts/launch.sh configs/pretrain/200m.yaml [key.path=value ...]
#
# Environment:
#   NPROC        GPUs per node (default: all visible GPUs)
#   NNODES       number of nodes (default: 1)
#   MASTER_ADDR  rendezvous host, required when NNODES > 1 (run the same command on every node)
#   MASTER_PORT  rendezvous port (default: 29500)
#   RUN_ID       rendezvous id shared by the nodes of one job (default: myk)
set -euo pipefail

NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
NNODES=${NNODES:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

if [[ "$NNODES" -gt 1 ]]; then
    : "${MASTER_ADDR:?set MASTER_ADDR to the rendezvous host for multi-node runs}"
    RDZV=(--nnodes "$NNODES" --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:${MASTER_PORT:-29500}" --rdzv_id "${RUN_ID:-myk}")
else
    RDZV=(--standalone --nnodes 1)
fi

exec torchrun "${RDZV[@]}" --nproc_per_node "$NPROC" -m myk_llm.train "$@"
