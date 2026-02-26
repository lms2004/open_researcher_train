#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${THIS_DIR}/../.." && pwd)"
MODEL_OPT_DIR="${ROOT_DIR}/examples/post_training/modelopt"

# Guard: arguments.sh and model confs use [ -z ${VAR} ] without :- ; with nounset they fail if VAR is unset
: "${SANDBOX_ENV_SETUP:=}"
: "${MLM_ENV_SETUP:=}"
: "${MLM_EXTRA_ARGS:=}"
: "${MLM_RESUME_ARGS:=}"
: "${SANDBOX_ROOT:=}"
: "${HF_TOKEN:=}"
: "${HF_MODEL_CKPT:=}"
: "${LAUNCH_SCRIPT:=}"
: "${MLM_WORK_DIR:=/tmp/megatron_workspace}"
: "${MLM_SKIP_INSTALL:=}"
: "${TP:=1}"
: "${ETP:=${TP}}"
: "${EP:=1}"
: "${PP:=1}"
: "${CP:=1}"
: "${DP:=1}"

# Required env
: "${MLM_MODEL_CKPT:?ERROR: set MLM_MODEL_CKPT to a Megatron checkpoint dir}"
: "${MLM_MODEL_SAVE:=checkpoints/minimal_run}"

# Optional env / defaults
: "${DATA_PACKED_JSONL:=${THIS_DIR}/data/converted_gpt_oss_search_correct.packed_262144.jsonl}"
: "${SPLIT:=98,2,0}"
: "${MAX_SEQ:=30000}"
: "${GBS:=4}"
: "${MICRO_BS:=1}"
: "${TRAIN_SAMPLES:=50}"
: "${LR:=5e-5}"
: "${MIN_LR:=5e-6}"
: "${PROMPT_FORMAT:=identity}"
: "${NUM_WORKERS:=1}"

MLM_MODEL_CFG="${1:-Qwen/Qwen3-0.6B}"

# Load model args from modelopt (gives MODEL_ARGS, TOKENIZER_MODEL, etc.)
SCRIPT_DIR="${MODEL_OPT_DIR}"
source "${MODEL_OPT_DIR}/conf/arguments.sh" "${MLM_MODEL_CFG}"

if [[ ! -f "${DATA_PACKED_JSONL}" ]]; then
  echo "ERROR: DATA_PACKED_JSONL not found: ${DATA_PACKED_JSONL}"
  echo "Run materialize.sh and pack.sh first, or set DATA_PACKED_JSONL to an existing packed jsonl."
  exit 1
fi

# (Very important) drop qk-layernorm if ckpt doesn't have q_layernorm extra_state
# This directly targets your missing key: decoder.layers.self_attention.q_layernorm._extra_state/...
MODEL_ARGS="$(echo " ${MODEL_ARGS} " | sed 's/ --qk-layernorm / /g')"

PRETRAIN_EXE="${ROOT_DIR}/pretrain_gpt.py"
mkdir -p "${MLM_MODEL_SAVE}"

set -x
${LAUNCH_SCRIPT} "${PRETRAIN_EXE}" \
  ${MODEL_ARGS} \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --load "${MLM_MODEL_CKPT}" \
  --save "${MLM_MODEL_SAVE}" \
  --sft \
  --tokenizer-type SFTTokenizer \
  --sft-tokenizer-prompt-format "${PROMPT_FORMAT}" \
  --data-path "${DATA_PACKED_JSONL}" \
  --split "${SPLIT}" \
  --micro-batch-size "${MICRO_BS}" \
  --global-batch-size "${GBS}" \
  --seq-length "${MAX_SEQ}" \
  --max-position-embeddings "${MAX_SEQ}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --lr "${LR}" \
  --min-lr "${MIN_LR}" \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --bf16 \
  --no-gradient-accumulation-fusion \
  --log-interval 1 \
  --save-interval 50 \
  --eval-iters 1 \
  --eval-interval 500000 \
  --num-workers "${NUM_WORKERS}" \
  --no-create-attention-mask-in-dataloader \
  --trust-remote-code \
  --export-force-local-attention \
  --finetune