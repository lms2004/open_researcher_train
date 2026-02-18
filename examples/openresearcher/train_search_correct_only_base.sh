#!/usr/bin/env bash
set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${THIS_DIR}/../.." && pwd)"
MODEL_OPT_DIR="${ROOT_DIR}/examples/post_training/modelopt"

# Guard against nounset failures in upstream arguments.sh, which accesses
# several vars as ${VAR} before checking.
: "${SANDBOX_ENV_SETUP:=}"
: "${MLM_ENV_SETUP:=}"
: "${MLM_EXTRA_ARGS:=}"
: "${MLM_RESUME_ARGS:=}"
: "${SANDBOX_ROOT:=}"
: "${HF_TOKEN:=}"
: "${HF_MODEL_CKPT:=}"
: "${LAUNCH_SCRIPT:=}"
: "${MLM_WORK_DIR:=/tmp/megatron_workspace}"
: "${TP:=1}"
: "${ETP:=${TP}}"
: "${EP:=1}"
: "${PP:=1}"
: "${CP:=1}"
: "${DP:=1}"

MLM_MODEL_CFG="${1:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
# arguments.sh expects SCRIPT_DIR to point to examples/post_training/modelopt
SCRIPT_DIR="${MODEL_OPT_DIR}"
source "${MODEL_OPT_DIR}/conf/arguments.sh" "${MLM_MODEL_CFG}"

# Data config:
# - prefer BLEND_PATH (per-split-data-args-path) for packed blend
# - fallback DATA_PACKED_JSONL (single jsonl path)
BLEND_PATH="${BLEND_PATH:-}"
DATA_PACKED_JSONL="${DATA_PACKED_JSONL:-${THIS_DIR}/data/converted_gpt_oss_search_correct.packed_262144.jsonl}"
SPLIT="${SPLIT:-98,2,0}"

# Core train hyperparams (modeled after internal train_search_correct_only_base.sh)
MAX_SEQ="${MAX_SEQ:-262144}"
GBS="${GBS:-60}"
MICRO_BS="${MICRO_BS:-1}"
LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-5e-6}"
LR_WARMUP_SAMPLES="${LR_WARMUP_SAMPLES:-2000}"
LR_DECAY_SAMPLES="${LR_DECAY_SAMPLES:-20872}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20872}"
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
PROMPT_FORMAT="${PROMPT_FORMAT:-identity}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# Runtime/logging paths
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/.openresearcher_runs}"
RUN_NAME="${RUN_NAME:-openresearcher_${MLM_MODEL_CFG//\//_}_$(date +%Y%m%d_%H%M%S)}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/logs/${RUN_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
DATACACHE_DIR="${DATACACHE_DIR:-${OUTPUT_ROOT}/data_cache}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard/${RUN_NAME}}"
mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}"

# Runtime env defaults synced from the internal launcher.
export UB_TIMEOUT="${UB_TIMEOUT:-720}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE:-2097152}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCHINDUCTOR_WORKER_START="${TORCHINDUCTOR_WORKER_START:-fork}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${MLM_MODEL_CKPT:-}" ]]; then
  echo "ERROR: MLM_MODEL_CKPT is required (Megatron ckpt dir, e.g. output from convert.sh)."
  exit 1
fi

if [[ -z "${MLM_MODEL_SAVE:-}" ]]; then
  MLM_MODEL_SAVE="${CHECKPOINT_DIR}"
fi
mkdir -p "${MLM_MODEL_SAVE}"

if [[ -n "${BLEND_PATH}" ]]; then
  DATA_ARGS="--per-split-data-args-path ${BLEND_PATH}"
else
  if [[ ! -f "${DATA_PACKED_JSONL}" ]]; then
    echo "ERROR: DATA_PACKED_JSONL not found: ${DATA_PACKED_JSONL}"
    echo "Set BLEND_PATH or DATA_PACKED_JSONL."
    exit 1
  fi
  DATA_ARGS="--data-path ${DATA_PACKED_JSONL} --split ${SPLIT}"
fi

if [[ -z "${MLM_TRAIN_ARGS:-}" ]]; then
  MLM_TRAIN_ARGS=" \
    --no-gradient-accumulation-fusion \
    --sequence-parallel \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --micro-batch-size ${MICRO_BS} \
    --global-batch-size ${GBS} \
    --seq-length ${MAX_SEQ} \
    --max-position-embeddings ${MAX_SEQ} \
    --log-interval 1 \
    --save-interval 50 \
    --eval-iters 1 \
    --eval-interval 500000 \
    --num-workers ${NUM_WORKERS} \
    --no-create-attention-mask-in-dataloader \
    --distributed-timeout-minutes 230 \
    --tiktoken-pattern v2 \
    --bf16 \
    --use-distributed-optimizer \
    --ddp-num-buckets 8 \
    --ddp-pad-buckets-for-high-nccl-busbw \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tp-comm-overlap \
    --override-opt_param-scheduler \
    --exit-duration-in-mins 235 \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --use-persistent-ckpt-worker \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --ckpt-assume-constant-structure \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-energy \
    --logging-level 20 \
    --log-memory-interval 900 \
    --manual-gc \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --log-progress \
    --trust-remote-code \
    --tensorboard-dir ${TENSORBOARD_DIR} \
  "
fi

if [[ -z "${MLM_OPTIM_ARGS:-}" ]]; then
  MLM_OPTIM_ARGS=" \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style ${LR_DECAY_STYLE} \
    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    --train-samples ${TRAIN_SAMPLES} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
  "
fi

MLM_DATA_ARGS=" \
  --sft \
  --tokenizer-type SFTTokenizer \
  --sft-tokenizer-prompt-format ${PROMPT_FORMAT} \
  ${DATA_ARGS} \
"

MLM_DEFAULT_ARGS=" \
  --finetune \
  --auto-detect-ckpt-format \
  --export-te-mcore-model \
"

if [[ "${MODEL_ARGS}" == *"MambaModel"* ]]; then
  PRETRAIN_EXE="${ROOT_DIR}/pretrain_mamba.py"
else
  PRETRAIN_EXE="${ROOT_DIR}/pretrain_gpt.py"
fi

# pretrain_mamba.py requires an explicit Mamba layer spec.
if [[ "${PRETRAIN_EXE}" == *"pretrain_mamba.py" ]]; then
  if [[ "${MLM_EXTRA_ARGS}" != *"--spec"* ]]; then
    MLM_EXTRA_ARGS="${MLM_EXTRA_ARGS} --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec"
  fi
fi

LOG_FILE="${LOGS_DIR}/train.log"
echo "writing logs to ${LOG_FILE}"

set -x
${LAUNCH_SCRIPT} "${PRETRAIN_EXE}" \
  ${MODEL_ARGS} \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --tensor-model-parallel-size "${TP}" \
  --expert-tensor-parallel-size "${ETP}" \
  --expert-model-parallel-size "${EP}" \
  --pipeline-model-parallel-size "${PP}" \
  --context-parallel-size "${CP}" \
  --load "${MLM_MODEL_CKPT}" \
  --save "${MLM_MODEL_SAVE}" \
  ${MLM_DATA_ARGS} \
  ${MLM_OPTIM_ARGS} \
  ${MLM_TRAIN_ARGS} \
  ${MLM_DEFAULT_ARGS} \
  ${MLM_RESUME_ARGS:-} ${MLM_EXTRA_ARGS:-} \
  2>&1 | tee "${LOG_FILE}"
