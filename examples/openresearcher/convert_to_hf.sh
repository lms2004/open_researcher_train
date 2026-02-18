#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_OPT_DIR="${SCRIPT_DIR}/../post_training/modelopt"

MLM_MODEL_CFG="${1:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
source "${MODEL_OPT_DIR}/conf/arguments.sh" "${MLM_MODEL_CFG}"

if [[ -z "${MLM_MODEL_CKPT:-}" ]]; then
  echo "ERROR: MLM_MODEL_CKPT is required (trained Megatron checkpoint)."
  exit 1
fi

if [[ -z "${EXPORT_DIR:-}" ]]; then
  EXPORT_DIR="${MLM_WORK_DIR}/${MLM_MODEL_CFG}_hf_export"
  echo "WARNING: EXPORT_DIR not set, use default ${EXPORT_DIR}"
fi

if [[ -z "${HF_MODEL_CKPT:-}" ]]; then
  HF_MODEL_CKPT="${TOKENIZER_MODEL}"
  echo "WARNING: HF_MODEL_CKPT not set, use ${HF_MODEL_CKPT} as base config."
fi

if [[ "${TP}" != "1" ]]; then
  TP=1
  echo "WARNING: TP forced to 1 for export."
fi

MLM_DEFAULT_ARGS="--finetune --auto-detect-ckpt-format --export-te-mcore-model --use-cpu-initialization"

${LAUNCH_SCRIPT} "${MODEL_OPT_DIR}/export.py" \
  ${MODEL_ARGS} \
  --tensor-model-parallel-size "${TP}" \
  --pipeline-model-parallel-size "${PP}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --load "${MLM_MODEL_CKPT}" \
  --pretrained-model-name "${HF_MODEL_CKPT}" \
  --export-dir "${EXPORT_DIR}" \
  ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS:-}
