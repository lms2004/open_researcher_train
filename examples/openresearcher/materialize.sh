#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RAW_JSONL="${RAW_JSONL:-${SCRIPT_DIR}/data/converted_gpt_oss_search_correct.jsonl}"
MATERIALIZED_JSONL="${MATERIALIZED_JSONL:-${SCRIPT_DIR}/data/converted_gpt_oss_search_correct.materialized.jsonl}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
DEBUG_FAILURES="${DEBUG_FAILURES:-}"
EXTRA_INFO="${EXTRA_INFO:-0}"
WORKERS="${WORKERS:-1}"
CHUNKSIZE="${CHUNKSIZE:-8}"

ARGS=(
  --input "${RAW_JSONL}"
  --output "${MATERIALIZED_JSONL}"
  --tokenizer-model "${TOKENIZER_MODEL}"
  --trust-remote-code
  --workers "${WORKERS}"
  --chunksize "${CHUNKSIZE}"
)

if [[ -n "${CHAT_TEMPLATE_FILE}" ]]; then
  ARGS+=(--chat-template-file "${CHAT_TEMPLATE_FILE}")
fi
if [[ -n "${DEBUG_FAILURES}" ]]; then
  ARGS+=(--debug-failures "${DEBUG_FAILURES}")
fi
if [[ "${EXTRA_INFO}" == "1" ]]; then
  ARGS+=(--extra-info)
fi

if [[ ! -f "${RAW_JSONL}" ]]; then
  echo "ERROR: input file not found: ${RAW_JSONL}"
  exit 1
fi

python3 "${SCRIPT_DIR}/materialize.py" \
  "${ARGS[@]}"
