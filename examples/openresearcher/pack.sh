#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MATERIALIZED_JSONL="${MATERIALIZED_JSONL:-${SCRIPT_DIR}/data/converted_gpt_oss_search_correct.materialized.jsonl}"
PACKED_JSONL="${PACKED_JSONL:-${SCRIPT_DIR}/data/converted_gpt_oss_search_correct.packed_262144.jsonl}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-262144}"
TOLERANCE="${TOLERANCE:-1000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
STARTING_EPOCH="${STARTING_EPOCH:-0}"

if [[ ! -f "${MATERIALIZED_JSONL}" ]]; then
  echo "ERROR: materialized file not found: ${MATERIALIZED_JSONL}"
  exit 1
fi

python3 "${SCRIPT_DIR}/pack.py" \
  --input "${MATERIALIZED_JSONL}" \
  --output "${PACKED_JSONL}" \
  --maximum-token "${MAX_SEQ_LEN}" \
  --tolerance "${TOLERANCE}" \
  --starting-epoch "${STARTING_EPOCH}" \
  --num-epochs "${NUM_EPOCHS}"
