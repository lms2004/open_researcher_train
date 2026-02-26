#!/bin/bash
# Convert HF -> Megatron **without** --export-te-mcore-model.
# Use this when you want a more compatible checkpoint for training, to avoid
# "Missing key in checkpoint state_dict: ... _extra_state" when the training
# env or TE/mcore version differs from the one used at export time.
#
# Usage: same as convert.sh, e.g.
#   MLM_MODEL_SAVE=../../../checkpoints/qwen3_0.6b_init_mlm ./convert_compat.sh Qwen/Qwen3-0.6B

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Common arguments and base model specific arguments
source "${SCRIPT_DIR}/conf/arguments.sh"

# No --export-te-mcore-model => checkpoint is more compatible at load time
MLM_DEFAULT_ARGS="
    --distributed-timeout-minutes 60 \
    --finetune \
    --auto-detect-ckpt-format \
"

if [ -z ${HF_TOKEN} ]; then
    printf "${MLM_WARNING} Variable ${PURPLE}HF_TOKEN${WHITE} is not set! HF snapshot download may fail!\n"
fi

if [ -z ${MLM_MODEL_SAVE} ]; then
    MLM_MODEL_SAVE=${MLM_WORK_DIR}/${MLM_MODEL_CFG}_mlm_compat
    printf "${MLM_WARNING} Variable ${PURPLE}MLM_MODEL_SAVE${WHITE} not set (default: ${MLM_MODEL_SAVE}).\n"
fi

if [ -z ${MLM_MODEL_CKPT} ]; then
    if [ -z ${HF_MODEL_CKPT} ]; then
        HF_MODEL_CKPT=${1}
    fi
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --pretrained-model-path ${HF_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
else
    ${LAUNCH_SCRIPT} ${SCRIPT_DIR}/convert_model.py \
        ${MODEL_ARGS} \
        --tensor-model-parallel-size ${TP} \
        --expert-tensor-parallel-size ${ETP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --load ${MLM_MODEL_CKPT} \
        --save ${MLM_MODEL_SAVE} \
        ${MLM_DEFAULT_ARGS} ${MLM_EXTRA_ARGS}
fi
