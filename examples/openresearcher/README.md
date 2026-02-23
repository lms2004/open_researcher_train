# OpenResearcher End-to-End (Public Megatron-LM)

This README describes the full OpenResearcher workflow in the public Megatron-LM repo.  
All related scripts are under `examples/openresearcher`.

## Overview

Workflow order:

1. Materialize raw JSONL
2. Sequence packing
3. SFT training (adapted public Megatron approach)
4. Convert trained weights to HF

## Prerequisites

```bash
uv venv --seed --python=3.12
uv pip install -U pip setuptools wheel pybind11 psutil

pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install --no-build-isolation -e ".[mlm,dev]"
pip install -U nvidia-modelopt
pip install transformer-engine[pytorch,flash-attn]
pip install mamba-ssm
```

## Scripts

- `examples/openresearcher/materialize.sh`
- `examples/openresearcher/materialize.py`
- `examples/openresearcher/pack.sh`
- `examples/openresearcher/pack.py`
- `examples/openresearcher/train_search_correct_only_base.sh`
- `examples/openresearcher/convert_to_hf.sh`
- `examples/openresearcher/README.md`

## Quick Start

```bash
cd examples/openresearcher
bash materialize.sh
bash pack.sh
```

## Full Workflow

### 1) Materialize Raw Data

```bash
cd examples/openresearcher

# By default, reads from data/converted_gpt_oss_search_correct.jsonl
# You only need to set TOKENIZER_MODEL / CHAT_TEMPLATE_FILE as needed
TOKENIZER_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
WORKERS=64 CHUNKSIZE=16 \
bash materialize.sh
```

### 2) Sequence Packing

```bash
cd examples/openresearcher

# By default, reads the output from the previous step:
# data/converted_gpt_oss_search_correct.materialized.jsonl
# Default output:
# data/converted_gpt_oss_search_correct.packed_262144.jsonl
bash pack.sh
```

### 3) SFT Training

Convert HF checkpoint to Megatron checkpoint (one-time, for continued training):

```bash
cd examples/post_training/modelopt

# OpenResearcher/OpenResearcher-30B-A3B is already SFT-trained
# from nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16.
# Use it as the initialization checkpoint for further training.
TP=1 EP=1 ETP=1 PP=1 CP=1 DP=1 \
HF_MODEL_CKPT=OpenResearcher/OpenResearcher-30B-A3B \
MLM_MODEL_SAVE=../../../checkpoints/openresearcher_30a3b_init_mlm \
MLM_SKIP_INSTALL=1 \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./convert.sh OpenResearcher/OpenResearcher-30B-A3B
```

Run SFT:

```bash
TP=2 EP=1 ETP=1 PP=1 CP=2 DP=1 \
MLM_MODEL_CKPT=checkpoints/openresearcher_30a3b_init_mlm \
MLM_MODEL_SAVE=checkpoints/openresearcher_30a3b_continue \
BLEND_PATH=examples/openresearcher/data/search_blend_path.json \
MAX_SEQ=262144 \
GBS=60 \
TRAIN_SAMPLES=20872 \
LR=5e-5 \
MIN_LR=5e-6 \
PROMPT_FORMAT=identity \
MLM_SKIP_INSTALL=1 \
bash examples/openresearcher/train_search_correct_only_base.sh
```

### 4) Convert Weights to HF

```bash
cd examples/openresearcher

TP=1 PP=1 EP=1 ETP=1 CP=1 DP=1 \
MLM_MODEL_CKPT=checkpoints/openresearcher_30a3b_continue \
HF_MODEL_CKPT=OpenResearcher/OpenResearcher-30B-A3B \
EXPORT_DIR=checkpoints/openresearcher_30a3b_continue_hf \
MLM_SKIP_INSTALL=1 \
bash convert_to_hf.sh OpenResearcher/OpenResearcher-30B-A3B
```

## Outputs

- Materialized data: `data/converted_gpt_oss_search_correct.materialized.jsonl`
- Packed data: `data/converted_gpt_oss_search_correct.packed_262144.jsonl`
- Continued-training init Megatron checkpoint: `checkpoints/openresearcher_30a3b_init_mlm`
- Continued-training output checkpoint: `checkpoints/openresearcher_30a3b_continue`
- HF export: `checkpoints/openresearcher_30a3b_continue_hf`

## Notes

- This script set has been streamlined and rewritten after aligning core logic with the internal `materialize.py/materialize_fast.py/pack.py/train_search_correct_only_base.sh` you provided.
- The training script is based on the public repository argument system (`examples/post_training/modelopt/conf/arguments.sh`) while retaining internal-style entry points for parallelism and long-context configuration.
- See `examples/openresearcher/README.md` for more details.
