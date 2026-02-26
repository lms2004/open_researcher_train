# OpenResearcher SFT：Qwen3-0.6B（单卡 A100 80GB）

在**单张 A100 80GB** 上，用 **Qwen3-0.6B** 和 **约 30k token 上下文** 跑 OpenResearcher 数据 SFT。  
模型配置：`examples/post_training/modelopt/conf/Qwen/Qwen3-0.6B.sh`。  
Qwen 训练使用 **`train_minimal.sh`**（单卡精简参数，默认 Qwen3-0.6B）。

建议先按 **一、跑通示例（50 条数据）** 跑通全流程，再按 **二、完整数据集** 换数据与步数做正式训练。

---

## 环境

与主 README 一致，例如：

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install --no-build-isolation -e ".[mlm,dev]"
pip install -U nvidia-modelopt
pip install transformer-engine[pytorch,flash-attn]
# 若用 Mamba 再装: pip install mamba-ssm
```

---

## 一、跑通示例（50 条数据）

使用 `data/converted_gpt_oss_search_correct.example.jsonl`（50 条）跑通：Materialize → Pack → HF→Megatron → SFT → 转回 HF。  
所有中间/结果路径带 `example` 或 `qwen3_0.6b_example`，与完整数据集互不覆盖。

### 1) Materialize（Qwen tokenizer，输入 50 条）

```bash
cd examples/openresearcher

RAW_JSONL=data/converted_gpt_oss_search_correct.example.jsonl \
MATERIALIZED_JSONL=data/converted_gpt_oss_search_correct.example.materialized.jsonl \
TOKENIZER_MODEL=Qwen/Qwen3-0.6B \
WORKERS=2 CHUNKSIZE=4 \
bash materialize.sh
```

输出：`data/converted_gpt_oss_search_correct.example.materialized.jsonl`。

### 2) Pack（max 30000 tokens）

```bash
cd examples/openresearcher

MATERIALIZED_JSONL=data/converted_gpt_oss_search_correct.example.materialized.jsonl \
PACKED_JSONL=data/converted_gpt_oss_search_correct.example.packed_30000.jsonl \
MAX_SEQ_LEN=30000 \
bash pack.sh
```

输出：`data/converted_gpt_oss_search_correct.example.packed_30000.jsonl`。

### 3) HF → Megatron（一次性）

使用 **`convert_compat.sh`**（不导出 TE mcore，ckpt 更兼容，避免训练 load 时 `_extra_state` 缺失）：

```bash
cd examples/post_training/modelopt

TP=1 EP=1 ETP=1 PP=1 CP=1 DP=1 \
HF_MODEL_CKPT=Qwen/Qwen3-0.6B \
MLM_MODEL_SAVE=../../../checkpoints/qwen3_0.6b_init_mlm \
MLM_SKIP_INSTALL=1 \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./convert_compat.sh Qwen/Qwen3-0.6B
```

### 4) SFT 训练（50 条示例数据，快速跑通）

```bash
# 在 Megatron-LM 仓库根目录执行
MLM_MODEL_CKPT=checkpoints/qwen3_0.6b_init_mlm \
MLM_MODEL_SAVE=checkpoints/qwen3_0.6b_example_sft \
DATA_PACKED_JSONL=examples/openresearcher/data/converted_gpt_oss_search_correct.example.packed_30000.jsonl \
MAX_SEQ=30000 \
GBS=4 \
TRAIN_SAMPLES=200 \
bash examples/openresearcher/train_minimal.sh Qwen/Qwen3-0.6B
```

说明：`train_minimal.sh` 默认即 Qwen/Qwen3-0.6B、MAX_SEQ=30000、GBS=4；`TRAIN_SAMPLES=200` 用于 50 条数据快速验证。

### 5) 转回 HuggingFace

```bash
cd examples/openresearcher

TP=1 PP=1 EP=1 ETP=1 CP=1 DP=1 \
MLM_MODEL_CKPT=../../checkpoints/qwen3_0.6b_example_sft \
HF_MODEL_CKPT=Qwen/Qwen3-0.6B \
EXPORT_DIR=../../checkpoints/qwen3_0.6b_example_sft_hf \
MLM_SKIP_INSTALL=1 \
bash convert_to_hf.sh Qwen/Qwen3-0.6B
```

导出目录（相对仓库根）：`checkpoints/qwen3_0.6b_example_sft_hf`。

---

## 二、完整数据集（跑通后再用）

确认 50 条流程无误后，用完整数据 `data/converted_gpt_oss_search_correct.jsonl` 训练。  
**只需改输入/输出路径和 TRAIN_SAMPLES**，其余与上面一致。

### 1) Materialize（完整数据）

```bash
cd examples/openresearcher

RAW_JSONL=data/converted_gpt_oss_search_correct.jsonl \
MATERIALIZED_JSONL=data/converted_gpt_oss_search_correct.materialized.jsonl \
TOKENIZER_MODEL=Qwen/Qwen3-0.6B \
WORKERS=4 CHUNKSIZE=8 \
bash materialize.sh
```

### 2) Pack

```bash
cd examples/openresearcher

MATERIALIZED_JSONL=data/converted_gpt_oss_search_correct.materialized.jsonl \
PACKED_JSONL=data/converted_gpt_oss_search_correct.packed_30000.jsonl \
MAX_SEQ_LEN=30000 \
bash pack.sh
```

### 3) HF → Megatron

与上面相同，使用 **`convert_compat.sh`**（若已做过可跳过）：

```bash
cd examples/post_training/modelopt
TP=1 EP=1 ETP=1 PP=1 CP=1 DP=1 \
HF_MODEL_CKPT=Qwen/Qwen3-0.6B \
MLM_MODEL_SAVE=../../../checkpoints/qwen3_0.6b_init_mlm \
MLM_SKIP_INSTALL=1 \
MLM_EXTRA_ARGS="--no-gradient-accumulation-fusion" \
./convert_compat.sh Qwen/Qwen3-0.6B
```

### 4) SFT 训练（完整数据）

```bash
# 在 Megatron-LM 仓库根目录执行
MLM_MODEL_CKPT=checkpoints/qwen3_0.6b_init_mlm \
MLM_MODEL_SAVE=checkpoints/qwen3_0.6b_openresearcher_sft \
DATA_PACKED_JSONL=examples/openresearcher/data/converted_gpt_oss_search_correct.packed_30000.jsonl \
MAX_SEQ=30000 \
GBS=8 \
TRAIN_SAMPLES=20872 \
bash examples/openresearcher/train_minimal.sh Qwen/Qwen3-0.6B
```

按实际样本数调整 `TRAIN_SAMPLES`（例如 materialized 行数或目标步数×GBS）。

### 5) 转回 HF

```bash
cd examples/openresearcher

TP=1 PP=1 EP=1 ETP=1 CP=1 DP=1 \
MLM_MODEL_CKPT=../../checkpoints/qwen3_0.6b_openresearcher_sft \
HF_MODEL_CKPT=Qwen/Qwen3-0.6B \
EXPORT_DIR=../../checkpoints/qwen3_0.6b_openresearcher_sft_hf \
MLM_SKIP_INSTALL=1 \
bash convert_to_hf.sh Qwen/Qwen3-0.6B
```

---

## 路径速查

| 阶段 | 50 条示例 | 完整数据 |
|------|------------|----------|
| 原始数据 | `data/converted_gpt_oss_search_correct.example.jsonl` | `data/converted_gpt_oss_search_correct.jsonl` |
| Materialized | `data/converted_gpt_oss_search_correct.example.materialized.jsonl` | `data/converted_gpt_oss_search_correct.materialized.jsonl` |
| Packed | `data/converted_gpt_oss_search_correct.example.packed_30000.jsonl` | `data/converted_gpt_oss_search_correct.packed_30000.jsonl` |
| Megatron 初始 | `checkpoints/qwen3_0.6b_init_mlm`（共用） | 同上 |
| SFT 结果 | `checkpoints/qwen3_0.6b_example_sft` | `checkpoints/qwen3_0.6b_openresearcher_sft` |
| HF 导出 | `checkpoints/qwen3_0.6b_example_sft_hf` | `checkpoints/qwen3_0.6b_openresearcher_sft_hf` |

以上路径均相对于 `examples/openresearcher`（data）或仓库根（checkpoints）。
