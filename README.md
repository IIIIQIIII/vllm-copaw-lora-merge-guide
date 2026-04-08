# vLLM Deployment & LoRA Merge Guide for CoPaw-Flash-9B

**Updated 2026-04-08**: Now includes the **correct physical merge method** that actually works! Plus comprehensive comparison between Base model, dynamic LoRA, and merged model approaches.

## Overview

This guide documents:
- **Two deployment methods**: Dynamic LoRA loading vs Physical merge
- Why ms-swift's default merge fails (triple-nested keys issue)
- **The correct way to merge LoRA** for Qwen3.5 VLM models
- Real-world performance comparison across 30+ datasets
- Base Model (Qwen3.5-9B) vs Agent-LoRA quantitative analysis

---

## TL;DR - Two Valid Options

### Option A: Dynamic LoRA Loading (Recommended for flexibility)

```bash
vllm serve /path/to/CoPaw-Flash-9B \
  --enable-lora \
  --lora-modules agent-lora=/path/to/CoPaw-Flash-9B-Agent-LoRA \
  --max-lora-rank 64 \
  --tensor-parallel-size 2 \
  --gdn-prefill-backend triton \
  --trust-remote-code
```

### Option B: Physical Merge (For frameworks without dynamic LoRA support)

Use `merge_lora_correct.py` from this repo - see [Correct Merge Method](#correct-merge-method) below.

---

## Environment

- **Hardware**: 2x NVIDIA H200 GPUs, 88 vCPU, 358GB RAM
- **Model**: CoPaw-Flash-9B (Qwen3.5-9B multimodal architecture)
- **LoRA**: Agent-LoRA (trained on 3,280 agent trajectories)
- **Framework**: vLLM 0.19.0, Transformers 5.5.0, PEFT 0.18.1
- **Python**: 3.11

---

## Why ms-swift Default Merge Fails

### The Problem

When using ms-swift's default merge, the output has **triple-nested keys**:

```
Expected: model.language_model.layers.0.mlp.down_proj.weight
Actual:   model.language_model.language_model.language_model.layers.0.mlp.down_proj.weight
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ (3x nested!)
```

### Root Cause Analysis

After deep investigation, we found:

1. **Qwen3.5 is natively a VLM** (Vision-Language Model)
   - Architecture: `Qwen3_5ForConditionalGeneration`
   - Has both `model.language_model.*` and `model.visual.*` branches

2. **LoRA trained on correct structure**
   - Target modules: `model.language_model.layers.*`
   - Weights correctly reference VLM structure

3. **The bug is in `model.save_pretrained()`**
   - PEFT's `merge_and_unload()` works correctly
   - `model.state_dict()` has correct keys
   - But `save_pretrained()` produces triple-nested keys!

4. **ms-swift's `model = model.model` doesn't cause the issue**
   - Initially suspected, but not the root cause
   - The issue is in HuggingFace's save logic for VLM models

---

## Correct Merge Method

### The Solution

Bypass `save_pretrained()` and manually save from `state_dict()`:

```python
# merge_lora_correct.py
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from peft import PeftModel
import safetensors.torch as st
import os
import json

BASE_PATH = "/path/to/CoPaw-Flash-9B"
LORA_PATH = "/path/to/CoPaw-Flash-9B-Agent-LoRA"
OUTPUT_PATH = "/path/to/CoPaw-Flash-9B-Agent-Merged"

# Step 1: Load with correct VLM architecture
print("Loading base model (VLM architecture)...")
base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Step 2: Load and merge LoRA
print("Loading and merging LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()

# Step 3: Get state_dict (keys are CORRECT here!)
state_dict = model.state_dict()

# Verify no triple nesting in state_dict
assert not any('language_model.language_model.language_model' in k for k in state_dict.keys()), \
    "Triple nesting detected in state_dict!"

# Step 4: Manually save (bypass save_pretrained bug)
print("Saving merged model...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Save weights in shards
shard_size = 5 * 1024 * 1024 * 1024  # 5GB
current_shard = {}
current_size = 0
shard_idx = 1
weight_map = {}

for key, tensor in state_dict.items():
    tensor_size = tensor.numel() * tensor.element_size()

    if current_size + tensor_size > shard_size and current_shard:
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        st.save_file(current_shard, f"{OUTPUT_PATH}/{shard_name}")
        shard_idx += 1
        current_shard = {}
        current_size = 0

    current_shard[key] = tensor.cpu()
    current_size += tensor_size
    weight_map[key] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"

# Save last shard
if current_shard:
    shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    st.save_file(current_shard, f"{OUTPUT_PATH}/{shard_name}")

# Rename files with correct total count
total_shards = shard_idx
for i in range(1, total_shards + 1):
    old = f"{OUTPUT_PATH}/model-{i:05d}-of-XXXXX.safetensors"
    new = f"{OUTPUT_PATH}/model-{i:05d}-of-{total_shards:05d}.safetensors"
    os.rename(old, new)
    for key in weight_map:
        if weight_map[key] == f"model-{i:05d}-of-XXXXX.safetensors":
            weight_map[key] = f"model-{i:05d}-of-{total_shards:05d}.safetensors"

# Save index file
index = {
    "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())},
    "weight_map": weight_map
}
with open(f"{OUTPUT_PATH}/model.safetensors.index.json", 'w') as f:
    json.dump(index, f, indent=2)

# Copy config files
import shutil
for f in ['config.json', 'generation_config.json']:
    if os.path.exists(f"{BASE_PATH}/{f}"):
        shutil.copy(f"{BASE_PATH}/{f}", f"{OUTPUT_PATH}/{f}")

# Save processor
processor = AutoProcessor.from_pretrained(BASE_PATH, trust_remote_code=True)
processor.save_pretrained(OUTPUT_PATH)

print(f"Done! Merged model saved to: {OUTPUT_PATH}")
```

### Verification

```python
import safetensors.torch as st

# Check for triple nesting (should be 0)
with st.safe_open("merged_model/model-00001-of-00004.safetensors", framework='pt') as f:
    triple_nested = [k for k in f.keys() if 'language_model.language_model.language_model' in k]
    print(f"Triple nested keys: {len(triple_nested)}")  # Should be 0!

# Verify LoRA was actually merged
with st.safe_open("base_model/model.safetensors", framework='pt') as base_f:
    with st.safe_open("merged_model/model-00001-of-00004.safetensors", framework='pt') as merged_f:
        key = "model.language_model.layers.0.mlp.down_proj.weight"
        diff = (merged_f.get_tensor(key).float() - base_f.get_tensor(key).float()).abs()
        print(f"Weight diff: mean={diff.mean():.8f}, max={diff.max():.6f}")
        # Should show non-zero diff (LoRA was merged!)
```

---

## Dynamic LoRA Loading (Alternative)

If you prefer not to merge, vLLM's dynamic LoRA loading works perfectly:

### Setup

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/CoPaw-Flash-9B \
  --enable-lora \
  --lora-modules agent-lora=/path/to/CoPaw-Flash-9B-Agent-LoRA \
  --max-lora-rank 64 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --gdn-prefill-backend triton \
  --trust-remote-code \
  --port 8000
```

### Usage

```python
import requests

# Use LoRA adapter
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "agent-lora",  # Specify LoRA adapter
    "messages": [{"role": "user", "content": "Analyze this dataset..."}],
})

# Or use base model
response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "/path/to/CoPaw-Flash-9B",  # Base model
    "messages": [{"role": "user", "content": "..."}],
})
```

---

## Performance Comparison

### Quantitative Analysis (30 datasets)

We tested **Qwen3.5-9B (Base)** vs **CoPaw+Agent-LoRA** on 30 real Kaggle datasets:

| Metric | Qwen3.5-9B (Base) | CoPaw+Agent-LoRA | Difference |
|--------|-------------------|------------------|------------|
| **Avg iterations** | 1.2 | 26.0 | **+24.8** |
| **Avg tokens/dataset** | 4,230 | 637,467 | **+150x** |
| **Natural completion rate** | 0%* | 89.7% | **+89.7pp** |
| **Datasets with output** | 0/16 (0%) | 28/29 (96.6%) | **+96.6pp** |
| **Python files generated** | 0 | 89 | **+89** |
| **PNG charts generated** | 0 | 143 | **+143** |
| **Total files generated** | 0 | 275 | **+275** |

*Base model stops after 1-2 iterations without producing any analysis output.

### Key Finding

**Agent-LoRA training is ESSENTIAL, not optional!**

The base Qwen3.5-9B model:
- Understands tool calling format ✓
- Can execute single tool calls ✓
- **Cannot work autonomously** ✗
- **Stops after 1-2 iterations** ✗
- **Produces zero analysis output** ✗

The Agent-LoRA model:
- Works autonomously for 26 iterations average ✓
- Produces complete data analysis ✓
- Generates scripts, charts, and reports ✓
- 96.6% of datasets get usable output ✓

### Tool Usage Patterns

| Tool | Qwen3.5-9B (Base) | CoPaw+Agent-LoRA |
|------|-------------------|------------------|
| Bash | 72% (18 calls) | 50% (375 calls) |
| Write | **0%** (0 calls) | 32% (239 calls) |
| Edit | **0%** (0 calls) | 11% (79 calls) |
| Read | 8% (2 calls) | 7% (50 calls) |
| Glob | 20% (5 calls) | 1% (10 calls) |

Base model only "looks" (ls, glob) but never "does" (write, edit).

---

## Troubleshooting

### Issue: Triple-nested keys in merged model

**Cause**: Using `model.save_pretrained()` for VLM models.

**Solution**: Use the manual save method in `merge_lora_correct.py`.

### Issue: PEFT "Target modules not found"

**Cause**: Loading model with wrong class (e.g., `AutoModelForCausalLM` instead of VLM class).

**Solution**:
```python
# Wrong
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)  # Loads as CausalLM

# Correct
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(...)  # Loads as VLM
```

### Issue: FlashInfer compilation errors with vLLM

**Cause**: FlashInfer JIT compilation issues on some CUDA versions.

**Solution**: Add `--gdn-prefill-backend triton` to vLLM command.

### Issue: Model loads slowly / GPU memory issues

**Solutions**:
- Reduce `--max-model-len` (try 32768 instead of 131072)
- Adjust `--gpu-memory-utilization` (try 0.85)
- Use `--tensor-parallel-size 2` for multi-GPU

---

## Files in This Repository

| File | Description |
|------|-------------|
| `README.md` | This guide |
| `merge_lora_correct.py` | **Correct LoRA merge script** (bypasses save_pretrained bug) |
| `fix_merged_model_keys.py` | Legacy: fix triple-nested keys (no longer needed) |
| `restore_visual_weights.py` | Legacy: restore lost visual weights |
| `convert_to_bfloat16.py` | Legacy: convert dtype |
| `LEGACY_MERGE_ISSUES.md` | Detailed documentation of ms-swift merge failures |

---

## Summary

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| **Dynamic LoRA** | Easy setup, switch adapters on-the-fly | Slight runtime overhead | Development, testing, multiple adapters |
| **Physical Merge** | Single model file, works everywhere | One-time merge needed | Production, frameworks without LoRA support |
| **ms-swift default** | - | **Broken** (triple nesting) | Never |

---

## References

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Qwen3.5 Model Card](https://huggingface.co/Qwen/Qwen3.5-9B)

---

## Contributing

Issues and PRs welcome! If you find other merge issues or solutions, please share.

## License

MIT License
