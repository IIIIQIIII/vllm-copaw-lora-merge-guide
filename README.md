# vLLM Deployment and LoRA Merge Guide for CoPaw-Flash-9B

A comprehensive guide on deploying CoPaw-Flash-9B models with vLLM and merging LoRA adapters, based on real-world experience with ms-swift.

## Overview

This guide documents the complete process of:
- Setting up Python environment with correct package installation order
- Deploying CoPaw-Flash-9B (Qwen3.5-9B based) with vLLM on dual H200 GPUs
- Merging LoRA adapters using ms-swift
- Troubleshooting and fixing common issues with multimodal model merging

## Environment

- **Hardware**: 2x NVIDIA H200 GPUs, 88 vCPU, 358GB RAM
- **Model**: CoPaw-Flash-9B (Qwen3.5-9B hybrid architecture: Gated DeltaNet + Gated Attention)
- **Framework**: vLLM 0.19.0, Transformers 5.x, ms-swift
- **Python**: 3.11

## Critical Package Installation Order

⚠️ **IMPORTANT**: Install vLLM FIRST, then transformers. Wrong order causes inference issues with new architectures.

```bash
# 1. Create uv environment
cd /home/shadeform
cat > pyproject.toml << 'EOF'
[project]
name = "shadeform"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "vllm>=0.19.0",
    "transformers>=5.0",
]
EOF

# 2. Install vLLM first (nightly for latest features)
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly

# 3. Install transformers 5.x (for Qwen3.5 support)
uv pip install --upgrade transformers

# 4. Install CUDA runtime
uv pip install nvidia-cuda-runtime-cu12
```

## Successful vLLM Deployment

### Base Model Deployment

```bash
# Set CUDA library path
CUDA12_LIB=$(python -c "import nvidia.cuda_runtime; import os; print(os.path.dirname(nvidia.cuda_runtime.__file__))")/lib

# Deploy with vLLM
LD_LIBRARY_PATH=$CUDA12_LIB:$LD_LIBRARY_PATH vllm serve ./models/CoPaw-Flash-9B \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --gdn-prefill-backend triton \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code
```

**Key Parameters**:
- `--tensor-parallel-size 2`: Use both H200 GPUs
- `--max-model-len 131072`: Half of model's max 262144 context (fits in memory)
- `--gdn-prefill-backend triton`: Use Triton instead of FlashInfer (avoids JIT compilation issues)
- `--gpu-memory-utilization 0.85`: Leave headroom for operations

## LoRA Merge with ms-swift

### Issues Discovered with ms-swift

When merging multimodal models, ms-swift has three critical issues:

| Issue | Symptom | Impact | Solution |
|-------|---------|--------|----------|
| **Triple-nested keys** | `model.language_model.*` becomes `model.language_model.language_model.language_model.*` | vLLM fails to load | Remap weight keys with script |
| **Missing visual weights** | Only merges LoRA target_modules, loses 333 visual weights | Multimodal features broken | Restore visual weights from base model |
| **dtype becomes float32** | Model changes from bfloat16 to float32 | Model size doubles (18GB→36GB) | Convert back to bfloat16 |

### Complete Fix Pipeline

#### Step 1: Merge with ms-swift

```bash
# Install dependencies
uv pip install "qwen_vl_utils>=0.0.14" decord -U

# Fix generation_config.json validation error
# Add "do_sample": true to generation_config.json

# Merge LoRA
swift export \
    --model ./models/CoPaw-Flash-9B \
    --adapters ./models/CoPaw-Flash-9B-Agent-LoRA \
    --merge_lora true \
    --output_dir ./models/CoPaw-Flash-9B-Agent-Merged
```

#### Step 2: Fix Weight Key Names

Create `fix_merged_model_keys.py`:

```python
import json
from pathlib import Path
import safetensors.torch

def fix_key_name(key: str) -> str:
    """Fix triple-nested keys"""
    if key.startswith('model.language_model.language_model.language_model.'):
        return key.replace(
            'model.language_model.language_model.language_model.',
            'model.language_model.',
            1
        )
    return key

def fix_merged_model(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each shard
    for input_file in sorted(input_path.glob('model-*.safetensors')):
        tensors = {}
        with safetensors.safe_open(str(input_file), framework='pt') as f:
            for key in f.keys():
                new_key = fix_key_name(key)
                tensors[new_key] = f.get_tensor(key)

        output_file = output_path / input_file.name
        safetensors.torch.save_file(tensors, str(output_file))

    # Update index file
    with open(input_path / 'model.safetensors.index.json', 'r') as f:
        index_data = json.load(f)

    new_weight_map = {}
    for key, shard in index_data['weight_map'].items():
        new_weight_map[fix_key_name(key)] = shard
    index_data['weight_map'] = new_weight_map

    with open(output_path / 'model.safetensors.index.json', 'w') as f:
        json.dump(index_data, f, indent=2)

    # Copy config files
    import shutil
    for config_file in ['config.json', 'generation_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja']:
        src = input_path / config_file
        if src.exists():
            shutil.copy2(src, output_path / config_file)

if __name__ == '__main__':
    fix_merged_model(
        './models/CoPaw-Flash-9B-Agent-Merged',
        './models/CoPaw-Flash-9B-Agent-Merged-Fixed'
    )
```

#### Step 3: Restore Visual Weights

Create `restore_visual_weights.py`:

```python
import json
from pathlib import Path
import safetensors.torch

def restore_visual_weights(base_model_dir: str, merged_dir: str, output_dir: str):
    base_path = Path(base_model_dir)
    merged_path = Path(merged_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract visual weights and lm_head from base model
    base_tensors = {}
    with safetensors.safe_open(str(base_path / 'model.safetensors'), framework='pt') as f:
        for key in f.keys():
            if key.startswith('model.visual') or key == 'lm_head.weight':
                base_tensors[key] = f.get_tensor(key)

    # Load language_model weights from merged model
    merged_tensors = {}
    for i in range(1, 9):
        shard_file = merged_path / f'model-0000{i}-of-00008.safetensors'
        with safetensors.safe_open(str(shard_file), framework='pt') as f:
            for key in f.keys():
                if key.startswith('model.language_model'):
                    merged_tensors[key] = f.get_tensor(key)

    # Combine all weights
    all_tensors = {**base_tensors, **merged_tensors}

    # Re-shard and save (8 shards)
    shards = [[] for _ in range(8)]

    # lm_head in first shard
    if 'lm_head.weight' in all_tensors:
        shards[0].append('lm_head.weight')

    # Distribute other weights
    other_keys = [k for k in sorted(all_tensors.keys()) if k != 'lm_head.weight']
    keys_per_shard = len(other_keys) // 7

    shard_idx = 1
    for i, key in enumerate(other_keys):
        shards[shard_idx].append(key)
        if len(shards[shard_idx]) >= keys_per_shard and shard_idx < 7:
            shard_idx += 1

    # Save shards
    weight_map = {}
    for i, shard_keys in enumerate(shards):
        if not shard_keys:
            continue

        shard_num = i + 1
        shard_file = output_path / f'model-0000{shard_num}-of-00008.safetensors'
        shard_tensors = {k: all_tensors[k] for k in shard_keys}
        safetensors.torch.save_file(shard_tensors, str(shard_file))

        for key in shard_keys:
            weight_map[key] = f'model-0000{shard_num}-of-00008.safetensors'

    # Update index
    with open(merged_path / 'model.safetensors.index.json', 'r') as f:
        index_data = json.load(f)

    index_data['weight_map'] = weight_map
    index_data['metadata'] = {'total_size': sum(t.nbytes for t in all_tensors.values())}

    with open(output_path / 'model.safetensors.index.json', 'w') as f:
        json.dump(index_data, f, indent=2)

    # Copy config files
    import shutil
    for config_file in ['config.json', 'generation_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'processor_config.json', 'chat_template.jinja']:
        src = merged_path / config_file
        if src.exists():
            shutil.copy2(src, output_path / config_file)

if __name__ == '__main__':
    restore_visual_weights(
        './models/CoPaw-Flash-9B',
        './models/CoPaw-Flash-9B-Agent-Merged-Fixed',
        './models/CoPaw-Flash-9B-Agent-Complete'
    )
```

#### Step 4: Convert to bfloat16

⚠️ **Critical**: ms-swift saves in float32, doubling model size!

Create `convert_to_bfloat16.py`:

```python
import json
from pathlib import Path
import safetensors.torch
import torch

def convert_to_bfloat16(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_file in sorted(input_path.glob('model-*.safetensors')):
        output_file = output_path / input_file.name

        tensors_bf16 = {}
        with safetensors.safe_open(str(input_file), framework='pt') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.float32:
                    tensors_bf16[key] = tensor.to(torch.bfloat16)
                else:
                    tensors_bf16[key] = tensor

        safetensors.torch.save_file(tensors_bf16, str(output_file))

    # Copy config files
    import shutil
    for config_file in ['config.json', 'generation_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'processor_config.json',
                        'chat_template.jinja', 'model.safetensors.index.json']:
        src = input_path / config_file
        if src.exists():
            shutil.copy2(src, output_path / config_file)

if __name__ == '__main__':
    convert_to_bfloat16(
        './models/CoPaw-Flash-9B-Agent-Complete',
        './models/CoPaw-Flash-9B-Agent-Complete-BF16'
    )
```

**Results**:
- Before: 35GB (float32)
- After: 19GB (bfloat16)
- **Saved: 16GB (46%)**

#### Step 5: Deploy Optimized Model

```bash
CUDA12_LIB=$(python -c "import nvidia.cuda_runtime; import os; print(os.path.dirname(nvidia.cuda_runtime.__file__))")/lib

LD_LIBRARY_PATH=$CUDA12_LIB:$LD_LIBRARY_PATH vllm serve ./models/CoPaw-Flash-9B-Agent-Complete-BF16 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --gdn-prefill-backend triton \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code
```

### Verification

```bash
# Check model loaded
curl http://localhost:8000/v1/models

# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./models/CoPaw-Flash-9B-Agent-Complete-BF16",
    "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
    "max_tokens": 100
  }'
```

**Success indicators**:
- ✅ Model size: 19GB (bfloat16)
- ✅ No weight loading warnings
- ✅ Response contains `"reasoning"` field (Agent LoRA capability)
- ✅ Normal chat inference

## Root Cause Analysis

### Why Does ms-swift Create Triple-Nested Keys?

1. **CoPaw-Flash-9B Architecture**: Multimodal model `Qwen3_5ForConditionalGeneration`
   - Contains `language_model` (Qwen3_5TextModel) + `visual` (Qwen2VLVisionModel)
   - Base weights: `model.language_model.layers.X.*`

2. **LoRA Adapter Configuration**:
   ```json
   "target_modules": "^(model\\.language_model(?=\\.).*\\.(down_proj|out_proj|...)$"
   ```
   - Only targets `model.language_model.*` path
   - Deliberately excludes visual module

3. **ms-swift Loading Process** (inferred):
   - Loads multimodal model with wrapper(s)
   - Each wrapper adds `model.language_model.` prefix
   - Multiple wrappings → triple nesting

### Why Are Visual Weights Missing?

- **LoRA only trained language_model**: LoRA adapter only contains deltas for `model.language_model.*` weights
- **ms-swift only merges target_modules**: Only processes weights matching the regex
- **Result**: 333 visual weights completely dropped during export

### Why float32 Instead of bfloat16?

- **ms-swift default behavior**: Saves merged weights in float32
- **No dtype preservation**: Doesn't check or preserve base model's dtype
- **Impact**: Model size doubles unnecessarily

## Lessons Learned

1. **Package installation order matters**: vLLM → transformers (not vice versa)
2. **Immediately verify merge results**:
   - Check weight key names match base model
   - Verify all modules present (language_model + visual)
   - Confirm dtype matches base model
   - Check total model size is reasonable
3. **dtype is critical**: Always convert back to base model's dtype after merging
4. **Multimodal model complexity**: Merge tools may not handle nested structures correctly
5. **Consider native tools**: For standard PEFT/LoRA, `peft.merge_and_unload()` may be more reliable
6. **Architecture support**: Ensure merge tools support your model architecture (Qwen3.5 is new)

## Common Errors and Solutions

### Error: `libcudart.so.12 not found`
```bash
uv pip install nvidia-cuda-runtime-cu12
```

### Error: FlashInfer JIT compilation failures
```bash
# Add flag: --gdn-prefill-backend triton
```

### Error: GPU memory insufficient
```bash
# Use both GPUs: --tensor-parallel-size 2
# Reduce context: --max-model-len 131072
# Lower utilization: --gpu-memory-utilization 0.85
```

### Error: `do_sample is not set to True`
```json
// Add to generation_config.json:
{
  "do_sample": true,
  "temperature": 1.0,
  "top_p": 0.95,
  // ... rest of config
}
```

### Error: Weight keys not found during loading
- Symptom: `Parameter language_model.language_model.* not found`
- Cause: Triple-nested keys from ms-swift
- Solution: Run `fix_merged_model_keys.py`

### Error: Visual weights not initialized
- Symptom: `Following weights were not initialized: visual.blocks.*`
- Cause: ms-swift dropped visual weights
- Solution: Run `restore_visual_weights.py`

## Model Comparison

| Model | Size | dtype | Visual Weights | Status |
|-------|------|-------|----------------|--------|
| CoPaw-Flash-9B (base) | 18GB | bfloat16 | ✅ 333 weights | ✅ Deployed |
| CoPaw-Flash-9B-Agent-Merged | 36GB | float32 | ❌ Missing | ❌ Failed |
| CoPaw-Flash-9B-Agent-Complete | 35GB | float32 | ✅ 333 weights | ⚠️ Works but large |
| CoPaw-Flash-9B-Agent-Complete-BF16 | 19GB | bfloat16 | ✅ 333 weights | ✅ Optimal |

## Tool Versions

```
vllm==0.19.0
transformers==5.5.0
ms-swift (from main branch, Jan 2026)
torch==2.5.1+cu121
safetensors==0.4.5
python==3.11
CUDA==12.1
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [ms-swift Documentation](https://github.com/modelscope/ms-swift)
- [CoPaw Model Card](https://huggingface.co/jason1966/CoPaw-Flash-9B)
- [Qwen3.5 Architecture](https://qwenlm.github.io/)

## License

This guide is provided as-is for educational purposes. Model and tool licenses apply to their respective components.
