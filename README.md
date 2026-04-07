# vLLM Deployment with Dynamic LoRA for CoPaw-Flash-9B

**Updated**: A practical guide based on real-world deployment experience. This document reflects lessons learned from ms-swift merge failures and the successful alternative: vLLM's dynamic LoRA loading.

## Overview

This guide documents:
- Why ms-swift LoRA merge fails for CoPaw-Flash-9B
- How to deploy CoPaw-Flash-9B with vLLM + dynamic LoRA loading
- Performance comparison: Base Model vs LoRA Agent
- Best practices for agent trajectory collection

## TL;DR - The Solution

**Don't merge LoRA adapters**. Use vLLM's dynamic LoRA loading instead:

```bash
vllm serve /path/to/CoPaw-Flash-9B \
  --enable-lora \
  --lora-modules agent-lora=/path/to/CoPaw-Flash-9B-Agent-LoRA \
  --max-lora-rank 64 \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

---

## Environment

- **Hardware**: 2x NVIDIA H200 GPUs, 88 vCPU, 358GB RAM
- **Model**: CoPaw-Flash-9B (Qwen3.5-9B multimodal architecture)
- **LoRA**: Agent-LoRA (trained on 3,280 agent trajectories)
- **Framework**: vLLM 0.19.0, Transformers 5.5.0
- **Python**: 3.11

---

## Why ms-swift Merge Failed

### The Problem

When attempting to merge Agent-LoRA with CoPaw-Flash-9B using ms-swift, we discovered **the LoRA weights were never actually merged**:

```python
# Verification shows merged model = base model (identical weights!)
with safetensors.safe_open('base_model.safetensors', framework='pt') as base_f:
    with safetensors.safe_open('merged_model.safetensors', framework='pt') as merged_f:
        base_tensor = base_f.get_tensor('model.language_model.layers.0.input_layernorm.weight')
        merged_tensor = merged_f.get_tensor('model.language_model.layers.0.input_layernorm.weight')

        diff = (base_tensor != merged_tensor).sum().item()
        # Result: diff = 0 (100% identical!)
```

### Root Cause

ms-swift export with `merge_lora=True` fails silently for multimodal models:
1. Loads base model weights ✓
2. Attempts LoRA merge ✗ (fails silently)
3. Saves weights as if merge succeeded ✓
4. Also creates triple-nested keys, loses visual weights, uses wrong dtype

**Result**: Merged model performs identically to base model because it IS the base model.

---

## The Solution: vLLM Dynamic LoRA Loading

### Setup Instructions

#### 1. Install Dependencies (Correct Order!)

⚠️ **Critical**: Install vLLM BEFORE transformers to avoid architecture inference issues.

```bash
# Create environment
cd /home/shadeform

# Install vLLM first
uv pip install vllm --torch-backend=auto

# Then install/upgrade transformers
uv pip install --upgrade transformers

# Add CUDA runtime
uv pip install nvidia-cuda-runtime-cu12
```

Verify transformers version supports TokenizersBackend:
```bash
python -c "import transformers; print(transformers.__version__)"
# Should be >= 5.5.0
```

#### 2. Start vLLM with LoRA

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model /path/to/CoPaw-Flash-9B \
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

**Key Parameters**:
- `--enable-lora`: Enable dynamic LoRA support
- `--lora-modules`: Format is `name=/path/to/adapter`
- `--max-lora-rank 64`: Must match LoRA adapter rank (check adapter_config.json)
- `--tensor-parallel-size 2`: Use both GPUs
- `--gdn-prefill-backend triton`: Avoid FlashInfer JIT compilation issues

#### 3. Verify LoRA is Loaded

```bash
curl http://localhost:8000/v1/models | jq .
```

Expected output:
```json
{
  "data": [
    {
      "id": "/path/to/CoPaw-Flash-9B",
      "object": "model",
      ...
    },
    {
      "id": "agent-lora",
      "object": "model",
      "parent": "/path/to/CoPaw-Flash-9B",
      ...
    }
  ]
}
```

#### 4. Use LoRA in API Calls

Specify `model: "agent-lora"` to use the LoRA adapter:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "agent-lora",  # Use LoRA adapter
        "messages": [{"role": "user", "content": "Analyze this dataset..."}],
        "max_tokens": 512,
    }
)
```

Or use base model:
```python
{
    "model": "/path/to/CoPaw-Flash-9B",  # Base model without LoRA
    ...
}
```

---

## Performance Comparison

### Verification Test

We tested both models with the same prompt: "Analyze the Iris dataset..."

**Base Model Response** (494 chars):
```
The user wants me to analyze the Iris dataset...
</think>

I'll analyze the Iris dataset step by step...
```

**LoRA Agent Response** (1,985 chars - 4x longer!):
```
We are going to work with the Iris dataset. Let's start...
Steps:
1. Load the data from CSV
2. Explore structure
3. Identify patterns
4. Create visualizations
5. Provide insights
</think>

I'll analyze the Iris dataset step by step, loading the data, exp...
```

✅ **LoRA is working!** The agent produces significantly more detailed responses.

### Autonomous Agent Test (15 iterations)

Testing on cardiovascular disease dataset:

| Metric | Base Model | LoRA Agent | Difference |
|--------|-----------|------------|------------|
| Time | 36.9s | 56.0s | +52% slower |
| Input tokens | 53,567 | 74,033 | +38% more |
| Output tokens | 6,234 | 7,695 | +23% more |
| Avg think length | 122.9 chars | 137.8 chars | +12% longer |
| Tool calls | 15 | 15 | Same |

**Generated Code**:
- Both created 7 Python scripts (867 lines total)
- 4/7 scripts run successfully (57%)
- 3/7 have syntax errors (43%)
- Code quality similar between models

**Tool Usage Patterns**:
- Base: 73% Bash commands, more command-line focused
- LoRA: 53% Bash, more Write operations, creates files proactively

---

## Recommendation

### When to Use Each

**Use Base Model** if:
- ✅ Speed matters (52% faster)
- ✅ Cost matters (38% fewer tokens)
- ✅ Short tasks (<15 iterations)
- ✅ Raw agent capability is sufficient

**Use LoRA Agent** if:
- ✅ Need detailed reasoning traces (for training data)
- ✅ Complex, long tasks (>30 iterations)
- ✅ Value quality over speed
- ✅ Want more proactive file creation

### Honest Assessment

In our 15-iteration test, **the difference was smaller than expected**.

Reasons:
1. Qwen3.5 base model already has strong agent capabilities
2. Both models produce `<think>` tags and can plan
3. Short tasks don't fully leverage LoRA's advantages
4. Code quality and error rates are similar

**For trajectory collection**: Use LoRA Agent for richer training data, despite higher cost.

---

## Troubleshooting

### Issue: "Tokenizer class TokenizersBackend does not exist"

**Cause**: transformers version < 5.5.0 doesn't support TokenizersBackend.

**Solution**:
```bash
uv pip install --upgrade transformers --force-reinstall
# Verify: should be >= 5.5.0
python -c "import transformers; print(transformers.__version__)"
```

### Issue: LoRA not showing in /v1/models

**Check**:
1. `--enable-lora` flag present
2. Path to LoRA adapter is correct
3. `--max-lora-rank` matches adapter rank (check adapter_config.json: `"r": 64`)

### Issue: Model loading slow or fails

**Solutions**:
- Reduce `--max-model-len` (try 131072 instead of 262144)
- Use `--gdn-prefill-backend triton` to avoid FlashInfer compilation
- Adjust `--gpu-memory-utilization` (try 0.85 or 0.80)

---

## Key Lessons Learned

1. **Don't trust silent failures**: ms-swift merge appeared successful but didn't merge LoRA weights
2. **Verify merges**: Always check if merged weights differ from base weights
3. **Dynamic LoRA > Merge**: vLLM's dynamic loading is more reliable for complex models
4. **Base models are strong**: Modern base models (Qwen3.5) have significant agent capabilities
5. **Test thoroughly**: Small improvements may not justify higher cost/slower speed

---

## Files in This Repository

- `README.md` - This document (updated approach)
- `fix_merged_model_keys.py` - Legacy: fix triple-nested keys from failed merge
- `restore_visual_weights.py` - Legacy: restore lost visual weights
- `convert_to_bfloat16.py` - Legacy: convert dtype from failed merge
- `LEGACY_MERGE_ISSUES.md` - Detailed documentation of ms-swift merge failures

**Note**: The fix scripts are kept for reference but are no longer needed with dynamic LoRA loading.

---

## References

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [CoPaw-Flash-9B Model](https://huggingface.co/CoPaw-AI/CoPaw-Flash-9B)
- [Agent-LoRA Adapter](https://huggingface.co/CoPaw-AI/CoPaw-Flash-9B-Agent-LoRA)

---

## Contributing

If you encounter issues or have improvements, please open an issue or PR.

## License

MIT License - Feel free to use and modify
