# Legacy: ms-swift LoRA Merge Issues (Archived)

This document archives the issues encountered when attempting to merge LoRA adapters with ms-swift. **This approach is no longer recommended** - use vLLM's dynamic LoRA loading instead (see main README).

## The Three Issues with ms-swift Merge

When merging CoPaw-Flash-9B with Agent-LoRA using ms-swift, we encountered three critical problems:

### Issue 1: LoRA Weights Not Merged (Most Critical)

**Problem**: ms-swift export with `merge_lora=True` failed silently. The merged model had identical weights to the base model.

**Verification**:
```python
import safetensors.torch as st

base = st.safe_open('CoPaw-Flash-9B/model.safetensors', framework='pt')
merged = st.safe_open('merged/model-00002-of-00008.safetensors', framework='pt')

key = 'model.language_model.layers.0.input_layernorm.weight'
base_tensor = base.get_tensor(key)
merged_tensor = merged.get_tensor(key)

diff = (base_tensor != merged_tensor).sum().item()
# Result: 0 (completely identical!)
```

**Root Cause**: ms-swift's merge logic doesn't handle Qwen3.5ForConditionalGeneration models correctly. The export succeeds but LoRA weights are never applied.

### Issue 2: Triple-Nested Weight Keys

**Problem**: Merged model created keys like:
```
model.language_model.language_model.language_model.embed_tokens.weight
```
Instead of:
```
model.language_model.embed_tokens.weight
```

**Fix Script** (no longer needed):
```python
# fix_merged_model_keys.py
def fix_key_name(key: str) -> str:
    if key.startswith('model.language_model.language_model.language_model.'):
        return key.replace(
            'model.language_model.language_model.language_model.',
            'model.language_model.',
            1
        )
    return key
```

### Issue 3: Missing Visual Weights

**Problem**: Merge only preserved language_model weights, dropping 333 visual weights:
- `visual.blocks.*.attn.*`
- `visual.blocks.*.mlp.*`
- All multimodal components

**Fix Script** (no longer needed):
```python
# restore_visual_weights.py
# Extracted visual weights from base model and merged with language_model weights
```

### Issue 4: Wrong dtype (float32 instead of bfloat16)

**Problem**: Merged model saved as float32 (36GB) instead of bfloat16 (19GB)

**Fix Script** (no longer needed):
```python
# convert_to_bfloat16.py
tensor_bf16 = tensor_fp32.to(torch.bfloat16)
```

## Why These Fixes Didn't Matter

Even after fixing all three issues, the merged model still performed identically to the base model because **Issue 1 was the real problem**: LoRA weights were never merged in the first place.

All the key remapping, weight restoration, and dtype conversion were applied to a model that was just the base model with extra problems.

## The ms-swift Command That Failed

```bash
cd ms-swift
swift export \
    --model_type qwen3-vl-chat \
    --model_id_or_path /path/to/CoPaw-Flash-9B \
    --adapters /path/to/CoPaw-Flash-9B-Agent-LoRA \
    --merge_lora true \
    --output_dir /path/to/output
```

This command:
1. ✓ Runs without errors
2. ✓ Creates output files
3. ✗ Doesn't actually merge LoRA
4. ✗ Creates broken weight structures
5. ✗ Loses visual components
6. ✗ Uses wrong dtype

## Lessons Learned

1. **Silent failures are dangerous**: ms-swift appeared to work but didn't
2. **Always verify merges**: Check if merged weights actually differ from base
3. **Multimodal models are tricky**: Standard merge tools may not handle them
4. **Dynamic loading is safer**: vLLM's approach avoids these issues entirely

## Why We Kept the Fix Scripts

The scripts are preserved in this repo as:
- Documentation of the debugging process
- Examples of weight manipulation with safetensors
- Reference for others encountering similar issues
- Educational value

But they should not be used - use vLLM dynamic LoRA instead.

## Migration Path

If you have a "merged" model from ms-swift:

1. **Stop using it** - it's just the base model with problems
2. **Use original base model** + **LoRA adapter** with vLLM dynamic loading
3. **Verify LoRA is working** by testing response quality

No need to fix or convert the old merged model.
