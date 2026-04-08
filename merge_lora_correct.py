#!/usr/bin/env python3
"""
Correct LoRA Merge Script for Qwen3.5 VLM Models

This script correctly merges LoRA adapters into Qwen3.5 VLM base models,
bypassing the save_pretrained() bug that causes triple-nested keys.

Problem: When using model.save_pretrained() for VLM models, the output has:
    model.language_model.language_model.language_model.layers.X.*
Instead of:
    model.language_model.layers.X.*

Solution: Manually save from state_dict() using safetensors.

Usage:
    python merge_lora_correct.py \
        --base-model /path/to/CoPaw-Flash-9B \
        --lora-adapter /path/to/CoPaw-Flash-9B-Agent-LoRA \
        --output /path/to/CoPaw-Flash-9B-Agent-Merged
"""

import os
import json
import argparse
import shutil
from pathlib import Path

import torch
import safetensors.torch as st
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correctly merge LoRA adapters into Qwen3.5 VLM models"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to base model (e.g., CoPaw-Flash-9B)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter (e.g., CoPaw-Flash-9B-Agent-LoRA)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=5.0,
        help="Maximum shard size in GB (default: 5.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[dtype_str]


def merge_and_save(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str,
    shard_size_gb: float = 5.0,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Merge LoRA adapter into base model and save with correct key structure.

    This function:
    1. Loads the base model as VLM (Qwen3_5ForConditionalGeneration)
    2. Loads and merges the LoRA adapter using PEFT
    3. Manually saves the state_dict to bypass save_pretrained() bug
    """

    # Step 1: Load base model with correct VLM architecture
    print(f"Loading base model from {base_model_path}...")
    print("  Using Qwen3_5ForConditionalGeneration (VLM architecture)")

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Base model loaded: {type(base_model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # Step 2: Load and merge LoRA adapter
    print(f"\nLoading LoRA adapter from {lora_adapter_path}...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    print("  LoRA merged successfully!")

    # Step 3: Get state_dict and verify no triple nesting
    print("\nExtracting state_dict...")
    state_dict = model.state_dict()

    # Verify keys are correct
    triple_nested = [k for k in state_dict.keys()
                    if 'language_model.language_model.language_model' in k]
    if triple_nested:
        raise RuntimeError(
            f"Triple nesting detected in state_dict! Found {len(triple_nested)} keys. "
            "This should not happen - please report this issue."
        )
    print(f"  State dict has {len(state_dict)} keys (no triple nesting)")

    # Step 4: Manually save in shards
    print(f"\nSaving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)

    shard_size = int(shard_size_gb * 1024 * 1024 * 1024)
    current_shard = {}
    current_size = 0
    shard_idx = 1
    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
            st.save_file(current_shard, f"{output_path}/{shard_name}")
            print(f"  Saved shard {shard_idx}: {len(current_shard)} tensors, {current_size / 1e9:.2f} GB")
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor.cpu()
        current_size += tensor_size
        weight_map[key] = f"model-{shard_idx:05d}-of-XXXXX.safetensors"

    # Save last shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        st.save_file(current_shard, f"{output_path}/{shard_name}")
        print(f"  Saved shard {shard_idx}: {len(current_shard)} tensors, {current_size / 1e9:.2f} GB")

    # Rename files with correct total count
    total_shards = shard_idx
    for i in range(1, total_shards + 1):
        old = f"{output_path}/model-{i:05d}-of-XXXXX.safetensors"
        new = f"{output_path}/model-{i:05d}-of-{total_shards:05d}.safetensors"
        os.rename(old, new)
        for key in weight_map:
            if weight_map[key] == f"model-{i:05d}-of-XXXXX.safetensors":
                weight_map[key] = f"model-{i:05d}-of-{total_shards:05d}.safetensors"

    # Save index file
    total_size = sum(t.numel() * t.element_size() for t in state_dict.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    with open(f"{output_path}/model.safetensors.index.json", 'w') as f:
        json.dump(index, f, indent=2)
    print(f"  Saved index file with {len(weight_map)} entries")

    # Step 5: Copy config files
    print("\nCopying config files...")
    config_files = [
        'config.json',
        'generation_config.json',
    ]
    for config_file in config_files:
        src = Path(base_model_path) / config_file
        if src.exists():
            shutil.copy(src, f"{output_path}/{config_file}")
            print(f"  Copied {config_file}")

    # Step 6: Save processor
    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    print("  Processor saved")

    print(f"\n{'='*60}")
    print(f"SUCCESS! Merged model saved to: {output_path}")
    print(f"Total shards: {total_shards}")
    print(f"Total size: {total_size / 1e9:.2f} GB")
    print(f"{'='*60}")

    return output_path


def verify_merged_model(merged_path: str, base_path: str = None):
    """
    Verify the merged model has correct structure and LoRA weights merged.
    """
    print(f"\nVerifying merged model at {merged_path}...")

    # Check for triple nesting
    shard_files = sorted(Path(merged_path).glob("model-*.safetensors"))
    if not shard_files:
        raise RuntimeError(f"No safetensor files found in {merged_path}")

    triple_nested_count = 0
    total_keys = 0

    with st.safe_open(str(shard_files[0]), framework='pt') as f:
        for key in f.keys():
            total_keys += 1
            if 'language_model.language_model.language_model' in key:
                triple_nested_count += 1

    print(f"  Checked {total_keys} keys in first shard")
    print(f"  Triple nested keys: {triple_nested_count}")

    if triple_nested_count > 0:
        raise RuntimeError(f"Found {triple_nested_count} triple-nested keys!")

    print("  Structure verification: PASSED")

    # If base path provided, verify weights differ
    if base_path:
        base_shards = sorted(Path(base_path).glob("model*.safetensors"))
        if base_shards:
            test_key = "model.language_model.layers.0.mlp.down_proj.weight"

            try:
                with st.safe_open(str(base_shards[0]), framework='pt') as base_f:
                    with st.safe_open(str(shard_files[0]), framework='pt') as merged_f:
                        if test_key in base_f.keys() and test_key in merged_f.keys():
                            base_tensor = base_f.get_tensor(test_key)
                            merged_tensor = merged_f.get_tensor(test_key)
                            diff = (merged_tensor.float() - base_tensor.float()).abs()
                            print(f"\n  Weight diff verification ({test_key}):")
                            print(f"    Mean diff: {diff.mean():.8f}")
                            print(f"    Max diff:  {diff.max():.6f}")
                            if diff.max() > 0:
                                print("    LoRA weights merged: YES")
                            else:
                                print("    WARNING: No weight difference detected!")
            except Exception as e:
                print(f"  Could not verify weight diff: {e}")

    return True


def main():
    args = parse_args()

    print("="*60)
    print("LoRA Merge Script for Qwen3.5 VLM Models")
    print("="*60)
    print(f"Base model:   {args.base_model}")
    print(f"LoRA adapter: {args.lora_adapter}")
    print(f"Output:       {args.output}")
    print(f"Shard size:   {args.shard_size_gb} GB")
    print(f"Dtype:        {args.dtype}")
    print("="*60)

    # Merge
    dtype = get_torch_dtype(args.dtype)
    output_path = merge_and_save(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output,
        shard_size_gb=args.shard_size_gb,
        dtype=dtype,
    )

    # Verify
    verify_merged_model(output_path, args.base_model)

    print("\nDone!")


if __name__ == "__main__":
    main()
