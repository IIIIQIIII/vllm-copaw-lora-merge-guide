#!/usr/bin/env python3
"""
将 float32 模型转换为 bfloat16 以减小模型大小
从 36GB -> ~18GB
"""

import json
from pathlib import Path
import safetensors.torch
import torch

def convert_to_bfloat16(input_dir: str, output_dir: str):
    """将模型从 float32 转换为 bfloat16"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("将模型从 float32 转换为 bfloat16")
    print("=" * 60)

    # 获取所有 safetensors 文件
    safetensor_files = sorted(input_path.glob('model-*.safetensors'))
    print(f"\n找到 {len(safetensor_files)} 个权重文件")

    total_size_before = 0
    total_size_after = 0

    # 处理每个分片
    for input_file in safetensor_files:
        output_file = output_path / input_file.name
        print(f"\n处理: {input_file.name}")

        # 读取权重并转换为 bfloat16
        tensors_bf16 = {}
        with safetensors.safe_open(str(input_file), framework='pt') as f:
            keys = list(f.keys())

            for key in keys:
                tensor = f.get_tensor(key)

                # 转换为 bfloat16（如果不是的话）
                if tensor.dtype == torch.float32:
                    tensor_bf16 = tensor.to(torch.bfloat16)
                    total_size_before += tensor.nbytes
                    total_size_after += tensor_bf16.nbytes
                else:
                    tensor_bf16 = tensor
                    total_size_before += tensor.nbytes
                    total_size_after += tensor.nbytes

                tensors_bf16[key] = tensor_bf16

        # 保存转换后的权重
        safetensors.torch.save_file(tensors_bf16, str(output_file))

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  - 已保存: {size_mb:.1f} MB")

    # 复制其他配置文件
    print("\n复制配置文件...")
    import shutil

    config_files = [
        'config.json',
        'generation_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'processor_config.json',
        'chat_template.jinja',
        'model.safetensors.index.json'
    ]

    for config_file in config_files:
        src = input_path / config_file
        if src.exists():
            shutil.copy2(src, output_path / config_file)
            print(f"  - 已复制: {config_file}")

    # 打印统计
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"转换前总大小: {total_size_before / (1024**3):.2f} GB")
    print(f"转换后总大小: {total_size_after / (1024**3):.2f} GB")
    print(f"节省空间: {(total_size_before - total_size_after) / (1024**3):.2f} GB ({(1 - total_size_after/total_size_before)*100:.1f}%)")
    print(f"\n输出目录: {output_dir}")

if __name__ == '__main__':
    input_dir = './models/CoPaw-Flash-9B-Agent-Complete'
    output_dir = './models/CoPaw-Flash-9B-Agent-Complete-BF16'

    convert_to_bfloat16(input_dir, output_dir)
