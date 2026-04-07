#!/usr/bin/env python3
"""
修复 ms-swift 合并后的模型权重键名
将三重嵌套的键名修复为与 base model 一致的结构
"""

import os
import json
from pathlib import Path
import safetensors.torch

def fix_key_name(key: str) -> str:
    """
    修复权重键名
    从: model.language_model.language_model.language_model.layers.X.*
    到: model.language_model.layers.X.*
    """
    if key.startswith('model.language_model.language_model.language_model.'):
        # 移除多余的两个 'language_model.' 前缀
        return key.replace(
            'model.language_model.language_model.language_model.',
            'model.language_model.',
            1
        )
    return key

def fix_merged_model(input_dir: str, output_dir: str):
    """修复合并后的模型权重键名"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有 safetensors 文件
    safetensor_files = sorted(input_path.glob('model-*.safetensors'))

    print(f"找到 {len(safetensor_files)} 个权重文件")

    # 处理每个分片
    for input_file in safetensor_files:
        output_file = output_path / input_file.name
        print(f"\n处理: {input_file.name}")

        # 读取权重
        tensors = {}
        with safetensors.safe_open(str(input_file), framework='pt') as f:
            keys = list(f.keys())
            print(f"  - 原始键数量: {len(keys)}")

            fixed_count = 0
            for key in keys:
                tensor = f.get_tensor(key)
                new_key = fix_key_name(key)

                if new_key != key:
                    fixed_count += 1
                    if fixed_count <= 3:  # 只打印前3个示例
                        print(f"  - 修复键名: {key[:60]}... -> {new_key[:60]}...")

                tensors[new_key] = tensor

        print(f"  - 修复了 {fixed_count} 个键名")

        # 保存修复后的权重
        safetensors.torch.save_file(tensors, str(output_file))
        print(f"  - 已保存到: {output_file}")

    # 复制其他配置文件
    print("\n复制配置文件...")
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
            dst = output_path / config_file

            # 对于 index 文件，需要更新里面的键名
            if config_file == 'model.safetensors.index.json':
                with open(src, 'r') as f:
                    index_data = json.load(f)

                # 修复 weight_map 中的键名
                if 'weight_map' in index_data:
                    new_weight_map = {}
                    for key, shard in index_data['weight_map'].items():
                        new_key = fix_key_name(key)
                        new_weight_map[new_key] = shard
                    index_data['weight_map'] = new_weight_map

                with open(dst, 'w') as f:
                    json.dump(index_data, f, indent=2)
                print(f"  - 已更新并复制: {config_file}")
            else:
                import shutil
                shutil.copy2(src, dst)
                print(f"  - 已复制: {config_file}")

    print(f"\n✅ 修复完成！输出目录: {output_dir}")

if __name__ == '__main__':
    input_dir = './models/CoPaw-Flash-9B-Agent-Merged'
    output_dir = './models/CoPaw-Flash-9B-Agent-Merged-Fixed'

    print("=" * 60)
    print("修复 CoPaw-Flash-9B-Agent-Merged 模型权重键名")
    print("=" * 60)

    fix_merged_model(input_dir, output_dir)
