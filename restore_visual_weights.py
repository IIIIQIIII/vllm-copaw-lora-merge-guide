#!/usr/bin/env python3
"""
从 base model 复制 visual 权重到合并后的模型
因为 ms-swift 合并时只处理了 language_model 权重，丢失了 visual 权重
"""

import json
from pathlib import Path
import safetensors.torch

def restore_visual_weights(base_model_dir: str, merged_dir: str, output_dir: str):
    """从 base model 恢复 visual 权重到合并模型"""
    base_path = Path(base_model_dir)
    merged_path = Path(merged_dir)
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("恢复 Visual 权重")
    print("=" * 60)

    # 1. 从 base model 提取 visual 权重和 lm_head
    print("\n步骤 1: 从 base model 加载权重...")
    base_tensors = {}
    with safetensors.safe_open(str(base_path / 'model.safetensors'), framework='pt') as f:
        for key in f.keys():
            # 复制 visual 权重和 lm_head
            if key.startswith('model.visual') or key == 'lm_head.weight':
                base_tensors[key] = f.get_tensor(key)

    print(f"  - 提取了 {len(base_tensors)} 个权重 (visual + lm_head)")

    # 2. 加载合并模型的所有 language_model 权重
    print("\n步骤 2: 从合并模型加载 language_model 权重...")
    merged_tensors = {}
    for i in range(1, 9):
        shard_file = merged_path / f'model-0000{i}-of-00008.safetensors'
        with safetensors.safe_open(str(shard_file), framework='pt') as f:
            for key in f.keys():
                if key.startswith('model.language_model'):
                    merged_tensors[key] = f.get_tensor(key)

    print(f"  - 加载了 {len(merged_tensors)} 个 language_model 权重")

    # 3. 合并权重
    print("\n步骤 3: 合并所有权重...")
    all_tensors = {**base_tensors, **merged_tensors}
    print(f"  - 总权重数: {len(all_tensors)}")

    # 4. 重新分片保存（8个分片）
    print("\n步骤 4: 重新分片并保存...")

    # 按照原始分片逻辑分组
    # shard 1: lm_head
    # shard 2-8: 其他权重
    shards = [[] for _ in range(8)]

    # lm_head 放在第一个分片
    if 'lm_head.weight' in all_tensors:
        shards[0].append('lm_head.weight')

    # 将其他权重分配到分片
    other_keys = [k for k in sorted(all_tensors.keys()) if k != 'lm_head.weight']

    # 计算每个分片大概应该有多少权重
    keys_per_shard = len(other_keys) // 7  # 7个分片来分配其他权重

    shard_idx = 1
    for i, key in enumerate(other_keys):
        shards[shard_idx].append(key)
        if len(shards[shard_idx]) >= keys_per_shard and shard_idx < 7:
            shard_idx += 1

    # 保存每个分片
    weight_map = {}
    for i, shard_keys in enumerate(shards):
        if not shard_keys:
            continue

        shard_num = i + 1
        shard_file = output_path / f'model-0000{shard_num}-of-00008.safetensors'

        shard_tensors = {k: all_tensors[k] for k in shard_keys}
        safetensors.torch.save_file(shard_tensors, str(shard_file))

        # 更新 weight_map
        for key in shard_keys:
            weight_map[key] = f'model-0000{shard_num}-of-00008.safetensors'

        print(f"  - {shard_file.name}: {len(shard_keys)} 个权重")

    # 5. 更新 index 文件
    print("\n步骤 5: 更新 model.safetensors.index.json...")
    with open(merged_path / 'model.safetensors.index.json', 'r') as f:
        index_data = json.load(f)

    index_data['weight_map'] = weight_map
    # 更新 metadata
    index_data['metadata'] = {
        'total_size': sum(t.nbytes for t in all_tensors.values())
    }

    with open(output_path / 'model.safetensors.index.json', 'w') as f:
        json.dump(index_data, f, indent=2)

    # 6. 复制其他配置文件
    print("\n步骤 6: 复制配置文件...")
    config_files = [
        'config.json',
        'generation_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'processor_config.json',
        'chat_template.jinja'
    ]

    import shutil
    for config_file in config_files:
        src = merged_path / config_file
        if src.exists():
            shutil.copy2(src, output_path / config_file)
            print(f"  - 已复制: {config_file}")

    print(f"\n✅ 完成！完整模型已保存到: {output_dir}")

if __name__ == '__main__':
    base_model_dir = './models/CoPaw-Flash-9B'
    merged_dir = './models/CoPaw-Flash-9B-Agent-Merged-Fixed'
    output_dir = './models/CoPaw-Flash-9B-Agent-Complete'

    restore_visual_weights(base_model_dir, merged_dir, output_dir)
