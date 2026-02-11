#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据配比工具 - 对已清洗的数据进行配比
"""

import os
import sys
import json
import yaml
from pathlib import Path
from collections import Counter

# 添加项目根目录到路径
# 脚本位于: project_root/src/training/scripts/run_data_balance.py
# 需要向上三级到达项目根目录
script_dir = Path(__file__).resolve().parent  # scripts/
training_dir = script_dir.parent  # training/
src_dir = training_dir.parent  # src/
project_root = src_dir.parent  # project root

sys.path.insert(0, str(project_root))

from src.training.dataset.data_processor import DataBalancer


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def analyze_distribution(samples, label_field='primary_label'):
    """分析数据分布"""
    labels = [s.get(label_field, '未分类') for s in samples]
    distribution = Counter(labels)
    
    print("\n" + "=" * 60)
    print("数据分布分析")
    print("=" * 60)
    print(f"总样本数: {len(samples)}\n")
    
    for label, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(samples) * 100
        print(f"  {label:15s}: {count:6d} ({percentage:5.1f}%)")
    
    return distribution


def balance_data_from_config(
    input_file: str,
    output_file: str,
    config_section: str = "balanced_training"
):
    """从配置文件读取配置并进行配比
    
    Args:
        input_file: 输入的JSON文件（已清洗的数据）
        output_file: 输出的JSON文件（配比后的数据）
        config_section: 配置文件中的配置章节名
    """
    print("=" * 60)
    print("数据配比工具")
    print("=" * 60)
    
    # 加载数据
    print(f"\n加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"原始数据: {len(samples)} 条")
    
    # 分析原始分布
    analyze_distribution(samples)
    
    # 加载配置
    config_path = project_root / "config" / "data_balance_config.yaml"
    config = load_config(config_path)
    
    if config_section not in config:
        print(f"\n❌ 错误: 配置文件中没有找到 '{config_section}' 配置")
        print(f"可用的配置: {list(config.keys())}")
        return
    
    balance_config = config[config_section]
    
    # 显示配置
    print(f"\n使用配置: {config_section}")
    print(f"  方法: {balance_config.get('method', 'N/A')}")
    
    # 创建配比器
    balancer = DataBalancer()
    
    # 执行配比
    print("\n开始配比...")
    
    method = balance_config.get('method')
    
    try:
        if method == "ratios":
            balanced = balancer.balance_by_ratios(
                samples,
                balance_config.get('target_ratios', {}),
                balance_config.get('total_samples'),
                strategy=balance_config.get('strategy', 'smart')
            )
        
        elif method == "counts":
            balanced = balancer.balance_by_target_counts(
                samples,
                balance_config.get('target_counts', {}),
                strategy=balance_config.get('strategy', 'smart')
            )
        
        elif method == "uniform":
            balanced = balancer.balance_uniform(
                samples,
                target_count=balance_config.get('target_count')
            )
        
        elif method == "min":
            balanced = balancer.balance_by_min_samples(
                samples,
                min_samples_per_label=balance_config.get('min_samples', 100)
            )
        
        elif method == "max":
            balanced = balancer.balance_by_max_samples(
                samples,
                max_samples_per_label=balance_config.get('max_samples', 1000)
            )
        
        else:
            print(f"❌ 未知的配比方法: {method}")
            return
        
        print(f"✓ 配比完成")
        
        # 分析配比后分布
        print("\n配比后分布:")
        analyze_distribution(balanced)
        
        # 保存结果
        print(f"\n保存到: {output_file}")
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(balanced, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 保存成功: {len(balanced)} 条数据")
        
        # 保存统计报告
        stats_file = output_file.replace('.json', '_balance_stats.json')
        stats = {
            "original_count": len(samples),
            "balanced_count": len(balanced),
            "method": method,
            "config": balance_config,
            "original_distribution": dict(Counter([s.get('primary_label', '未分类') for s in samples])),
            "balanced_distribution": dict(Counter([s.get('primary_label', '未分类') for s in balanced]))
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 统计报告: {stats_file}")
        
    except Exception as e:
        print(f"\n❌ 配比失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据配比工具")
    parser.add_argument(
        "--input",
        type=str,
        default="output/train.json",
        help="输入文件 (默认: output/train.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/train_balanced.json",
        help="输出文件 (默认: output/train_balanced.json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balanced_training",
        help="配置章节名 (默认: balanced_training)"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="列出所有可用的配置"
    )
    
    args = parser.parse_args()
    
    # 列出配置
    if args.list_configs:
        config_path = project_root / "config" / "data_balance_config.yaml"
        config = load_config(config_path)
        
        print("\n可用的配置:")
        print("=" * 60)
        for name, cfg in config.items():
            if isinstance(cfg, dict) and 'method' in cfg:
                method = cfg.get('method', 'N/A')
                print(f"\n{name}:")
                print(f"  方法: {method}")
                if method == 'ratios' and 'target_ratios' in cfg:
                    print(f"  目标比例: {cfg['target_ratios']}")
                elif method == 'counts' and 'target_counts' in cfg:
                    print(f"  目标数量: {cfg['target_counts']}")
                elif method == 'uniform' and 'target_count' in cfg:
                    print(f"  目标数量: {cfg['target_count']}")
                elif method == 'min' and 'min_samples' in cfg:
                    print(f"  最小样本数: {cfg['min_samples']}")
                elif method == 'max' and 'max_samples' in cfg:
                    print(f"  最大样本数: {cfg['max_samples']}")
        print("=" * 60)
        return
    
    # 执行配比
    input_file = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_file = args.output if os.path.isabs(args.output) else os.path.join(project_root, args.output)
    
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        print("\n请先运行数据清洗:")
        print("  python src/training/scripts/run_data_filter_with_config.py --max_samples 200")
        return
    
    balance_data_from_config(
        input_file=input_file,
        output_file=output_file,
        config_section=args.config
    )
    
    print("\n" + "=" * 60)
    print("✓ 数据配比完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
