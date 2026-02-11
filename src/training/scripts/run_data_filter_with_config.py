#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用配置文件运行数据清洗与过滤
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.dataset.data_processor import MedicalDataProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RunDataFilter")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_data_filter(config_path: str, max_samples: int = None):
    """运行数据过滤
    
    Args:
        config_path: 配置文件路径
        max_samples: 最大处理样本数（用于测试）
    """
    
    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 获取配置参数
    data_file = config.get('data_file', '')
    data_dir = config.get('data_dir', '')
    output_dir = config.get('output_dir', 'output')
    max_workers = config.get('max_workers', 4)
    
    # 确定数据源
    if data_file:
        data_path = data_file
        is_single_file = True
    elif data_dir:
        data_path = data_dir
        is_single_file = False
    else:
        logger.error("错误: 配置文件中必须指定 data_file 或 data_dir")
        sys.exit(1)
    
    # API配置
    deepseek_config = config.get('deepseek', {})
    api_key = os.environ.get('DEEPSEEK_API_KEY') or deepseek_config.get('api_key', '')
    enable_annotation = deepseek_config.get('enable_annotation', False)
    
    # 过滤器配置
    privacy_config = config.get('privacy_filter', {})
    quality_config = config.get('quality_filter', {})
    enable_privacy = privacy_config.get('enabled', True)
    enable_quality = quality_config.get('enabled', True)
    
    # 输出配置
    output_config = config.get('output', {})
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("数据清洗配置:")
    logger.info(f"  数据源: {data_path} ({'文件' if is_single_file else '目录'})")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  工作线程数: {max_workers}")
    logger.info(f"  隐私过滤: {'启用' if enable_privacy else '禁用'}")
    logger.info(f"  质量过滤: {'启用' if enable_quality else '禁用'}")
    logger.info(f"  DeepSeek标注: {'启用' if enable_annotation else '禁用'}")
    if max_samples:
        logger.info(f"  ⚠️  测试模式: 只处理前 {max_samples} 条样本")
    logger.info("=" * 60)
    
    # 检查数据路径
    if not os.path.exists(data_path):
        logger.error(f"错误: 数据路径不存在: {data_path}")
        logger.error("请检查配置文件中的 data_file 或 data_dir 路径")
        sys.exit(1)
    
    # 检查API密钥（如果需要标注）
    if enable_annotation and (not api_key or api_key.startswith('sk-xxx')):
        logger.error("错误: 启用了DeepSeek标注但未设置有效的API密钥")
        logger.error("请设置环境变量: export DEEPSEEK_API_KEY=your_key")
        logger.error("或在配置文件中填写 deepseek.api_key")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据目录
    if is_single_file:
        # 单个文件：加载并格式化
        logger.info("加载原始数据文件...")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"原始数据样本数: {len(raw_data)}")
        
        # 如果指定了最大样本数，只取前N条
        if max_samples and max_samples < len(raw_data):
            logger.info(f"测试模式：只取前 {max_samples} 条样本")
            raw_data = raw_data[:max_samples]
        
        # 转换数据格式为处理器需要的格式
        # 注意：data_processor 期望的是 'question' 和 'answer' 字段
        formatted_data = []
        for item in raw_data:
            if 'question' in item and 'answer' in item:
                # 已经是正确格式
                formatted_data.append(item)
            elif 'instruction' in item and 'output' in item:
                # 转换 instruction/output 格式
                formatted_data.append({
                    'question': item['instruction'],
                    'answer': item['output']
                })
            elif 'prompt' in item and 'response' in item:
                # 转换 prompt/response 格式
                formatted_data.append({
                    'question': item['prompt'],
                    'answer': item['response']
                })
            elif 'query' in item and 'response' in item:
                # 转换 query/response 格式
                formatted_data.append({
                    'question': item['query'],
                    'answer': item['response']
                })
        
        logger.info(f"格式化后样本数: {len(formatted_data)}")
        
        if len(formatted_data) == 0:
            logger.error("错误: 无法识别数据格式")
            logger.error("支持的格式: {'question': '...', 'answer': '...'} 或 {'instruction': '...', 'output': '...'}")
            sys.exit(1)
        
        # 临时保存格式化数据
        temp_data_dir = os.path.join(output_dir, 'temp_raw')
        os.makedirs(temp_data_dir, exist_ok=True)
        
        temp_data_path = os.path.join(temp_data_dir, 'data.json')
        with open(temp_data_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"临时数据已保存到: {temp_data_path}")
        
        # 使用临时目录作为数据源
        final_data_dir = temp_data_dir
    else:
        # 目录：直接使用
        logger.info(f"使用数据目录: {data_path}")
        final_data_dir = data_path
    
    # 创建数据处理器
    logger.info("初始化数据处理器...")
    processor = MedicalDataProcessor(
        data_dir=final_data_dir,
        output_dir=output_dir,
        max_workers=max_workers,
        deepseek_api_key=api_key if enable_annotation else None,
        enable_privacy_filter=enable_privacy,
        enable_quality_filter=enable_quality
    )
    
    # 处理数据
    logger.info("开始处理数据...")
    logger.info("-" * 60)
    
    datasets = processor.process_all_data(enable_annotation=enable_annotation)
    
    logger.info("-" * 60)
    logger.info("数据处理完成！")
    
    # 输出统计信息
    logger.info("\n处理结果统计:")
    logger.info(f"  训练集: {len(datasets['train'])} 样本")
    logger.info(f"  验证集: {len(datasets['validation'])} 样本")
    logger.info(f"  测试集: {len(datasets['test'])} 样本")
    logger.info(f"  总计: {len(datasets['train']) + len(datasets['validation']) + len(datasets['test'])} 样本")
    
    if hasattr(processor, 'stats'):
        stats = processor.stats
        logger.info(f"\n过滤统计:")
        logger.info(f"  隐私过滤: {stats.get('privacy_filtered', 0)} 样本")
        logger.info(f"  质量过滤: {stats.get('quality_filtered', 0)} 样本")
        if enable_annotation:
            logger.info(f"  标注样本: {stats.get('annotated_samples', 0)} 样本")
    
    # 输出文件列表
    logger.info("\n输出文件:")
    output_files = list(Path(output_dir).glob('*'))
    for f in sorted(output_files):
        if f.is_file():
            size = f.stat().st_size / 1024 / 1024  # MB
            logger.info(f"  {f.name} ({size:.2f} MB)")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 数据清洗完成！")
    logger.info(f"✓ 输出目录: {output_dir}")
    logger.info("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用配置文件运行数据清洗")
    parser.add_argument(
        "--config",
        type=str,
        default="config/data_filter_config.yaml",
        help="配置文件路径 (默认: config/data_filter_config.yaml)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大处理样本数（用于测试，默认处理全部）"
    )
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        logger.error(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 运行数据过滤
    try:
        run_data_filter(config_path, max_samples=args.max_samples)
    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
# python src/training/scripts/run_data_filter_with_config.py --config config/data_filter_config.yaml --max_samples 2000