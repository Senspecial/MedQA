#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用配置文件运行医疗QA模型推理
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from src.inference.medical_qa_inference import MedicalQAInference


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_interactive_mode(inferencer: MedicalQAInference):
    """交互式模式"""
    print("\n" + "="*60)
    print("医疗QA交互式对话")
    print("="*60)
    print("输入问题开始对话，输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")
    
    inferencer.interactive_chat()


def run_single_mode(inferencer: MedicalQAInference, config: Dict):
    """单问题模式"""
    question = config['inference']['single']['question']
    gen_config = config['generation']
    
    print("\n" + "="*60)
    print("单问题推理")
    print("="*60)
    print(f"\n问题: {question}\n")
    
    answer = inferencer.generate(
        question,
        max_new_tokens=gen_config['max_new_tokens'],
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        top_k=gen_config['top_k'],
        repetition_penalty=gen_config['repetition_penalty'],
        do_sample=gen_config['do_sample'],
    )
    
    print(f"回答: {answer}\n")
    print("="*60)


def run_batch_mode(inferencer: MedicalQAInference, config: Dict):
    """批量推理模式"""
    batch_config = config['inference']['batch']
    gen_config = config['generation']
    
    input_file = batch_config['input_file']
    output_file = batch_config['output_file']
    batch_size = batch_config.get('batch_size', 4)
    max_samples = batch_config.get('max_samples')
    
    print("\n" + "="*60)
    print("批量推理")
    print("="*60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"批次大小: {batch_size}")
    print("="*60 + "\n")
    
    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取问题
    if isinstance(data, list):
        if isinstance(data[0], str):
            questions = data
            metadata = [{"index": i} for i in range(len(data))]
        elif isinstance(data[0], dict):
            questions = []
            metadata = []
            for item in data:
                question = item.get('question') or item.get('query') or item.get('instruction') or ''
                questions.append(question)
                metadata.append(item)
    else:
        raise ValueError("不支持的输入文件格式")
    
    # 限制样本数
    if max_samples and max_samples < len(questions):
        questions = questions[:max_samples]
        metadata = metadata[:max_samples]
    
    print(f"共 {len(questions)} 个问题\n")
    
    # 批量生成
    answers = inferencer.batch_generate(
        questions,
        batch_size=batch_size,
        max_new_tokens=gen_config['max_new_tokens'],
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        top_k=gen_config['top_k'],
        repetition_penalty=gen_config['repetition_penalty'],
        do_sample=gen_config['do_sample'],
    )
    
    # 构建结果
    results = []
    for question, answer, meta in zip(questions, answers, metadata):
        result = {
            "question": question,
            "answer": answer,
        }
        # 保留原始元数据
        if isinstance(meta, dict):
            for key, value in meta.items():
                if key not in ['question', 'query', 'instruction']:
                    result[key] = value
        results.append(result)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 推理完成！结果已保存到: {output_file}")
    
    # 显示示例
    print("\n" + "="*60)
    print("示例结果（前3个）:")
    print("="*60)
    for i, result in enumerate(results[:3], 1):
        print(f"\n[{i}] 问题: {result['question']}")
        answer_preview = result['answer'][:150] + "..." if len(result['answer']) > 150 else result['answer']
        print(f"    回答: {answer_preview}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用配置文件运行医疗QA模型推理")
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['interactive', 'single', 'batch'],
        default=None,
        help="推理模式（覆盖配置文件中的设置）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return
    
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 提取配置
    model_config = config['model']
    gen_config = config['generation']
    inference_config = config['inference']
    
    # 确定推理模式
    mode = args.mode or inference_config.get('mode', 'interactive')
    
    print(f"\n推理模式: {mode}")
    print(f"模型路径: {model_config['model_path']}")
    
    # 初始化推理器
    inferencer = MedicalQAInference(
        model_path=model_config['model_path'],
        base_model_path=model_config.get('base_model_path'),
        is_lora=model_config.get('is_lora', False),
        merge_lora=model_config.get('merge_lora', True),
        system_prompt=config.get('system_prompt'),
        device=model_config.get('device', 'cuda'),
        load_in_8bit=model_config.get('load_in_8bit', False),
        load_in_4bit=model_config.get('load_in_4bit', False),
    )
    
    # 根据模式运行
    if mode == 'interactive':
        run_interactive_mode(inferencer)
    elif mode == 'single':
        run_single_mode(inferencer, config)
    elif mode == 'batch':
        run_batch_mode(inferencer, config)
    else:
        print(f"❌ 未知的推理模式: {mode}")
        return


if __name__ == "__main__":
    main()
#python src/inference/run_inference_with_config.py --config config/inference_config.yaml --mode interactive