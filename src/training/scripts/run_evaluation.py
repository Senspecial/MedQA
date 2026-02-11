#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用配置文件运行SFT模型评估
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
training_dir = script_dir.parent
src_dir = training_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.dataset.data_processor import MedicalQAEvaluator
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate_with_config(config_path: str):
    """使用配置文件进行评估"""
    
    # 加载配置
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 提取配置
    model_path = config['model']['model_path']
    device = config['model']['device']
    
    test_data_path = config['test_data']['test_data_path']
    num_samples = config['test_data'].get('num_samples')
    random_seed = config['test_data'].get('random_seed', 42)
    
    output_dir = config['output']['output_dir']
    
    gen_config = config['generation']
    eval_config = config['evaluation_metrics']
    
    # 获取API密钥
    api_key = os.environ.get('DEEPSEEK_API_KEY') or config['evaluation_metrics']['judge_model'].get('api_key', '')
    
    print("\n" + "=" * 70)
    print("SFT模型评估配置")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    print(f"测试数据: {test_data_path}")
    print(f"评估样本数: {num_samples or '全部'}")
    print(f"输出目录: {output_dir}")
    print(f"使用评审模型: {eval_config.get('use_judge_model', False)}")
    print(f"计算困惑度: {eval_config.get('calculate_perplexity', False)}")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载模型
    print(f"\n📥 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    print("✓ 模型加载完成")
    
    # 2. 加载测试数据
    print(f"\n📂 加载测试数据...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"测试集样本数: {len(test_data)}")
    
    # 采样
    import random
    random.seed(random_seed)
    
    if num_samples and num_samples < len(test_data):
        test_samples = random.sample(test_data, num_samples)
        print(f"随机抽样 {num_samples} 个样本")
    else:
        test_samples = test_data
    
    # 3. 生成回答
    print(f"\n🤖 生成回答...")
    
    system_prompt = config.get('system_prompt', '你是一个专业的医疗助手。')
    
    def generate_response(question: str) -> str:
        """生成回答"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                do_sample=gen_config['do_sample'],
                repetition_penalty=gen_config.get('repetition_penalty', 1.0),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    # 生成回答
    generated_samples = []
    for sample in tqdm(test_samples, desc="生成回答"):
        question = sample.get('question') or sample.get('query') or ""
        ground_truth = sample.get('answer') or sample.get('response') or ""
        
        generated_answer = generate_response(question)
        
        generated_samples.append({
            'question': question,
            'answer': generated_answer,
            'ground_truth': ground_truth,
            'id': sample.get('id', ''),
            'primary_label': sample.get('primary_label', '')
        })
    
    print(f"✓ 已生成 {len(generated_samples)} 个回答")
    
    # 保存生成样本
    if config['output'].get('save_generated_samples', True):
        samples_path = os.path.join(output_dir, 'generated_samples.json')
        with open(samples_path, 'w', encoding='utf-8') as f:
            json.dump(generated_samples, f, ensure_ascii=False, indent=2)
        print(f"✓ 生成样本已保存: {samples_path}")
    
    results = {}
    
    # 4. 计算困惑度（可选）
    if eval_config.get('calculate_perplexity', False):
        print(f"\n📊 计算困惑度...")
        ppl_samples = test_samples[:eval_config.get('ppl_max_samples')] if eval_config.get('ppl_max_samples') else test_samples
        
        total_loss = 0
        total_tokens = 0
        
        for sample in tqdm(ppl_samples, desc="计算PPL"):
            question = sample.get('question') or ""
            answer = sample.get('answer') or ""
            
            text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        import numpy as np
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        results['perplexity'] = float(perplexity)
        results['avg_loss'] = float(avg_loss)
        
        print(f"✓ 困惑度: {perplexity:.2f}")
        print(f"  平均Loss: {avg_loss:.4f}")
        
        # 评价
        quality_standards = config.get('quality_standards', {})
        excellent_ppl = quality_standards.get('excellent_ppl', 15.0)
        good_ppl = quality_standards.get('good_ppl', 30.0)
        acceptable_ppl = quality_standards.get('acceptable_ppl', 50.0)
        
        if perplexity < excellent_ppl:
            print(f"  评价: ✅ 优秀 (< {excellent_ppl})")
        elif perplexity < good_ppl:
            print(f"  评价: ⚠️ 良好 ({excellent_ppl}-{good_ppl})")
        elif perplexity < acceptable_ppl:
            print(f"  评价: ⚠️ 可接受 ({good_ppl}-{acceptable_ppl})")
        else:
            print(f"  评价: ❌ 需要改进 (> {acceptable_ppl})")
    
    # 5. 使用评审模型评估（可选）
    if eval_config.get('use_judge_model', False):
        if not api_key:
            print("\n⚠️ 未设置API密钥，跳过评审模型评估")
        else:
            print(f"\n📊 使用评审模型评估...")
            
            evaluator = MedicalQAEvaluator(
                api_key=api_key,
                base_url=eval_config['judge_model']['base_url']
            )
            
            # 批量评估
            evaluated_samples = evaluator.batch_evaluate(
                generated_samples,
                batch_size=eval_config['judge_model']['batch_size'],
                max_workers=eval_config['judge_model']['max_workers']
            )
            
            # 生成报告
            if config['output'].get('save_evaluation_report', True):
                report_path = os.path.join(output_dir, 'evaluation_report.json')
                report = evaluator.generate_evaluation_report(evaluated_samples, report_path)
                
                # 提取统计信息
                if 'statistics' in report:
                    stats = report['statistics']
                    results['judge_evaluation'] = stats
                    
                    print(f"\n评估结果:")
                    print(f"  通过率: {stats.get('pass_rate', 0):.1f}%")
                    print(f"  越权率: {stats.get('overreach_rate', 0):.1f}%")
                    print(f"  幻觉率: {stats.get('hallucination_rate', 0):.1f}%")
                    print(f"  红旗遗漏率: {stats.get('red_flag_omission_rate', 0):.1f}%")
                    
                    # 与质量标准对比
                    quality_standards = config.get('quality_standards', {})
                    print(f"\n质量标准检查:")
                    
                    pass_rate = stats.get('pass_rate', 0) / 100
                    if pass_rate >= quality_standards.get('min_pass_rate', 0.7):
                        print(f"  ✅ 通过率达标 ({pass_rate*100:.1f}% >= {quality_standards.get('min_pass_rate', 0.7)*100:.1f}%)")
                    else:
                        print(f"  ❌ 通过率不达标 ({pass_rate*100:.1f}% < {quality_standards.get('min_pass_rate', 0.7)*100:.1f}%)")
                    
                    overreach_rate = stats.get('overreach_rate', 0) / 100
                    if overreach_rate <= quality_standards.get('max_overreach_rate', 0.1):
                        print(f"  ✅ 越权率合格 ({overreach_rate*100:.1f}% <= {quality_standards.get('max_overreach_rate', 0.1)*100:.1f}%)")
                    else:
                        print(f"  ❌ 越权率过高 ({overreach_rate*100:.1f}% > {quality_standards.get('max_overreach_rate', 0.1)*100:.1f}%)")
                    
                    hallucination_rate = stats.get('hallucination_rate', 0) / 100
                    if hallucination_rate <= quality_standards.get('max_hallucination_rate', 0.15):
                        print(f"  ✅ 幻觉率合格 ({hallucination_rate*100:.1f}% <= {quality_standards.get('max_hallucination_rate', 0.15)*100:.1f}%)")
                    else:
                        print(f"  ❌ 幻觉率过高 ({hallucination_rate*100:.1f}% > {quality_standards.get('max_hallucination_rate', 0.15)*100:.1f}%)")
            
            # 保存详细结果
            if config['output'].get('save_detailed_results', True):
                detailed_path = os.path.join(output_dir, 'detailed_results.json')
                with open(detailed_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluated_samples, f, ensure_ascii=False, indent=2)
                print(f"\n✓ 详细结果已保存: {detailed_path}")
    
    # 6. 保存总结
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("✅ 评估完成！")
    print(f"{'='*70}")
    print(f"结果保存在: {output_dir}")
    print(f"  - evaluation_summary.json: 评估总结")
    if eval_config.get('use_judge_model'):
        print(f"  - evaluation_report.json: 详细评估报告")
        print(f"  - detailed_results.json: 详细评估结果")
    if config['output'].get('save_generated_samples'):
        print(f"  - generated_samples.json: 生成样本")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用配置文件评估SFT模型")
    parser.add_argument(
        "--config",
        type=str,
        default="config/evaluation_config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"❌ 错误: 配置文件不存在: {config_path}")
        return
    
    # 运行评估
    try:
        evaluate_with_config(config_path)
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
#python src/training/scripts/run_evaluation.py --config config/evaluation_config.yaml
#python src/training/scripts/run_evaluation.py --config config/dpo_evaluation_config.yaml