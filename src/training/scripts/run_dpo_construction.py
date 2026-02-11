#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DPO负样本构造脚本
从SFT数据生成DPO训练所需的chosen/rejected对
"""

import os
import sys
import json
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目根目录到路径
script_dir = Path(__file__).resolve().parent
training_dir = script_dir.parent
src_dir = training_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(project_root))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.training.dataset.dpo_negative_constructor import (
    ResponseCandidate, 
    DPOSample, 
    JudgeModel
)

import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """生成统计"""
    total_samples: int = 0
    generated_samples: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    skipped_samples: int = 0
    avg_candidates_per_sample: float = 0.0
    avg_score_difference: float = 0.0


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_sft_model(model_config: Dict, project_root: Path):
    """加载SFT模型"""
    model_path = model_config['model_path']
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    is_lora = model_config.get('is_lora', False)
    device = model_config['device']
    
    if is_lora:
        base_model_path = model_config.get('base_model_path')
        if not os.path.isabs(base_model_path):
            base_model_path = os.path.join(project_root, base_model_path)
        
        logger.info(f"加载LoRA模型...")
        logger.info(f"  基础模型: {base_model_path}")
        logger.info(f"  LoRA适配器: {model_path}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        logger.info("✓ LoRA模型加载完成")
    else:
        logger.info(f"加载模型: {model_path}")
        
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
        
        logger.info("✓ 模型加载完成")
    
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    gen_config: Dict,
    device: str
) -> List[Tuple[str, Dict]]:
    """生成多个候选回答"""
    strategies = gen_config['strategies']
    responses = []
    
    for strategy in strategies:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # 构造输入
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                temperature=strategy['temperature'],
                top_p=strategy['top_p'],
                do_sample=strategy['do_sample'],
                repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        responses.append((response, strategy))
    
    return responses


def evaluate_responses(
    judge_model: JudgeModel,
    question: str,
    responses: List[Tuple[str, Dict]]
) -> List[ResponseCandidate]:
    """评估候选回答"""
    candidates = []
    
    for response, strategy in responses:
        try:
            scores = judge_model.evaluate_response(question, response)
            
            # 计算综合得分 (quality为主，减去幻觉和越权的惩罚)
            overall_score = (
                scores['quality_score'] * 0.4 +
                scores['readability_score'] * 0.2 -
                scores['hallucination_score'] * 0.2 -
                scores['overreach_score'] * 0.2
            )
            
            candidate = ResponseCandidate(
                response=response,
                score=overall_score,
                hallucination_score=scores['hallucination_score'],
                overreach_score=scores['overreach_score'],
                quality_score=scores['quality_score'],
                readability_score=scores['readability_score'],
                details={
                    'strategy': strategy['name'],
                    'overall_comment': scores.get('overall_comment', ''),
                    'specific_issues': scores.get('specific_issues', [])
                }
            )
            candidates.append(candidate)
        except Exception as e:
            logger.warning(f"评估失败: {e}")
            continue
    
    return candidates


def select_dpo_pair(
    candidates: List[ResponseCandidate],
    selection_config: Dict
) -> Optional[Tuple[ResponseCandidate, ResponseCandidate]]:
    """
    选择chosen和rejected对
    
    新逻辑：
    1. chosen: 先取quality_score Top-k，再从中挑选幻觉和越权得分最低的
    2. rejected: 从剩余候选中选择质量差但仍可读的，且有明显问题（幻觉/越权）的
    """
    if len(candidates) < 2:
        return None
    
    chosen_criteria = selection_config['chosen_criteria']
    rejected_criteria = selection_config['rejected_criteria']
    
    # ========== 步骤1: 选择chosen ==========
    # 1.1 先按quality_score排序，取Top-k
    top_k = chosen_criteria.get('top_k', 3)  # 默认Top-3
    sorted_by_quality = sorted(candidates, key=lambda c: c.quality_score, reverse=True)
    top_k_candidates = sorted_by_quality[:min(top_k, len(sorted_by_quality))]
    
    logger.debug(f"Top-{top_k} quality scores: {[c.quality_score for c in top_k_candidates]}")
    
    # 1.2 从Top-k中，挑选幻觉和越权得分都低的
    chosen_candidates = [
        c for c in top_k_candidates
        if (c.quality_score >= chosen_criteria['min_quality_score'] and
            c.hallucination_score <= chosen_criteria['max_hallucination_score'] and
            c.overreach_score <= chosen_criteria['max_overreach_score'] and
            c.readability_score >= chosen_criteria['min_readability_score'])
    ]
    
    if not chosen_candidates:
        logger.debug("没有符合chosen条件的候选")
        return None
    
    # 1.3 在符合条件的候选中，选择幻觉+越权得分之和最低的
    def compute_safety_score(c: ResponseCandidate) -> float:
        """安全性得分：幻觉+越权（越低越安全）"""
        return c.hallucination_score + c.overreach_score
    
    chosen = min(chosen_candidates, key=compute_safety_score)
    
    logger.debug(f"Chosen: quality={chosen.quality_score:.2f}, "
                f"hallucination={chosen.hallucination_score:.2f}, "
                f"overreach={chosen.overreach_score:.2f}")
    
    # ========== 步骤2: 选择rejected ==========
    # 2.1 从剩余候选中筛选
    rejected_candidates = [
        c for c in candidates
        if (c != chosen and
            c.readability_score >= rejected_criteria['min_readability_score'] and
            c.readability_score <= rejected_criteria['max_readability_score'])
    ]
    
    if not rejected_candidates:
        logger.debug("没有符合rejected条件的候选")
        return None
    
    # 2.2 计算"负样本得分"（越高越适合作为rejected）
    # 理想的rejected：质量明显更差 + 有明显的幻觉或越权问题 + 但仍然可读
    weights = rejected_criteria['weights']
    
    def compute_negative_score(candidate: ResponseCandidate) -> float:
        """
        负样本得分计算
        - 质量差距：chosen比它好多少
        - 问题明显性：幻觉或越权问题越严重越好
        - 可读性适中：不要太差也不要太好
        """
        score = 0.0
        
        # 质量差距（越大越好，说明chosen明显更好）
        quality_gap = chosen.quality_score - candidate.quality_score
        if quality_gap < rejected_criteria['min_quality_gap']:
            return -1000  # 质量差距不够，不适合作为rejected
        score += weights['quality'] * quality_gap
        
        # 幻觉问题（越严重越好作为负样本）
        score += weights['hallucination'] * candidate.hallucination_score
        
        # 越权问题（越严重越好作为负样本）
        score += weights['overreach'] * candidate.overreach_score
        
        # 可读性：适中最好（5-7分），太差或太好都不理想
        # 太差：模型学不到什么，太好：容易混淆
        readability_penalty = abs(candidate.readability_score - 6.0)
        score -= weights['readability'] * readability_penalty
        
        return score
    
    # 2.3 选择负样本得分最高的作为rejected
    rejected_scores = [(c, compute_negative_score(c)) for c in rejected_candidates]
    valid_rejected = [(c, s) for c, s in rejected_scores if s > 0]
    
    if not valid_rejected:
        logger.debug("没有有效的rejected候选（质量差距不够）")
        return None
    
    rejected, rejected_score = max(valid_rejected, key=lambda x: x[1])
    
    logger.debug(f"Rejected: quality={rejected.quality_score:.2f}, "
                f"hallucination={rejected.hallucination_score:.2f}, "
                f"overreach={rejected.overreach_score:.2f}, "
                f"negative_score={rejected_score:.2f}")
    
    # ========== 步骤3: 最终验证 ==========
    # 3.1 验证质量差距
    min_diff = selection_config.get('min_score_difference', 1.5)
    quality_diff = chosen.quality_score - rejected.quality_score
    
    if quality_diff < min_diff:
        logger.debug(f"质量差距不足: {quality_diff:.2f} < {min_diff}")
        return None
    
    # 3.2 验证chosen确实更安全（幻觉+越权更低）
    chosen_safety = compute_safety_score(chosen)
    rejected_safety = compute_safety_score(rejected)
    
    if chosen_safety >= rejected_safety:
        logger.debug(f"Chosen不够安全: {chosen_safety:.2f} >= {rejected_safety:.2f}")
        # 允许一定容忍度（如果质量差距很大）
        if quality_diff < min_diff * 2:
            return None
    
    logger.debug(f"✓ 选择成功: quality_gap={quality_diff:.2f}, "
                f"safety_gap={rejected_safety - chosen_safety:.2f}")
    
    return (chosen, rejected)


def construct_dpo_data(config_path: str):
    """构造DPO数据"""
    # 加载配置
    logger.info(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 提取配置
    input_config = config['input_data']
    sft_model_config = config['sft_model']
    gen_config = config['generation']
    judge_config = config['judge_model']
    selection_config = config['selection_strategy']
    output_config = config['output']
    
    # 路径处理
    data_path = input_config['data_path']
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    
    output_path = output_config['output_path']
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    
    report_path = output_config.get('report_path')
    if report_path and not os.path.isabs(report_path):
        report_path = os.path.join(project_root, report_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("DPO负样本构造配置")
    logger.info("=" * 70)
    logger.info(f"输入数据: {data_path}")
    logger.info(f"样本数量: {input_config.get('num_samples') or '全部'}")
    logger.info(f"每个问题生成: {len(gen_config['strategies'])} 个候选回答")
    logger.info(f"输出路径: {output_path}")
    logger.info("=" * 70)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 加载输入数据
    logger.info(f"\n📂 加载输入数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    logger.info(f"总样本数: {len(input_data)}")
    
    # 采样
    num_samples = input_config.get('num_samples')
    random_seed = input_config.get('random_seed', 42)
    random.seed(random_seed)
    
    if num_samples and num_samples < len(input_data):
        input_samples = random.sample(input_data, num_samples)
        logger.info(f"随机抽样 {num_samples} 个样本")
    else:
        input_samples = input_data
    
    # 2. 加载SFT模型
    device = sft_model_config['device']
    model, tokenizer = load_sft_model(sft_model_config, project_root)
    
    # 3. 初始化评审模型
    logger.info(f"\n🔍 初始化评审模型...")
    api_key = os.environ.get('DEEPSEEK_API_KEY') or judge_config.get('api_key', '')
    if not api_key:
        logger.error("❌ 错误: 未设置API密钥")
        logger.error("请设置环境变量: export DEEPSEEK_API_KEY=your_key")
        return
    
    judge_model = JudgeModel(
        api_key=api_key,
        base_url=judge_config['base_url'],
        model=judge_config.get('model', 'deepseek-chat')
    )
    logger.info("✓ 评审模型初始化完成")
    
    # 4. 构造DPO样本
    logger.info(f"\n🔧 开始构造DPO样本...")
    system_prompt = config.get('system_prompt', '你是一个专业的医疗助手。')
    
    dpo_samples = []
    stats = GenerationStats()
    stats.total_samples = len(input_samples)
    
    all_candidates_data = []  # 保存所有候选回答（用于分析）
    
    for idx, sample in enumerate(tqdm(input_samples, desc="构造DPO样本")):
        question = sample.get('question') or sample.get('query') or sample.get('instruction') or ""
        
        if not question:
            stats.skipped_samples += 1
            continue
        
        try:
            # 生成候选回答
            responses = generate_responses(
                model, tokenizer, question, system_prompt, gen_config, device
            )
            
            # 评估候选回答
            candidates = evaluate_responses(judge_model, question, responses)
            
            if not candidates:
                stats.skipped_samples += 1
                continue
            
            stats.generated_samples += 1
            stats.avg_candidates_per_sample += len(candidates)
            
            # 保存所有候选（如果需要）
            if output_config.get('save_all_candidates', True):
                all_candidates_data.append({
                    'question': question,
                    'candidates': [asdict(c) for c in candidates],
                    'sample_id': sample.get('id', f'sample_{idx}')
                })
            
            # 选择chosen和rejected对
            pair = select_dpo_pair(candidates, selection_config)
            
            if pair is None:
                stats.invalid_pairs += 1
                continue
            
            chosen, rejected = pair
            stats.valid_pairs += 1
            stats.avg_score_difference += (chosen.quality_score - rejected.quality_score)
            
            # 创建DPO样本
            dpo_sample = DPOSample(
                prompt=question,
                chosen=chosen.response,
                rejected=rejected.response,
                chosen_score=chosen.score,  # 使用综合得分
                rejected_score=rejected.score,  # 使用综合得分
                metadata={
                    'source_id': sample.get('id', f'sample_{idx}'),
                    'chosen_strategy': chosen.details.get('strategy'),
                    'rejected_strategy': rejected.details.get('strategy'),
                    'score_difference': chosen.quality_score - rejected.quality_score,
                    'num_candidates': len(candidates),
                    # 详细分数存在metadata中
                    'chosen_scores': {
                        'overall': chosen.score,
                        'hallucination': chosen.hallucination_score,
                        'overreach': chosen.overreach_score,
                        'quality': chosen.quality_score,
                        'readability': chosen.readability_score
                    },
                    'rejected_scores': {
                        'overall': rejected.score,
                        'hallucination': rejected.hallucination_score,
                        'overreach': rejected.overreach_score,
                        'quality': rejected.quality_score,
                        'readability': rejected.readability_score
                    }
                }
            )
            
            dpo_samples.append(dpo_sample)
            
        except Exception as e:
            logger.warning(f"处理样本 {idx} 失败: {e}")
            stats.skipped_samples += 1
            continue
    
    # 计算平均值
    if stats.generated_samples > 0:
        stats.avg_candidates_per_sample /= stats.generated_samples
    if stats.valid_pairs > 0:
        stats.avg_score_difference /= stats.valid_pairs
    
    # 5. 保存结果
    logger.info(f"\n💾 保存DPO数据...")
    
    # 转换为字典格式
    dpo_data = [asdict(sample) for sample in dpo_samples]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ DPO数据已保存: {output_path}")
    logger.info(f"  有效样本数: {len(dpo_data)}")
    
    # 保存所有候选（如果需要）
    if output_config.get('save_all_candidates', True) and all_candidates_data:
        candidates_path = output_path.replace('.json', '_all_candidates.json')
        with open(candidates_path, 'w', encoding='utf-8') as f:
            json.dump(all_candidates_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 所有候选已保存: {candidates_path}")
    
    # 6. 生成统计报告
    if output_config.get('save_report', True) and report_path:
        logger.info(f"\n📊 生成统计报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'statistics': asdict(stats),
            'success_rate': stats.valid_pairs / stats.total_samples if stats.total_samples > 0 else 0,
            'sample_quality': {
                'avg_score_difference': stats.avg_score_difference,
                'avg_candidates_per_sample': stats.avg_candidates_per_sample
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 统计报告已保存: {report_path}")
    
    # 7. 打印统计信息
    logger.info(f"\n" + "=" * 70)
    logger.info("DPO样本构造完成")
    logger.info("=" * 70)
    logger.info(f"总样本数: {stats.total_samples}")
    logger.info(f"成功生成: {stats.generated_samples}")
    logger.info(f"有效DPO对: {stats.valid_pairs}")
    logger.info(f"无效DPO对: {stats.invalid_pairs}")
    logger.info(f"跳过样本: {stats.skipped_samples}")
    logger.info(f"成功率: {stats.valid_pairs / stats.total_samples * 100:.1f}%")
    logger.info(f"平均候选数: {stats.avg_candidates_per_sample:.1f}")
    logger.info(f"平均分差: {stats.avg_score_difference:.2f}")
    logger.info("=" * 70)
    
    # 质量检查
    quality_control = config.get('quality_control', {})
    min_valid_pairs = quality_control.get('min_valid_pairs', 10)
    
    if stats.valid_pairs < min_valid_pairs:
        logger.warning(f"⚠️ 警告: 有效样本数 ({stats.valid_pairs}) 少于最小要求 ({min_valid_pairs})")
        logger.warning("建议调整选择策略或增加输入样本数")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构造DPO训练数据")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dpo_construction_config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 配置文件路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    # 检查配置文件
    if not os.path.exists(config_path):
        logger.error(f"❌ 错误: 配置文件不存在: {config_path}")
        return
    
    # 运行构造
    try:
        construct_dpo_data(config_path)
    except Exception as e:
        logger.error(f"\n❌ 构造失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
# python src/training/scripts/run_dpo_construction.py --config config/dpo_construction_config.yaml
