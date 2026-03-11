"""
增量预训练入口脚本 (run_cpt.py)

使用方式:
    # 使用默认配置（顶层字段，数据路径从 config 读取）
    python src/training/scripts/run_cpt.py --config config/cpt_config.yaml

    # 快速验证场景
    python src/training/scripts/run_cpt.py \\
        --config config/cpt_config.yaml --scenario quick_test

    # 标准训练场景
    python src/training/scripts/run_cpt.py \\
        --config config/cpt_config.yaml --scenario standard

    # 命令行覆盖数据路径 / 输出目录等
    python src/training/scripts/run_cpt.py \\
        --config config/cpt_config.yaml \\
        --train_path data/pt/train_encyclopedia.json \\
        --valid_path data/pt/valid_encyclopedia.json \\
        --test_path  data/pt/test_encyclopedia.json \\
        --output_dir model_output/qwen3_5_0.8b_cpt

训练完成后，评估结果保存在 <output_dir>/cpt_eval_results.json。
后续合并 LoRA:
    python src/training/scripts/merge_lora_model.py \\
        --base_model Qwen/Qwen3.5-0_8B-Base \\
        --lora_model model_output/qwen3_5_0_8b_cpt \\
        --output_dir model_output/qwen3_5_0_8b_cpt_merged
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.logger import setup_logger
from src.training.trainer.cpt_trainer import build_cpt_trainer_from_config

logger = setup_logger(__name__)

# 具名场景字段（不参与顶层 base 合并）
_SCENARIO_KEYS = {"quick_test", "standard"}


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str, scenario: str | None = None) -> dict:
    """加载 YAML 配置，可选地合并具名 scenario 覆盖顶层配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 顶层 base 配置（排除场景字段）
    base_cfg = {k: v for k, v in raw.items() if k not in _SCENARIO_KEYS}

    if scenario:
        if scenario not in raw:
            available = [k for k in raw if k in _SCENARIO_KEYS]
            raise KeyError(
                f"场景 '{scenario}' 不存在。可用场景: {available}"
            )
        logger.info(f"使用场景配置: {scenario}")
        return _deep_merge(copy.deepcopy(base_cfg), raw[scenario])

    return base_cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# CLI 参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="增量预训练 (CPT) 入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config/cpt_config.yaml", help="配置文件路径")
    parser.add_argument("--scenario", default=None, help="具名场景 (quick_test / standard)")

    # 数据路径覆盖
    parser.add_argument("--train_path", default=None, help="训练集文件路径（覆盖 config）")
    parser.add_argument("--valid_path", default=None, help="验证集文件路径（覆盖 config）")
    parser.add_argument("--test_path",  default=None, help="测试集文件路径（覆盖 config）")

    # 其他覆盖
    parser.add_argument("--output_dir",   default=None, help="输出目录（覆盖 config）")
    parser.add_argument("--base_model",   default=None, help="基础模型路径（覆盖 config）")
    parser.add_argument("--max_length",   type=int, default=None, help="最大序列长度")
    parser.add_argument("--max_samples",  type=int, default=None, help="最大训练样本数")
    parser.add_argument("--no_pack",      action="store_true", help="关闭 pack_sequences 模式")
    parser.add_argument("--resume",       action="store_true", help="从最新 checkpoint 续训")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info(f"加载配置: {args.config}" + (f"  场景: {args.scenario}" if args.scenario else ""))
    config = load_config(args.config, scenario=args.scenario)

    # 命令行参数覆盖 config
    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})
    eval_cfg  = config.setdefault("evaluation", {})

    if args.train_path:  data_cfg["train_path"] = args.train_path
    if args.valid_path:  data_cfg["valid_path"] = args.valid_path
    if args.test_path:   data_cfg["test_path"]  = args.test_path
    if args.output_dir:  model_cfg["output_dir"] = args.output_dir
    if args.base_model:  model_cfg["base_model_path"] = args.base_model
    if args.max_length is not None:  data_cfg["max_length"] = args.max_length
    if args.max_samples is not None: data_cfg["max_train_samples"] = args.max_samples
    if args.no_pack:                 data_cfg["pack_sequences"] = False

    # 必须有训练集
    train_path = data_cfg.get("train_path")
    if not train_path:
        raise ValueError(
            "未指定训练集路径！请在 config 中设置 data.train_path 或使用 --train_path。"
        )

    valid_path   = data_cfg.get("valid_path")
    test_path    = data_cfg.get("test_path")
    max_length   = data_cfg.get("max_length", 1024)
    pack_seqs    = data_cfg.get("pack_sequences", True)
    max_samples  = data_cfg.get("max_train_samples")
    text_field   = data_cfg.get("text_field", "text")
    resume       = args.resume or config.get("training", {}).get("resume_from_checkpoint", False)
    eval_bs      = eval_cfg.get("eval_batch_size", 4)

    # 打印配置摘要
    logger.info("=" * 60)
    logger.info("增量预训练配置摘要")
    logger.info("=" * 60)
    logger.info(f"  基础模型    : {model_cfg.get('base_model_path')}")
    logger.info(f"  输出目录    : {model_cfg.get('output_dir')}")
    logger.info(f"  训练集      : {train_path}")
    logger.info(f"  验证集      : {valid_path or '(无)'}")
    logger.info(f"  测试集      : {test_path  or '(无)'}")
    logger.info(f"  最大序列长  : {max_length}")
    logger.info(f"  序列拼接    : {pack_seqs}")
    logger.info(f"  最大训练样本: {max_samples if max_samples else '全量'}")
    logger.info(f"  续训        : {resume}")
    logger.info(f"  评估 batch  : {eval_bs}")
    logger.info("=" * 60)

    # 构建训练器并运行
    trainer = build_cpt_trainer_from_config(config)
    results = trainer.train(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        max_length=max_length,
        pack_sequences=pack_seqs,
        max_samples=max_samples,
        text_field=text_field,
        resume_from_checkpoint=resume,
        eval_batch_size=eval_bs,
    )

    output_dir = results["output_dir"]
    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info(f"  LoRA adapter: {output_dir}")
    logger.info(f"  评估结果   : {output_dir}/cpt_eval_results.json")
    if "valid" in results:
        m = results["valid"]
        logger.info(f"  [valid] PPL={m['ppl']:.2f}  Accuracy={m['accuracy']*100:.2f}%")
    if "test" in results:
        m = results["test"]
        logger.info(f"  [test]  PPL={m['ppl']:.2f}  Accuracy={m['accuracy']*100:.2f}%")
    logger.info("")
    logger.info("后续步骤: 合并 LoRA 后填入 SFT 配置的 base_model_path")
    logger.info(
        f"  python src/training/scripts/merge_lora_model.py "
        f"--base_model {model_cfg.get('base_model_path')} "
        f"--lora_model {output_dir} "
        f"--output_dir {output_dir}_merged"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


#python src/training/scripts/run_cpt.py --config config/cpt_config.yaml --scenario standard

