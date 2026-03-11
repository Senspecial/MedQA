"""
SFT 训练入口脚本 (run_sft.py)

使用方式:
    python src/training/scripts/run_sft.py --config config/sft_config.yaml

    # 命令行覆盖部分参数
    python src/training/scripts/run_sft.py \\
        --config config/sft_config.yaml \\
        --base_model  /path/to/model \\
        --output_dir  model_output/my_sft \\
        --train_path  data/SFT/train_zh_0.json \\
        --num_epochs  1

    # 断点续训
    python src/training/scripts/run_sft.py \\
        --config config/sft_config.yaml --resume

    # 只跑评估（跳过训练）
    python src/training/scripts/run_sft.py \\
        --config config/sft_config.yaml --eval_only
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
from src.training.dataset.medical_dataset import MedicalDataset, load_system_prompt_from_config
from src.training.trainer.sft_trainer import build_sft_trainer_from_config

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(config: dict) -> str:
    """从 sft_config 中读取 system_prompt_path，加载系统提示。"""
    data_cfg = config.get("data", {})
    prompt_path = data_cfg.get("system_prompt_path", "config/system_prompt.yaml")
    return load_system_prompt_from_config(prompt_path)


# ---------------------------------------------------------------------------
# CLI 参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT 训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",      default="config/sft_config.yaml", help="SFT 配置文件路径")
    parser.add_argument("--base_model",  default=None, help="基础模型路径（覆盖 config）")
    parser.add_argument("--output_dir",  default=None, help="输出目录（覆盖 config）")
    parser.add_argument("--train_path",  default=None, help="训练集路径（覆盖 config）")
    parser.add_argument("--valid_path",  default=None, help="验证集路径（覆盖 config）")
    parser.add_argument("--test_path",   default=None, help="测试集路径（覆盖 config）")
    parser.add_argument("--max_length",  type=int, default=None, help="最大序列长度（覆盖 config）")
    parser.add_argument("--num_epochs",  type=int, default=None, help="训练 epoch 数（覆盖 config）")
    parser.add_argument("--resume",      action="store_true", help="从最新 checkpoint 续训")
    parser.add_argument("--eval_only",   action="store_true", help="跳过训练，只运行测试集评估")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info(f"加载配置: {args.config}")
    config = load_config(args.config)

    # 命令行覆盖
    model_cfg = config.setdefault("model", {})
    data_cfg  = config.setdefault("data", {})
    train_cfg = config.setdefault("training", {})
    eval_cfg  = config.setdefault("evaluation", {})

    if args.base_model:  model_cfg["base_model_path"] = args.base_model
    if args.output_dir:  model_cfg["output_dir"]      = args.output_dir
    if args.train_path:  data_cfg["train_path"]        = args.train_path
    if args.valid_path:  data_cfg["valid_path"]        = args.valid_path
    if args.test_path:   data_cfg["test_path"]         = args.test_path
    if args.max_length:  data_cfg["max_length"]        = args.max_length
    if args.num_epochs:  train_cfg["num_train_epochs"] = args.num_epochs
    if args.resume:      train_cfg["resume_from_checkpoint"] = True

    # 必要参数校验
    if not model_cfg.get("base_model_path"):
        raise ValueError("请在 config 或 --base_model 中指定基础模型路径")
    if not data_cfg.get("train_path") and not args.eval_only:
        raise ValueError("请在 config 或 --train_path 中指定训练集路径")

    max_length = data_cfg.get("max_length", 512)
    system_prompt = load_system_prompt(config)

    # 打印摘要
    logger.info("=" * 60)
    logger.info("SFT 配置摘要")
    logger.info("=" * 60)
    logger.info(f"  基础模型  : {model_cfg.get('base_model_path')}")
    logger.info(f"  输出目录  : {model_cfg.get('output_dir')}")
    logger.info(f"  训练集    : {data_cfg.get('train_path', '(未指定)')}")
    logger.info(f"  验证集    : {data_cfg.get('valid_path', '(无)')}")
    logger.info(f"  测试集    : {data_cfg.get('test_path',  '(无)')}")
    logger.info(f"  最大长度  : {max_length}")
    logger.info(f"  仅评估    : {args.eval_only}")
    logger.info("=" * 60)

    os.makedirs(model_cfg["output_dir"], exist_ok=True)

    # 构建 Trainer
    trainer = build_sft_trainer_from_config(config)

    # ── 训练阶段 ──────────────────────────────────────────────────────
    if not args.eval_only:
        train_ds = MedicalDataset(
            data_cfg["train_path"],
            dataset_type="sft",
            max_length=max_length,
            system_prompt=system_prompt,
            max_samples=data_cfg.get("max_train_samples"),
        )
        valid_ds = None
        if data_cfg.get("valid_path"):
            valid_ds = MedicalDataset(
                data_cfg["valid_path"],
                dataset_type="sft",
                max_length=max_length,
                system_prompt=system_prompt,
                max_samples=data_cfg.get("max_valid_samples"),
            )

        trainer.train(train_dataset=train_ds, eval_dataset=valid_ds)

    # ── 测试集评估阶段 ────────────────────────────────────────────────
    if eval_cfg.get("enabled", True) and data_cfg.get("test_path"):
        if trainer.model is None:
            # eval_only 模式：需要先加载模型
            trainer.load_model_and_tokenizer()

        test_ds = MedicalDataset(
            data_cfg["test_path"],
            dataset_type="sft",
            max_length=max_length,
            system_prompt=system_prompt,
        )
        test_tokenized = test_ds.get_sft_dataset(trainer.tokenizer)

        trainer.evaluate_test_set(
            test_dataset=test_tokenized,
            raw_test_path=data_cfg["test_path"],
            system_prompt=system_prompt,
            max_new_tokens=eval_cfg.get("max_new_tokens", 256),
            num_samples=eval_cfg.get("num_samples", 200),
            bertscore_model_path=eval_cfg.get("bertscore_model_path"),
            save_path=eval_cfg.get("save_path"),
        )
    elif not data_cfg.get("test_path"):
        logger.info("未配置测试集路径，跳过测试集评估")

    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info(f"  模型已保存至: {model_cfg['output_dir']}")
    logger.info(f"  评估结果   : {model_cfg['output_dir']}/sft_eval_results.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
