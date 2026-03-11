# MedQA 完整训练流水线
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │                     完整流水线执行顺序                               │
# ├──────┬──────────────────────────────────────────────────────────────┤
# │ 步骤 │ 说明                                                         │
# ├──────┼──────────────────────────────────────────────────────────────┤
# │  0   │ [可选] 增量预训练 (CPT) — 领域知识注入                       │
# │  1   │ 数据过滤 — 质量筛选 + 隐私过滤                               │
# │  2   │ 数据均衡 — 按类别配比                                        │
# │  3   │ SFT 训练 — 有监督微调                                        │
# │  4   │ LoRA 合并 — 将 adapter 合并到 base model                    │
# │  5   │ DPO 数据构建 — 生成 chosen/rejected 对                       │
# │  6   │ DPO 训练 — 偏好对齐                                          │
# │  7   │ 评估 — 自动质量评估                                          │
# │  8   │ 推理 — 交互/批量推理                                         │
# └──────┴──────────────────────────────────────────────────────────────┘
#
# ============================================================
# 步骤 0: 增量预训练 (Continual Pre-Training, CPT)  [可选]
# ============================================================
# 目标: 在有监督微调前，先让模型在医学领域原始语料（教材、指南、论文等）
#       上进行领域知识注入，使用因果语言建模 (CLM) 目标。
#
# 配置文件: config/cpt_config.yaml
#
# 运行:
#   # 使用默认配置
#   python src/training/scripts/run_cpt.py --config config/cpt_config.yaml
#
#   # 使用快速验证场景
#   python src/training/scripts/run_cpt.py \
#       --config config/cpt_config.yaml --scenario quick_test \
#       --input_paths data/pretrain/medical.txt
#
#   # 使用标准场景（更多 epoch、更大 batch）
#   python src/training/scripts/run_cpt.py \
#       --config config/cpt_config.yaml --scenario standard \
#       --input_paths data/pretrain/medical_textbooks.txt \
#                     data/pretrain/clinical_guidelines.txt \
#                     data/pretrain/medical_papers.jsonl
#
# CPT 完成后合并 LoRA:
#   python src/training/scripts/merge_lora_model.py \
#       --base_model Qwen/Qwen2.5-1.5B-Instruct \
#       --lora_model model_output/qwen2_5_1_5b_cpt \
#       --output_dir model_output/qwen2_5_1_5b_cpt_merged
#
# 然后将 model_output/qwen2_5_1_5b_cpt_merged 填入 SFT 配置的 base_model_path。
#
# ============================================================
# 步骤 1: 数据过滤
# ============================================================
#   python src/training/scripts/run_data_filter_with_config.py \
#       --config config/data_filter_config.yaml --max_samples 2000
#
# ============================================================
# 步骤 2: 数据均衡
# ============================================================
#   python src/training/scripts/run_data_balance.py \
#       --input output/train.json --output output/train_balanced.json \
#       --config balanced_training
#
# ============================================================
# 步骤 3: SFT 训练
# ============================================================
#   python -m src.training.trainer.run_sft
#
# ============================================================
# 步骤 4: LoRA 合并 (SFT)
# ============================================================
#   python scripts/merge_sft_model.py
#   # 或
#   python src/training/scripts/merge_lora_model.py \
#       --base_model Qwen/Qwen2.5-1.5B-Instruct \
#       --lora_model model_output/qwen2_5_1_5b_instruct_sft \
#       --output_dir model_output/qwen2_5_1_5b_instruct_sft_merged
#
# ============================================================
# 步骤 5: DPO 数据构建
# ============================================================
#   python src/training/scripts/run_dpo_construction.py \
#       --config config/dpo_construction_config.yaml
#
# ============================================================
# 步骤 6: DPO 训练
# ============================================================
#   python src/training/scripts/run_dpo_training.py \
#       --config_path config/dpo_training_config.yaml
#
# ============================================================
# 步骤 7: 评估
# ============================================================
#   python src/training/scripts/run_evaluation.py \
#       --config config/evaluation_config.yaml
#   python src/training/scripts/run_evaluation.py \
#       --config config/dpo_evaluation_config.yaml
#
# ============================================================
# 步骤 8: 推理
# ============================================================
#   python src/inference/run_inference_with_config.py \
#       --config config/inference_config.yaml --mode interactive
