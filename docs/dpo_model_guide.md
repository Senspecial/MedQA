# DPO模型训练和评估流程详解

## 📚 核心概念

### DPO模型的组成

```
DPO模型 = 基础模型 + SFT知识 + DPO对齐
```

由于使用LoRA训练，实际上涉及多次合并：

```
训练阶段:
┌─────────────┐
│ 基础模型    │  Qwen2.5-1.5B-Instruct
└─────┬───────┘
      │ + (merge)
      ↓
┌─────────────┐
│ SFT LoRA    │  model_output/qwen2_5_1_5b_instruct_sft/
└─────┬───────┘
      │ = (merged model)
      ↓
┌─────────────┐
│基础+SFT模型 │  [内存中，未保存]
└─────┬───────┘
      │ + (apply new LoRA)
      ↓
┌─────────────┐
│ DPO LoRA    │  [训练时创建]
└─────┬───────┘
      │ = (训练完成)
      ↓
┌─────────────┐
│ DPO模型     │  [内存中]
└─────────────┘
```

## 🔄 训练流程（run_dpo_training.py）

### 步骤详解

#### 1. 加载SFT模型

```python
# 第199-203行
model, tokenizer = load_and_merge_lora_model(
    base_model_path=base_model_path,           # Qwen2.5-1.5B-Instruct
    lora_checkpoint_path=sft_checkpoint,        # SFT LoRA
    merge_lora=True  # ⚠️ 合并！得到完整模型
)
```

**结果**: `model` = 基础模型 + SFT知识（完整模型，非LoRA）

#### 2. 配置DPO LoRA

```python
# 第311行
peft_config = setup_lora_config(lora_config)  # 新的DPO LoRA配置
```

**注意**: 这是一个**新的**LoRA配置，用于DPO训练

#### 3. DPO训练

```python
# 第407行
trainer = DPOTrainer(
    model=model,              # 完整模型（基础+SFT）
    peft_config=peft_config,  # 新的DPO LoRA
    ...
)

# 第424行
train_result = trainer.train()  # 训练DPO LoRA
```

**DPOTrainer会做什么**:
- 将 `peft_config` 应用到 `model` 上
- 训练这个新的DPO LoRA适配器
- 适配器学习的是"如何在SFT基础上进一步对齐"

#### 4. 保存模型（修改后）

```python
# 第427-454行（新版本）
if save_merged and isinstance(model, PeftModel):
    # 合并DPO LoRA
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    
    # 保存的是完整模型：基础 + SFT + DPO
```

**保存内容**:
- `model_output/qwen2_5_1_5b_dpo/`: 完整合并模型
- `model_output/qwen2_5_1_5b_dpo/dpo_lora_adapter/`: DPO LoRA适配器（可选）

## 📊 评估流程

### 配置文件 (dpo_evaluation_config.yaml)

```yaml
model:
  model_path: "model_output/qwen2_5_1_5b_dpo"  # 完整模型
  is_lora: false  # ✅ 不是LoRA，是完整模型
  merge_lora: false  # ✅ 不需要合并
```

### 运行评估

```bash
python src/training/scripts/run_evaluation.py \
    --config_path config/dpo_evaluation_config.yaml
```

### 评估指标

1. **通过率** - DPO应该提高
2. **越权率** - DPO应该降低（更安全）
3. **幻觉率** - DPO应该降低（更准确）
4. **红旗遗漏率** - 保持低水平

## 🆚 三种模型对比

### 1. 基础模型
```yaml
model_path: "Qwen2.5-1.5B-Instruct/qwen/Qwen2___5-1___5B-Instruct"
is_lora: false
```

### 2. SFT模型
```yaml
model_path: "model_output/qwen2_5_1_5b_instruct_sft"  # LoRA
base_model_path: "Qwen2.5-1.5B-Instruct/..."
is_lora: true
merge_lora: true  # 需要合并才能使用
```

### 3. DPO模型（新版）
```yaml
model_path: "model_output/qwen2_5_1_5b_dpo"  # 完整模型
is_lora: false  # 已经是完整模型
merge_lora: false
```

## ⚠️ 常见错误

### 错误1: DPO评估使用原始基础模型

```yaml
# ❌ 错误配置
model:
  model_path: "model_output/qwen2_5_1_5b_dpo"  # DPO LoRA
  base_model_path: "Qwen2.5-1.5B-Instruct"     # 原始基础模型
  is_lora: true
```

**问题**: DPO LoRA是基于"基础+SFT"训练的，不能直接加载到原始基础模型

**解决**: 保存完整合并模型

### 错误2: 只保存LoRA适配器

```python
# ❌ 旧版本
trainer.save_model(output_dir)  # 只保存DPO LoRA
```

**问题**: 
- 保存的DPO LoRA需要"基础+SFT"才能使用
- 但"基础+SFT"没有保存

**解决**: 合并后保存完整模型

## 📁 目录结构

```
model_output/
├── qwen2_5_1_5b_instruct_sft/        # SFT LoRA适配器
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
└── qwen2_5_1_5b_dpo/                 # DPO完整模型（新版）
    ├── config.json
    ├── model.safetensors             # 完整模型权重
    ├── tokenizer*
    └── dpo_lora_adapter/             # 可选：DPO LoRA
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## 🎯 最佳实践

### 训练配置

```yaml
# config/dpo_training_config.yaml
model:
  base_model_path: "Qwen2.5-1.5B-Instruct/..."
  sft_checkpoint_path: "model_output/qwen2_5_1_5b_instruct_sft"
  is_lora: true
  save_merged_dpo: true  # ✅ 保存完整模型

lora:
  enabled: true  # 使用LoRA训练DPO（省显存）
```

### 评估配置

```yaml
# config/dpo_evaluation_config.yaml
model:
  model_path: "model_output/qwen2_5_1_5b_dpo"
  is_lora: false  # 完整模型
  
baseline_comparison:
  enabled: true
  baseline_model_path: "model_output/qwen2_5_1_5b_instruct_sft"
  baseline_is_lora: true  # SFT是LoRA
```

### 对比评估脚本

```bash
# 1. 评估SFT模型（基线）
python src/training/scripts/run_evaluation.py \
    --config_path config/evaluation_config.yaml

# 2. 评估DPO模型
python src/training/scripts/run_evaluation.py \
    --config_path config/dpo_evaluation_config.yaml

# 3. 对比结果
python scripts/compare_models.py \
    output/evaluation/ \
    output/evaluation_dpo/
```

## 📈 期望的改进

DPO训练后，预期看到：

| 指标 | SFT | DPO | 改进 |
|------|-----|-----|------|
| 通过率 | 45% | **55%+** | ⬆️ +10% |
| 越权率 | 25% | **15%** | ⬇️ -10% |
| 幻觉率 | 20% | **15%** | ⬇️ -5% |
| 红旗遗漏 | 10% | **5%** | ⬇️ -5% |

**关键改进点**:
- ✅ 更安全：减少确诊、给剂量等越权行为
- ✅ 更准确：减少编造信息和错误
- ✅ 更合规：保持或提升急症识别能力

## 🔍 故障排查

### 问题: 评估时加载失败

```
Error: Unable to load adapter
```

**检查**:
```bash
# 1. 确认DPO模型是否是完整模型
ls -lh model_output/qwen2_5_1_5b_dpo/*.safetensors

# 2. 检查是否有adapter_config.json（如果有，说明是LoRA）
ls model_output/qwen2_5_1_5b_dpo/adapter_config.json

# 3. 确认评估配置
cat config/dpo_evaluation_config.yaml | grep "is_lora"
```

**修复**:
- 如果保存的是LoRA: 重新训练并保存完整模型
- 如果保存的是完整模型: 设置 `is_lora: false`

---

**更新时间**: 2026-02-01  
**版本**: v2.0（修复DPO模型保存逻辑）
