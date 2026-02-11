# SFT模型合并指南

## 🎯 为什么要单独合并SFT模型？

### 原来的流程（复杂）
```
训练DPO时:
基础模型 + SFT LoRA → (临时合并) → 训练DPO → 保存完整模型
↑ 临时合并，未保存
```

问题：
- SFT完整模型没有保存
- 无法单独评估SFT模型
- DPO训练时需要重复合并

### 新的流程（清晰）✨
```
1. 合并并保存SFT完整模型
   基础模型 + SFT LoRA → SFT完整模型 ✅

2. 评估SFT完整模型
   SFT完整模型 → 评估 ✅

3. 基于SFT完整模型训练DPO
   SFT完整模型 + DPO LoRA → DPO完整模型 ✅
```

好处：
- ✅ SFT模型可单独使用和评估
- ✅ 路径关系清晰
- ✅ DPO训练逻辑简单

---

## 🚀 使用方法

### 方法1: 使用默认参数

```bash
python scripts/merge_sft_model.py
```

**默认配置**:
- 基础模型: `Qwen2.5-1.5B-Instruct/qwen/Qwen2___5-1___5B-Instruct`
- SFT LoRA: `model_output/qwen2_5_1_5b_instruct_sft`
- 输出: `model_output/qwen2_5_1_5b_instruct_sft_merged`

### 方法2: 自定义参数

```bash
python scripts/merge_sft_model.py \
    --base_model "path/to/base/model" \
    --sft_lora "path/to/sft/lora" \
    --output "path/to/output"
```

---

## 📊 预期输出

### 控制台输出

```
======================================================================
SFT模型合并工具
======================================================================

📂 路径配置:
  基础模型: /root/autodl-tmp/MedQA/Qwen2.5-1.5B-Instruct/...
  SFT LoRA: /root/autodl-tmp/MedQA/model_output/qwen2_5_1_5b_instruct_sft
  输出路径: /root/autodl-tmp/MedQA/model_output/qwen2_5_1_5b_instruct_sft_merged

步骤1: 加载基础模型...
✓ 基础模型加载完成

步骤2: 加载SFT LoRA适配器...
✓ SFT LoRA加载完成

📊 模型参数:
  可训练参数: 18,874,368
  总参数: 1,543,746,560
  LoRA参数占比: 1.22%

步骤3: 合并SFT LoRA到基础模型...
✓ LoRA已合并到基础模型

步骤4: 保存完整模型...
✓ 模型已保存

步骤5: 验证保存的文件...

📊 保存的文件:
  model.safetensors: 3089.7 MB
  tokenizer.json: 11.1 MB
  ...

  总大小: 3.02 GB

✅ 验证通过: 这是一个完整的合并模型

======================================================================
✅ SFT模型合并完成！
======================================================================
```

### 目录结构

```
model_output/qwen2_5_1_5b_instruct_sft_merged/
├── config.json                    # 模型配置
├── generation_config.json         # 生成配置
├── model.safetensors             # 完整模型权重 (~3GB)
├── tokenizer.json                # tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
└── ...
```

**关键标志**:
- ✅ 有 `model.safetensors` (完整权重)
- ❌ 没有 `adapter_config.json` (不是LoRA)
- ✅ 总大小 ~3GB (完整模型)

---

## 📝 后续步骤

### 1. 评估SFT完整模型

更新 `config/evaluation_config.yaml`:

```yaml
model:
  model_path: "model_output/qwen2_5_1_5b_instruct_sft_merged"
  is_lora: false  # 完整模型
  merge_lora: false
```

运行评估:

```bash
python src/training/scripts/run_evaluation.py \
    --config_path config/evaluation_config.yaml
```

### 2. 基于SFT完整模型训练DPO

更新 `config/dpo_training_config.yaml`:

```yaml
model:
  base_model_path: "model_output/qwen2_5_1_5b_instruct_sft_merged"  # SFT完整模型
  sft_checkpoint_path: null  # 不需要了
  is_lora: false  # 现在base就是完整的SFT模型
  save_merged_dpo: true
```

运行DPO训练:

```bash
python src/training/scripts/run_dpo_training.py
```

### 3. 对比三个模型

| 模型 | 路径 | 类型 | 评估配置 |
|------|------|------|----------|
| 基础模型 | `Qwen2.5-1.5B-Instruct/...` | 完整 | `is_lora: false` |
| SFT模型 | `qwen2_5_1_5b_instruct_sft_merged/` | 完整 | `is_lora: false` |
| DPO模型 | `qwen2_5_1_5b_dpo/` | 完整 | `is_lora: false` |

---

## ⚠️ 注意事项

### 1. 磁盘空间

合并后的模型约 **3GB**，确保有足够空间：

```bash
# 检查可用空间
df -h /root/autodl-tmp/MedQA/model_output/
```

### 2. 内存需求

合并过程需要加载完整模型到内存/GPU：
- GPU内存: 建议 >= 8GB
- 系统内存: 建议 >= 16GB

### 3. 时间成本

- 加载模型: ~2-3分钟
- 合并: ~1-2分钟
- 保存: ~1-2分钟
- **总计**: ~5-7分钟

---

## 🔍 故障排查

### 错误: 基础模型路径不存在

```bash
# 检查路径
ls -la Qwen2.5-1.5B-Instruct/qwen/Qwen2___5-1___5B-Instruct/
```

### 错误: SFT LoRA路径不存在

```bash
# 检查SFT LoRA
ls -la model_output/qwen2_5_1_5b_instruct_sft/adapter_config.json
```

### 错误: CUDA out of memory

降低模型精度：

修改脚本中的：
```python
torch_dtype=torch.float16  # 改为 torch.float32 或 torch.bfloat16
```

或使用CPU（慢但稳定）：
```python
device_map="cpu"  # 改为 cpu
```

### 警告: 仍然是LoRA适配器

如果看到此警告，说明合并失败。检查：
1. PEFT版本: `pip show peft`
2. 是否调用了 `merge_and_unload()`

---

## 📚 参考

- [模型合并原理](./dpo_model_guide.md)
- [DPO训练配置](./dpo_training_config.md)
- [评估配置](./evaluation_config_guide.md)

---

**更新时间**: 2026-02-01  
**版本**: v1.0
