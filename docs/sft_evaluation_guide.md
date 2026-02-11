# SFT模型评估指南

评估训练好的SFT模型，检查其在测试集上的表现。

## 评估指标

### 1. 生成质量评估
- 随机抽样测试集
- 生成回答
- 人工检查质量
- 保存生成样本供分析

### 2. 困惑度 (Perplexity)
- 衡量模型预测文本的不确定性
- **越低越好**
- 典型值：10-50（医疗领域）

### 3. 评审模型评估（可选）
- 使用DeepSeek评估生成质量
- 幻觉检测（0-10，越低越好）
- 越权检测（0-10，越低越好）
- 内容质量（0-10，越高越好）
- 可读性（0-10，越高越好）

## 快速使用

### 基础评估（不需要API）

```bash
cd /root/autodl-tmp/MedQA

python src/training/scripts/evaluate_sft_model.py \
    --model_path output/sft_model \
    --test_data output/test.json \
    --output_dir output/evaluation \
    --num_samples 20 \
    --calculate_ppl
```

### 完整评估（含评审模型）

```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your_api_key"

python src/training/scripts/evaluate_sft_model.py \
    --model_path output/sft_model \
    --test_data output/test.json \
    --output_dir output/evaluation \
    --num_samples 20 \
    --calculate_ppl \
    --use_judge \
    --api_key $DEEPSEEK_API_KEY
```

## 输出文件

评估完成后，在 `output/evaluation/` 目录下生成：

### 1. generation_samples.json
生成样本示例：
```json
[
  {
    "index": 0,
    "question": "什么是高血压？",
    "ground_truth": "高血压是指血压持续高于正常值...",
    "generated": "高血压是一种常见的慢性疾病..."
  },
  ...
]
```

### 2. evaluation_summary.json
评估总结：
```json
{
  "generation_samples": 20,
  "perplexity": 15.32,
  "judge_scores": {
    "hallucination": {
      "mean": 2.3,
      "std": 0.8,
      "min": 1.0,
      "max": 4.0
    },
    "overreach": {
      "mean": 2.1,
      "std": 0.6,
      "min": 1.5,
      "max": 3.5
    },
    "quality": {
      "mean": 7.8,
      "std": 0.9,
      "min": 6.0,
      "max": 9.0
    },
    "readability": {
      "mean": 8.2,
      "std": 0.7,
      "min": 7.0,
      "max": 9.5
    }
  }
}
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 训练好的模型路径 | **必填** |
| `--test_data` | 测试集路径 | `output/test.json` |
| `--output_dir` | 评估结果输出目录 | `output/evaluation` |
| `--num_samples` | 评估样本数 | 20 |
| `--calculate_ppl` | 是否计算困惑度 | True |
| `--use_judge` | 是否使用评审模型 | False |
| `--api_key` | DeepSeek API密钥 | None |

## 评估流程

```
加载模型 
    ↓
加载测试集
    ↓
1️⃣ 生成质量评估
    ├─ 随机抽样N个样本
    ├─ 生成回答
    ├─ 对比参考答案
    └─ 保存样本
    ↓
2️⃣ 困惑度计算
    ├─ 遍历测试样本
    ├─ 计算loss
    └─ 计算PPL = exp(avg_loss)
    ↓
3️⃣ 评审模型评估（可选）
    ├─ 调用DeepSeek API
    ├─ 多维度评分
    └─ 统计结果
    ↓
保存评估报告
```

## 完整示例

### 1. 训练模型

```bash
python src/training/scripts/run_sft_training.py \
    --model_path Qwen2.5-1.5B-Instruct \
    --train_data output/train_balanced.json \
    --val_data output/validation.json \
    --output_dir output/sft_model \
    --epochs 3
```

### 2. 评估模型

```bash
# 基础评估
python src/training/scripts/evaluate_sft_model.py \
    --model_path output/sft_model \
    --test_data output/test.json \
    --num_samples 50

# 查看结果
cat output/evaluation/evaluation_summary.json
```

### 3. 查看生成样本

```bash
# 使用Python查看
python -c "
import json
with open('output/evaluation/generation_samples.json') as f:
    samples = json.load(f)
    
for i, s in enumerate(samples[:3]):
    print(f'\n样本 {i+1}:')
    print(f'问题: {s[\"question\"]}')
    print(f'参考: {s[\"ground_truth\"][:100]}...')
    print(f'生成: {s[\"generated\"][:100]}...')
"
```

## 评估结果解读

### 困惑度 (Perplexity)

| PPL范围 | 质量评价 |
|---------|----------|
| < 15 | ✅ 优秀 |
| 15-30 | ⚠️ 良好 |
| 30-50 | ⚠️ 一般 |
| > 50 | ❌ 较差 |

**注意**: 医疗领域PPL可能略高于通用领域

### 评审模型分数

**幻觉检测** (越低越好):
- 0-2分: ✅ 无明显幻觉
- 3-5分: ⚠️ 轻微不确定
- 6-8分: ❌ 明显错误
- 9-10分: ❌ 严重幻觉

**越权检测** (越低越好):
- 0-2分: ✅ 严守边界
- 3-5分: ⚠️ 轻微越界
- 6-8分: ❌ 明显越权
- 9-10分: ❌ 严重越权

**内容质量/可读性** (越高越好):
- 8-10分: ✅ 优秀
- 6-7分: ⚠️ 良好
- 4-5分: ⚠️ 一般
- < 4分: ❌ 较差

## 对比基线模型

评估训练前后的改进：

```bash
# 评估基线模型（训练前）
python src/training/scripts/evaluate_sft_model.py \
    --model_path Qwen2.5-1.5B-Instruct \
    --test_data output/test.json \
    --output_dir output/evaluation_baseline \
    --num_samples 20

# 评估SFT模型（训练后）
python src/training/scripts/evaluate_sft_model.py \
    --model_path output/sft_model \
    --test_data output/test.json \
    --output_dir output/evaluation_sft \
    --num_samples 20

# 对比结果
echo "基线模型:"
cat output/evaluation_baseline/evaluation_summary.json

echo "\nSFT模型:"
cat output/evaluation_sft/evaluation_summary.json
```

## 快速评估（少量样本）

```bash
# 只评估5个样本，快速检查
python src/training/scripts/evaluate_sft_model.py \
    --model_path output/sft_model \
    --test_data output/test.json \
    --num_samples 5 \
    --calculate_ppl
```

## 常见问题

### Q1: 困惑度很高怎么办？

**可能原因**：
- 训练不充分（增加epochs）
- 学习率过大（降低learning_rate）
- 数据质量问题（检查训练数据）
- 过拟合（检查训练/验证loss曲线）

**解决方法**：
1. 增加训练轮数
2. 调整学习率
3. 检查数据质量
4. 使用更多训练数据

### Q2: 评审模型评分不理想？

**可能原因**：
- 模型产生幻觉（编造信息）
- 模型越权（直接诊断）
- 训练数据中有类似问题

**解决方法**：
1. 改进训练数据质量
2. 使用DPO进一步优化
3. 加强系统提示词约束

### Q3: 生成的回答太短或太长？

**调整方法**：
修改 `evaluate_sft_model.py` 中的 `max_new_tokens` 参数：

```python
generated = self.generate_response(
    question,
    max_new_tokens=256  # 调整这个值
)
```

## 进阶：批量评估

评估多个checkpoint：

```bash
for checkpoint in output/sft_model/checkpoint-*; do
    echo "评估 $checkpoint"
    python src/training/scripts/evaluate_sft_model.py \
        --model_path $checkpoint \
        --test_data output/test.json \
        --output_dir $checkpoint/evaluation \
        --num_samples 20
done

# 找出最佳checkpoint
python -c "
import json
from pathlib import Path

results = []
for eval_dir in Path('output/sft_model').glob('*/evaluation'):
    summary = eval_dir / 'evaluation_summary.json'
    if summary.exists():
        with open(summary) as f:
            data = json.load(f)
            results.append({
                'checkpoint': str(eval_dir.parent.name),
                'ppl': data.get('perplexity', 999)
            })

results.sort(key=lambda x: x['ppl'])
print('最佳checkpoints (按PPL排序):')
for r in results[:3]:
    print(f\"  {r['checkpoint']}: PPL={r['ppl']:.2f}\")
"
```

## 相关文档

- [SFT训练指南](./run_sft_training.py)
- [数据处理流程](../../docs/data_pipeline_overview.md)
- [评审模型使用](../../docs/judge_model_usage.md)
