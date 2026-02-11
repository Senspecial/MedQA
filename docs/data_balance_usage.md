# 数据配比使用指南

## 什么是数据配比

数据配比是为了解决训练数据中类别不平衡的问题。例如：
- "医学科普"有5000条
- "疾病机制"只有200条
- "症状咨询"有3000条

不平衡的数据会导致模型偏向数量多的类别，影响训练效果。

## 快速开始

### 1. 查看当前数据分布

```bash
cd /root/autodl-tmp/MedQA

python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --list-configs
```

### 2. 执行数据配比

```bash
# 使用默认配置（均匀配比）
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json

# 使用指定配置
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config balanced_training
```

## 配比方法

### 方法1: 均匀配比 (uniform)

让所有类别的样本数相同。

**适用场景**: 类别严重不平衡，需要平等对待所有类别

**配置示例**:
```yaml
balanced_training:
  method: "uniform"
  target_count: 1500  # 每个类别都配比到1500个样本
```

**示例**:
```
原始分布:
  医学科普: 3000  
  疾病机制: 500   
  症状咨询: 2000  

配比后:
  医学科普: 1500  
  疾病机制: 1500  (过采样：复制样本)
  症状咨询: 1500  
```

### 方法2: 按比例配比 (ratios)

按照指定的比例分配各类别。

**适用场景**: 某些类别更重要，需要不同权重

**配置示例**:
```yaml
focused_training:
  method: "ratios"
  target_ratios:
    "疾病机制": 0.30  # 30%
    "症状咨询": 0.25  # 25%
    "医学科普": 0.20  # 20%
    "药物信息": 0.15  # 15%
    "检查解释": 0.08  # 8%
    "通用寒暄": 0.02  # 2%
  total_samples: 10000
  strategy: "smart"
```

**示例**:
```
目标: 总共10000个样本，疾病机制占30%

配比后:
  疾病机制: 3000  (30%)
  症状咨询: 2500  (25%)
  医学科普: 2000  (20%)
  ...
```

### 方法3: 按数量配比 (counts)

为每个类别指定具体数量。

**适用场景**: 需要精确控制每个类别的样本数

**配置示例**:
```yaml
method_counts:
  method: "counts"
  target_counts:
    "医学科普": 2500
    "疾病机制": 2000
    "检查解释": 1500
    "症状咨询": 2000
    "药物信息": 1500
    "通用寒暄": 500
  strategy: "smart"
```

### 方法4: 最小样本数限制 (min)

确保每个类别至少有指定数量的样本。

**适用场景**: 避免稀有类别样本过少

**配置示例**:
```yaml
minimum_coverage:
  method: "min"
  min_samples: 300  # 每个类别至少300个样本
```

**示例**:
```
原始分布:
  医学科普: 5000  → 保持 5000
  疾病机制: 100   → 补充到 300 (过采样)
  症状咨询: 2000  → 保持 2000

配比后:
  医学科普: 5000
  疾病机制: 300
  症状咨询: 2000
```

### 方法5: 最大样本数限制 (max)

限制每个类别最多有指定数量的样本。

**适用场景**: 控制数据总量，避免某类过多

**配置示例**:
```yaml
scale_control:
  method: "max"
  max_samples: 2000  # 每个类别最多2000个样本
```

**示例**:
```
原始分布:
  医学科普: 5000  → 下采样到 2000
  疾病机制: 800   → 保持 800
  症状咨询: 3000  → 下采样到 2000

配比后:
  医学科普: 2000
  疾病机制: 800
  症状咨询: 2000
```

## 采样策略

### oversample (过采样)
- 样本不足时：复制现有样本达到目标
- 优点：不丢失信息
- 缺点：可能导致过拟合

### undersample (欠采样)
- 样本过多时：随机删除样本达到目标
- 优点：不引入重复
- 缺点：可能丢失信息

### smart (智能采样) - 推荐
- 自动选择：样本少用过采样，样本多用欠采样
- 平衡效果和信息保留

## 使用示例

### 场景1: 快速测试（均匀配比）

```bash
# 配比训练集（每个类别配比到相同数量）
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config balanced_training

# 配比验证集
python src/training/scripts/run_data_balance.py \
    --input output/validation.json \
    --output output/validation_balanced.json \
    --config balanced_training
```

### 场景2: 重点突出某些类别

```bash
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_focused.json \
    --config focused_training
```

### 场景3: 确保最小覆盖

```bash
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_min.json \
    --config minimum_coverage
```

### 场景4: 控制数据规模

```bash
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_scaled.json \
    --config scale_control
```

## 查看结果

配比完成后会生成两个文件：

1. **配比后的数据**: `output/train_balanced.json`
   - 可以直接用于训练

2. **统计报告**: `output/train_balanced_balance_stats.json`
   - 包含原始分布、配比后分布、配比参数等信息

```bash
# 查看统计报告
cat output/train_balanced_balance_stats.json
```

## 自定义配置

编辑 `config/data_balance_config.yaml`，添加你自己的配置：

```yaml
# 自定义配置
my_custom_config:
  method: "ratios"
  target_ratios:
    "疾病机制": 0.40
    "症状咨询": 0.30
    "医学科普": 0.20
    "药物信息": 0.10
  total_samples: 5000
  strategy: "smart"
```

使用自定义配置：

```bash
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_custom.json \
    --config my_custom_config
```

## 在数据清洗中集成配比

也可以在数据清洗时直接配比：

```python
from src.training.dataset.data_processor import MedicalDataProcessor

processor = MedicalDataProcessor(
    data_dir="data/raw",
    output_dir="output"
)

# 配比配置
balance_config = {
    "method": "uniform",
    "target_count": 1500
}

# 处理时配比
datasets = processor.process_all_data(
    enable_annotation=False,
    balance_config=balance_config  # 传入配比配置
)
```

## 注意事项

### 1. 过采样的影响

- ✅ 优点：保留所有信息，适合数据稀缺
- ⚠️ 缺点：会导致样本重复，可能过拟合
- 💡 建议：配合数据增强使用

### 2. 欠采样的影响

- ✅ 优点：不引入重复，避免过拟合
- ⚠️ 缺点：丢失信息，可能欠拟合
- 💡 建议：确保保留的样本有代表性

### 3. 配比时机

**推荐顺序**:
```
数据清洗 → 质量过滤 → 去重 → 配比 → 训练集拆分
```

### 4. 验证集和测试集

- 验证集：可以配比（确保各类别都能评估）
- 测试集：建议保持原始分布（反映真实场景）

## 完整工作流

```bash
# 1. 数据清洗
python src/training/scripts/run_data_filter_with_config.py \
    --max_samples 200

# 2. 查看分布
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --list-configs

# 3. 配比训练集
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config balanced_training

# 4. 配比验证集
python src/training/scripts/run_data_balance.py \
    --input output/validation.json \
    --output output/validation_balanced.json \
    --config balanced_training

# 5. 测试集保持原始分布（可选配比）

# 6. 使用配比后的数据训练
python src/training/scripts/run_sft.py \
    --train_data output/train_balanced.json \
    --val_data output/validation_balanced.json
```

## 常见问题

### Q1: 配比后数据量变化很大？

这是正常的。配比方法会增加或减少样本数：
- 过采样会增加总量
- 欠采样会减少总量
- 智能采样会权衡两者

### Q2: 某个类别样本太少怎么办？

方案：
1. 使用过采样（复制样本）
2. 收集更多数据
3. 使用数据增强
4. 考虑合并相似类别

### Q3: 配比会影响模型性能吗？

- 正面影响：减少类别偏见，提高少数类的性能
- 注意事项：过度配比可能导致模型不适应真实分布
- 建议：在真实分布的测试集上验证

## 相关文档

- [数据清洗快速开始](./data_cleaning_quickstart.md)
- [配置文件说明](../config/data_balance_config.yaml)
