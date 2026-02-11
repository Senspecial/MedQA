# 从原始数据到SFT训练数据 - 完整流程

## 📋 流程概览（正确版本）

```
原始数据 (merged_data.json)
    ↓
【步骤1】数据过滤 (Data Filtering)
    ├─ 格式转换
    ├─ 隐私过滤
    ├─ 文本清洗
    └─ 去重
    ↓
【步骤2】数据标注 (Data Annotation)
    ├─ DeepSeek API 自动标注
    ├─ 打一级标签 (primary_label)
    ├─ 多维度评分 (safety, relevance, etc.)
    └─ 质量过滤
    ↓
【步骤3】数据集拆分 (Dataset Split) ⚠️ 关键步骤
    ├─ 训练集 (80%)
    ├─ 验证集 (10%)
    └─ 测试集 (10%)
    ↓
【步骤4】数据配比 (Data Balancing) ⚠️ 只对训练集！
    ├─ 分析训练集标签分布
    ├─ 选择配比策略
    └─ 平衡训练集各类别数量
    ↓
SFT训练数据
├─ train_balanced.json (配比后)
├─ validation.json (保持原始分布)
└─ test.json (保持原始分布)
```

## ⚠️ 重要：为什么要先拆分后配比？

### ❌ 错误做法：先配比后拆分
```
全部数据配比 → 拆分train/val/test
```
**问题**：
1. **数据泄露**：过采样会复制样本，同一样本可能出现在train和test中
2. **测试集失真**：test集被配比后不反映真实分布
3. **评估不准确**：无法准确评估模型在真实场景的表现

### ✅ 正确做法：先拆分后配比
```
拆分train/val/test → 只对train配比
```
**好处**：
1. **避免数据泄露**：train/val/test完全独立
2. **测试集真实**：保持原始分布，准确评估
3. **验证集可选**：可配比（用于调参）或不配比（用于监控）

## 🔧 详细步骤

### 步骤1: 数据过滤

**目标**: 清洗和标准化原始数据

**功能**:
1. **格式转换**: 
   - 输入: `{"question": "...", "answer": "..."}`
   - 统一为标准格式

2. **隐私过滤**:
   - 检测: 身份证、手机号、邮箱、地址等
   - 处理: 脱敏（`***`）或直接丢弃

3. **文本清洗**:
   - 去除特殊字符
   - 标准化标点符号
   - 过滤过短/过长文本

4. **去重**:
   - 基于 `question + answer` 的 MD5
   - 保留唯一样本

**运行命令**:
```bash
python src/training/scripts/run_data_filter_with_config.py \
    --max_samples 200  # 测试用，生产环境去掉此参数
```

**配置文件**: `config/data_filter_config.yaml`
```yaml
deepseek:
  enable_annotation: false  # 步骤1不标注
privacy_filter:
  enabled: true
  strict_mode: false  # 脱敏而不是丢弃
quality_filter:
  enabled: false  # 步骤1不过滤质量
```

**输出**:
- `output/train.json` - 训练集
- `output/validation.json` - 验证集
- `output/test.json` - 测试集
- `output/filter_report.json` - 统计报告

**数据示例**:
```json
{
  "question": "什么是高血压？",
  "answer": "高血压是指血压持续高于正常值...",
  "id": "d14a9d0a8dce9c4bc0b4ea8a2f94eb11",
  "domain": "medical"
}
```

---

### 步骤2: 数据标注

**目标**: 为数据添加标签和质量评分

**功能**:
1. **一级标签** (`primary_label`):
   - 医学科普
   - 疾病机制
   - 检查解释
   - 症状咨询
   - 药物信息
   - 通用寒暄
   - 其他

2. **多维度评分** (`scores`):
   - `safety`: 安全性 (0-10)
   - `relevance`: 相关性 (0-10)
   - `authenticity`: 真实性 (0-10)
   - `uncertainty`: 不确定性 (0-10, 越低越好)
   - `helpfulness`: 帮助性 (0-10)
   - `overall_score`: 综合分数

3. **标注原因** (`annotation_reason`):
   - AI给出的详细评价

4. **质量过滤**:
   - 根据评分阈值过滤低质量数据
   - 只保留 `filter_passed: true` 的样本

**运行命令**:
```bash
# 设置API密钥
export DEEPSEEK_API_KEY="your_api_key"

python src/training/scripts/run_data_filter_with_config.py
```

**配置文件**: `config/data_filter_config.yaml`
```yaml
deepseek:
  api_key: ""  # 或从环境变量读取
  enable_annotation: true  # 🔑 步骤2启用标注

quality_filter:
  enabled: true  # 🔑 步骤2启用质量过滤
  thresholds:
    safety: 6.0
    relevance: 7.0
    authenticity: 6.0
    uncertainty_max: 7.0
    helpfulness: 6.0
    overall: 6.5
```

**输出**:
- 带标签和评分的训练数据

**数据示例**:
```json
{
  "question": "什么是高血压？",
  "answer": "高血压是指血压持续高于正常值...",
  "id": "d14a9d0a8dce9c4bc0b4ea8a2f94eb11",
  "domain": "medical",
  "primary_label": "医学科普",  // 🆕 添加的标签
  "scores": {  // 🆕 添加的评分
    "safety": 9,
    "relevance": 9,
    "authenticity": 8,
    "uncertainty": 3,
    "helpfulness": 8
  },
  "annotation_reason": "回答准确、全面，提供了高血压的定义...",
  "annotated": true,
  "overall_score": 8.2,
  "filter_passed": true,  // 通过质量过滤
  "filter_reasons": []
}
```

---

### 步骤3: 数据配比

**目标**: 平衡各类别的样本数量

**为什么需要配比**:
```
标注后的分布可能是:
  医学科普: 5000条 ✗ 太多
  疾病机制: 200条  ✗ 太少
  症状咨询: 3000条
  ...
```

不平衡的数据会导致模型偏向数量多的类别。

**配比方法**:

1. **均匀配比** (推荐用于测试):
   - 所有类别样本数相同
   - 例如: 每个类别都配比到1500个

2. **按比例配比** (推荐用于生产):
   - 重要类别占比更高
   - 例如: 疾病机制30%、症状咨询25%...

3. **其他方法**:
   - 按数量: 为每个类别指定具体数量
   - 最小限制: 确保每个类别至少N个样本
   - 最大限制: 限制每个类别最多N个样本

**运行命令**:
```bash
# 查看可用配置
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --list-configs

# 执行配比
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config balanced_training
```

**配置文件**: `config/data_balance_config.yaml`
```yaml
# 均匀配比
balanced_training:
  method: "uniform"
  target_count: 1500

# 按比例配比
focused_training:
  method: "ratios"
  target_ratios:
    "疾病机制": 0.30
    "症状咨询": 0.25
    "医学科普": 0.20
    "药物信息": 0.15
    "检查解释": 0.08
    "通用寒暄": 0.02
  total_samples: 10000
  strategy: "smart"
```

**输出**:
- `output/train_balanced.json` - 配比后的训练数据
- `output/train_balanced_balance_stats.json` - 配比统计

**配比前后对比**:
```
配比前:
  医学科普: 5000 (50%)
  疾病机制: 200  (2%)
  症状咨询: 3000 (30%)
  ...

配比后 (uniform, target_count=1500):
  医学科普: 1500 (20%)  ← 欠采样
  疾病机制: 1500 (20%)  ← 过采样
  症状咨询: 1500 (20%)  ← 欠采样
  ...
```

---

## 🚀 完整工作流（当前最佳实践）

### 推荐方案：两步走

**步骤1: 数据过滤+标注+拆分**
```bash
cd /root/autodl-tmp/MedQA

# 编辑配置: config/data_filter_config.yaml
# - enable_annotation: false (测试) 或 true (生产)
# - quality_filter.enabled: false (测试) 或 true (生产)

# 运行（会自动拆分为 train/val/test）
python src/training/scripts/run_data_filter_with_config.py --max_samples 200

# 输出:
# - output/train.json (80%)
# - output/validation.json (10%)
# - output/test.json (10%)
```

**步骤2: 只对训练集配比**
```bash
# 对训练集进行配比
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config balanced_training

# 最终用于训练的数据:
# ✅ output/train_balanced.json - 配比后的训练集
# ✅ output/validation.json - 原始分布的验证集
# ✅ output/test.json - 原始分布的测试集
```

**步骤3: 使用数据训练**
```bash
python src/training/scripts/run_sft.py \
    --train_data output/train_balanced.json \
    --val_data output/validation.json \
    --test_data output/test.json
```

---

## 📊 数据流转示例

### 输入 (原始数据)
```json
{
  "question": "什么是高血压？",
  "answer": "高血压是血压高的病..."
}
```

### 步骤1输出 (过滤后)
```json
{
  "question": "什么是高血压？",
  "answer": "高血压是血压高的病...",
  "id": "abc123...",
  "domain": "medical"
}
```

### 步骤2输出 (标注后)
```json
{
  "question": "什么是高血压？",
  "answer": "高血压是血压高的病...",
  "id": "abc123...",
  "domain": "medical",
  "primary_label": "医学科普",
  "scores": {
    "safety": 8,
    "relevance": 9,
    "authenticity": 7,
    "uncertainty": 4,
    "helpfulness": 8
  },
  "overall_score": 7.6,
  "annotated": true,
  "filter_passed": true
}
```

### 步骤3输出 (配比后)
- 保持所有字段不变
- 调整各类别的样本数量
- 通过过采样（复制）或欠采样（删除）达到目标分布

---

## ⚙️ 配置文件总览

### `config/data_filter_config.yaml`

控制步骤1和步骤2:

```yaml
# 基本配置
data_file: "/root/autodl-tmp/MedQA/merged_data.json"
output_dir: "/root/autodl-tmp/MedQA/output"
max_workers: 4

# DeepSeek API配置（步骤2）
deepseek:
  api_key: ""
  enable_annotation: false  # 步骤1: false, 步骤2: true

# 隐私过滤（步骤1）
privacy_filter:
  enabled: true
  strict_mode: false

# 质量过滤（步骤2）
quality_filter:
  enabled: false  # 步骤1: false, 步骤2: true
  thresholds:
    safety: 6.0
    relevance: 7.0
    overall: 6.5

# 输出格式
output:
  save_csv: true
  save_json: true
  save_chat_format: true
  save_instruction_format: true
  generate_report: true
```

### `config/data_balance_config.yaml`

控制步骤3:

```yaml
# 均匀配比
balanced_training:
  method: "uniform"
  target_count: 1500

# 按比例配比
focused_training:
  method: "ratios"
  target_ratios:
    "疾病机制": 0.30
    "症状咨询": 0.25
    "医学科普": 0.20
  total_samples: 10000
  strategy: "smart"
```

---

## 📈 数据质量指标

### 步骤1后的指标
- 原始样本数
- 过滤掉的样本数（隐私、格式、长度）
- 去重样本数
- 保留样本数

### 步骤2后的指标
- 标注样本数
- 通过质量过滤的样本数
- 各维度平均分数
- 各标签分布

### 步骤3后的指标
- 配比前后各标签数量
- 过采样/欠采样的样本数
- 最终训练数据总量

---

## 🎯 最佳实践

### 1. 测试阶段
```bash
# 只处理少量数据测试流程
python src/training/scripts/run_data_filter_with_config.py --max_samples 200
python src/training/scripts/run_data_balance.py --input output/train.json --output output/train_balanced.json
```

### 2. 生产环境

**步骤1: 数据过滤** (不需要API)
```bash
# 配置: enable_annotation=false, quality_filter.enabled=false
python src/training/scripts/run_data_filter_with_config.py
# 输出: output/train.json (已过滤、去重)
```

**步骤2: 数据标注** (需要API，较慢)
```bash
# 配置: enable_annotation=true, quality_filter.enabled=true
export DEEPSEEK_API_KEY="your_key"
python src/training/scripts/run_data_filter_with_config.py
# 输出: output/train.json (带标签和评分)
```

**步骤3: 数据配比** (快速)
```bash
python src/training/scripts/run_data_balance.py \
    --input output/train.json \
    --output output/train_balanced.json \
    --config focused_training
# 输出: output/train_balanced.json (最终训练数据)
```

### 3. 成本优化

**标注成本**:
- DeepSeek API: 约¥0.001/样本
- 10万样本: 约¥100
- 建议: 分批处理，先处理高质量子集

**时间成本**:
- 步骤1: 几分钟（10万样本）
- 步骤2: 几小时（取决于API速度）
- 步骤3: 几秒钟

---

## 📚 相关文档

- [数据清洗快速开始](./data_cleaning_quickstart.md)
- [数据配比使用指南](./data_balance_usage.md)
- [评审模型使用指南](./judge_model_usage.md)
- [DPO负样本构造](../src/training/dataset/README_DPO.md)

---

## 🔍 总结

```
原始数据 (merged_data.json)
    ↓ 步骤1: 数据过滤
    ├─ 格式统一、隐私脱敏、文本清洗、去重
    ↓
过滤数据 (train.json)
    ↓ 步骤2: 数据标注
    ├─ DeepSeek API标注、多维度评分、质量过滤
    ↓
标注数据 (train.json with labels)
    ↓ 步骤3: 数据配比
    ├─ 分析分布、选择策略、平衡数量
    ↓
训练数据 (train_balanced.json)
    ↓
SFT模型训练 ✨
```

这就是从原始数据到SFT训练数据的完整流程！🎉
