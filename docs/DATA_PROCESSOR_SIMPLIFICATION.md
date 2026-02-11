# 数据处理功能更新总结

## 已移除的功能

### 1. 医疗问答过滤 (`filter_medical_qa`)
- **移除原因**：不再需要基于关键词的医疗问答过滤
- **影响**：所有标准化后的数据都会被保留，不再进行医疗关键词匹配过滤

### 2. 数据增强 (`augment_qa` 和 `_generate_question_variants`)
- **移除原因**：不再需要自动生成问题变体和摘要
- **影响**：
  - 不再生成问题的变体（如改写、添加礼貌用语等）
  - 不再生成答案摘要
  - 统计信息中移除了 `augmented_samples` 字段
  - 每个原始样本只产生一条处理后的数据

### 3. 格式转换功能
- **移除的方法**：
  - `convert_to_chat_format()` - 转换为聊天格式
  - `convert_all_to_chat_format()` - 批量转换聊天格式
  - `create_instruction_data()` - 创建指令微调格式
- **移除的命令行参数**：
  - `--chat_format`
  - `--instruction_format`
- **影响**：不再自动生成 `*_chat.json` 和 `train_instructions.json` 文件

## 更新后的数据处理流程

### 简化后的流程

```
原始数据
  ↓
标准化格式 (normalize_qa_format)
  ↓
隐私过滤 (privacy_filter)
  ↓
标注 (可选，DeepSeek API)
  ↓
质量过滤 (可选，基于评分)
  ↓
数据配比 (可选)
  ↓
拆分数据集 (train/val/test)
  ↓
保存 (CSV + JSON)
```

### 保留的核心功能

✅ **隐私信息过滤**
- 检测和脱敏个人隐私信息

✅ **数据标准化**
- 统一问答格式
- 文本清洗

✅ **DeepSeek标注**（可选）
- 一级标签分类
- 五维度质量评分

✅ **质量过滤**（可选）
- 基于评分的智能过滤

✅ **数据配比**（新增）
- 按标签平衡数据集

✅ **测试样本评估**（新增）
- 评估回答质量
- 检测越权和幻觉问题

## 输出文件变化

### 之前输出的文件
- train.csv / train.json
- validation.csv / validation.json
- test.csv / test.json
- train_chat.json ❌ 已移除
- validation_chat.json ❌ 已移除
- test_chat.json ❌ 已移除
- train_instructions.json ❌ 已移除
- filter_report.json

### 现在输出的文件
- train.csv / train.json
- validation.csv / validation.json
- test.csv / test.json
- filter_report.json
- evaluated_samples.json（如果进行了评估）
- evaluation_report.json（如果进行了评估）
- passed_samples.json（如果进行了评估）
- problem_samples.json（如果进行了评估）

## 统计信息变化

### 之前的统计字段
```python
{
    "processed_files": 10,
    "processed_samples": 5000,
    "filtered_samples": 500,
    "augmented_samples": 3000,  # ❌ 已移除
    "privacy_filtered": 50,
    "quality_filtered": 200,
    "annotated_samples": 4500
}
```

### 现在的统计字段
```python
{
    "processed_files": 10,
    "processed_samples": 5000,
    "filtered_samples": 500,
    "privacy_filtered": 50,
    "quality_filtered": 200,
    "annotated_samples": 4500
}
```

## 使用示例

### 基本使用
```bash
python data_processor.py \
    --data_dir data/raw \
    --output_dir data/processed \
    --deepseek_api_key $DEEPSEEK_API_KEY \
    --enable_annotation \
    --workers 4
```

### Python代码使用
```python
from data_processor import MedicalDataProcessor

processor = MedicalDataProcessor(
    data_dir="data/raw",
    output_dir="data/processed",
    deepseek_api_key="your_api_key",
    enable_privacy_filter=True,
    enable_quality_filter=True
)

# 处理数据
datasets = processor.process_all_data(enable_annotation=True)

# 数据配比（可选）
balance_config = {
    "method": "uniform",
    "target_count": 1000
}
datasets = processor.process_all_data(
    enable_annotation=True,
    balance_config=balance_config
)

# 评估测试样本（可选）
report = processor.evaluate_test_samples("data/test_samples.json")
```

## 性能提升

由于移除了数据增强功能，处理速度会显著提升：

- ✅ **处理速度更快**：不需要生成多个变体
- ✅ **数据量更小**：每个样本只有一条记录
- ✅ **质量更高**：专注于原始数据的质量控制
- ✅ **更易管理**：输出文件更少，结构更清晰

## 迁移建议

如果你之前依赖这些功能，建议：

1. **聊天格式转换**
   - 可以在训练时动态转换
   - 或使用独立脚本进行转换

2. **指令微调格式**
   - 可以在训练管道中处理
   - 或使用独立脚本生成

3. **数据增强**
   - 如需要可以在训练时进行在线增强
   - 或使用专门的数据增强工具

## 兼容性说明

✅ **向后兼容**：原有的数据加载和处理逻辑保持不变
✅ **更简洁**：代码更易维护和理解
✅ **更专注**：专注于核心的数据质量控制

---

**更新完成时间**：2026-01-30
**影响的功能**：数据增强、格式转换、医疗问答过滤
**保留的功能**：所有核心数据处理和质量控制功能
