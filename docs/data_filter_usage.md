# 数据清洗与过滤使用说明

## 功能概述

本数据处理工具提供以下功能：

1. **个人隐私信息过滤**：自动检测和过滤身份证号、手机号、邮箱、银行卡等敏感信息
2. **DeepSeek API标注**：使用DeepSeek模型自动标注一级标签
3. **多维度质量评分**：基于安全性、相关性、真实性、不确定性、帮助性五个维度进行评分
4. **智能过滤**：根据评分阈值自动过滤低质量数据

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn jieba tqdm requests
```

### 2. 准备API密钥

获取DeepSeek API密钥并设置环境变量：

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

### 3. 基本使用

#### 方式一：命令行使用

```bash
python data_processor.py \
    --data_dir /path/to/raw/data \
    --output_dir /path/to/output \
    --deepseek_api_key $DEEPSEEK_API_KEY \
    --enable_annotation \
    --enable_privacy_filter \
    --enable_quality_filter \
    --workers 4
```

#### 方式二：Python代码使用

```python
from data_processor import MedicalDataProcessor

# 初始化处理器
processor = MedicalDataProcessor(
    data_dir="data/raw",
    output_dir="data/processed",
    deepseek_api_key="your_api_key",
    enable_privacy_filter=True,
    enable_quality_filter=True,
    max_workers=4
)

# 处理数据（启用DeepSeek标注）
datasets = processor.process_all_data(enable_annotation=True)

# 可选：转换为聊天格式
processor.convert_all_to_chat_format()

# 可选：生成指令微调数据
processor.create_instruction_data()
```

## 功能详解

### 1. 隐私信息过滤

自动检测和过滤以下类型的个人隐私信息：

- **身份证号**：18位身份证号码
- **手机号**：11位手机号码（1开头）
- **座机号**：固定电话号码
- **邮箱地址**：标准邮箱格式
- **银行卡号**：16-19位银行卡号
- **IP地址**：IPv4地址格式
- **姓名信息**：姓氏+称谓模式（如：张医生、李先生）
- **详细地址**：带有门牌号的详细地址

示例：

```python
from data_processor import PrivacyFilter

filter = PrivacyFilter()

text = "我的手机号是13800138000，请联系我"
filtered_text, detected_types = filter.filter_text(text)
# filtered_text: "我的手机号是***，请联系我"
# detected_types: ["手机号"]
```

### 2. DeepSeek API标注

使用DeepSeek模型为每个问答对生成：

#### 一级标签分类
- 医学科普：普及医学知识
- 疾病机制：疾病的发病机理
- 检查解释：医学检查结果解读
- 症状咨询：症状相关问题
- 药物信息：药物使用、副作用等
- 通用寒暄：日常问候、感谢等
- 其他：无法分类的内容

#### 五维度评分（0-10分）

1. **安全性（Safety）**
   - 回答是否包含危险建议
   - 是否可能误导用户做出危险行为
   - 评分越高越安全

2. **相关性（Relevance）**
   - 回答是否切题
   - 是否回答了用户的问题
   - 评分越高越相关

3. **真实性（Authenticity）**
   - 信息是否准确可靠
   - 是否符合医学常识和科学依据
   - 评分越高越可靠

4. **不确定性（Uncertainty）**
   - 回答是否过于模糊
   - 是否给出了明确的信息
   - 评分越低越明确（注意：这个指标是反向的）

5. **帮助性（Helpfulness）**
   - 是否对用户有实际帮助
   - 信息是否有价值
   - 评分越高越有帮助

### 3. 质量过滤机制

系统提供两级过滤机制：

#### 第一级：评分阈值过滤

每个维度都有最低阈值，样本必须全部达标才能通过：

```python
# 默认阈值
thresholds = {
    "safety": 6.0,          # 安全性 ≥ 6分
    "relevance": 7.0,       # 相关性 ≥ 7分
    "authenticity": 6.0,    # 真实性 ≥ 6分
    "uncertainty_max": 7.0, # 不确定性 ≤ 7分
    "helpfulness": 6.0,     # 帮助性 ≥ 6分
    "overall": 6.5          # 总体平均分 ≥ 6.5分
}
```

#### 第二级：总体评分过滤

计算五个维度的综合得分（不确定性取反），必须达到总体阈值：

```python
overall_score = (
    safety + relevance + authenticity + 
    (10 - uncertainty) + helpfulness
) / 5
```

### 4. 输出结果

处理完成后会生成以下文件：

```
output_dir/
├── train.csv                 # 训练集（CSV格式）
├── train.json                # 训练集（JSON格式）
├── train_chat.json           # 训练集（聊天格式）
├── train_instructions.json   # 训练集（指令格式）
├── validation.csv            # 验证集
├── validation.json
├── validation_chat.json
├── test.csv                  # 测试集
├── test.json
├── test_chat.json
└── filter_report.json        # 过滤统计报告
```

#### 数据格式示例

**标准格式（JSON）：**

```json
{
  "id": "abc123...",
  "question": "什么是高血压？",
  "answer": "高血压是指动脉血压持续升高的疾病...",
  "primary_label": "医学科普",
  "scores": {
    "safety": 9,
    "relevance": 10,
    "authenticity": 9,
    "uncertainty": 2,
    "helpfulness": 9
  },
  "overall_score": 9.0,
  "filter_passed": true,
  "annotated": true
}
```

**聊天格式：**

```json
{
  "id": "abc123...",
  "conversations": [
    {"role": "user", "content": "什么是高血压？"},
    {"role": "assistant", "content": "高血压是指动脉血压持续升高的疾病..."}
  ],
  "primary_label": "医学科普"
}
```

### 5. 统计报告

`filter_report.json` 包含详细的过滤统计信息：

```json
{
  "timestamp": "2026-01-30T10:00:00",
  "statistics": {
    "processed_files": 10,
    "processed_samples": 5000,
    "filtered_samples": 500,
    "privacy_filtered": 50,
    "quality_filtered": 200,
    "annotated_samples": 4500
  },
  "label_distribution": {
    "医学科普": 1500,
    "疾病机制": 1000,
    "症状咨询": 1200,
    ...
  },
  "score_statistics": {
    "safety": {
      "mean": 8.2,
      "std": 1.5,
      "min": 4,
      "max": 10
    },
    ...
  }
}
```

## 高级用法

### 自定义过滤阈值

```python
from data_processor import QualityFilter

# 创建自定义过滤器
custom_filter = QualityFilter(
    safety_threshold=7.0,      # 更严格的安全性要求
    relevance_threshold=8.0,   # 更严格的相关性要求
    authenticity_threshold=7.0,
    uncertainty_max=6.0,       # 要求更明确的回答
    helpfulness_threshold=7.0,
    overall_threshold=7.0
)

# 使用自定义过滤器
processor.quality_filter = custom_filter
```

### 批量标注与过滤

```python
# 只对部分数据进行标注（节省API调用）
samples = [...] # 你的样本列表

# 标注前100个样本
annotated = processor.deepseek_annotator.batch_annotate(
    samples[:100],
    batch_size=10,
    max_workers=3
)

# 应用质量过滤
filtered = processor.quality_filter.filter_sample(annotated[0])
```

### 单独使用各个组件

```python
# 1. 只使用隐私过滤
from data_processor import PrivacyFilter

privacy_filter = PrivacyFilter()
clean_text, privacy_types = privacy_filter.filter_text(text)

# 2. 只使用DeepSeek标注
from data_processor import DeepSeekAnnotator

annotator = DeepSeekAnnotator(api_key="xxx")
annotation = annotator.annotate_qa(question, answer)

# 3. 只使用质量过滤
from data_processor import QualityFilter

quality_filter = QualityFilter()
passed, reason = quality_filter.filter_by_scores(sample)
```

## 注意事项

1. **API调用限制**：DeepSeek API有速率限制，建议设置合理的并发数（max_workers=3）
2. **成本控制**：标注会消耗API调用额度，建议先在小数据集上测试
3. **隐私过滤精度**：正则匹配可能有误判，建议人工抽查
4. **阈值调整**：根据实际数据质量调整过滤阈值
5. **数据备份**：处理前备份原始数据

## 性能优化

- 使用多线程并行处理文件（`--workers 4`）
- 批量标注减少API调用开销（`batch_size=10`）
- 关闭不需要的功能（如不需要标注可不传API key）
- 对大数据集分批处理

## 常见问题

**Q: 如何不使用DeepSeek标注？**

A: 不传入`deepseek_api_key`参数，或设置`enable_annotation=False`

**Q: 隐私过滤太严格怎么办？**

A: 修改`PrivacyFilter`类中的正则表达式，或调整过滤逻辑

**Q: 如何调整质量过滤标准？**

A: 在初始化时传入自定义的`QualityFilter`实例

**Q: API调用失败怎么办？**

A: 检查API密钥、网络连接，查看日志中的错误信息

## 更新日志

- v1.0.0 (2026-01-30)
  - 添加个人隐私信息过滤功能
  - 集成DeepSeek API进行自动标注
  - 实现多维度质量评分和过滤
  - 生成详细的统计报告
