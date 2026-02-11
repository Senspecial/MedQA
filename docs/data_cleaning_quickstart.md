# 数据清洗快速开始指南

## 1. 配置文件说明

配置文件位于：`config/data_filter_config.yaml`

### 关键配置项：

```yaml
# 数据源
data_file: "/root/autodl-tmp/MedQA/merged_data.json"  # 单个数据文件
# 或使用 data_dir 指定包含多个数据文件的目录

# 输出目录
output_dir: "/root/autodl-tmp/MedQA/output"

# 功能开关
deepseek:
  enable_annotation: false  # DeepSeek标注（需要API key，较慢）
  
privacy_filter:
  enabled: true  # 隐私过滤（推荐开启）
  strict_mode: false  # false=脱敏，true=直接丢弃
  
quality_filter:
  enabled: false  # 质量过滤（需要先启用标注）
```

## 2. 运行方式

### 方式1：使用Shell脚本（推荐）

```bash
# 使用默认配置
bash scripts/run_clean.sh

# 使用自定义配置
bash scripts/run_clean.sh --config config/my_config.yaml
```

### 方式2：直接运行Python脚本

```bash
# 使用默认配置
python src/training/scripts/run_data_filter_with_config.py

# 使用自定义配置
python src/training/scripts/run_data_filter_with_config.py --config config/data_filter_config.yaml
```

## 3. 处理流程

```
原始数据 → 格式转换 → 隐私过滤 → (可选)标注 → (可选)质量过滤 → 数据集拆分 → 多格式输出
```

### 3.1 数据格式转换

支持的输入格式：
- `{"question": "...", "answer": "..."}`
- `{"instruction": "...", "output": "..."}`
- `{"prompt": "...", "response": "..."}`
- `{"query": "...", "response": "..."}`

统一转换为：`{"question": "...", "answer": "..."}`

### 3.2 隐私过滤

自动检测和处理以下隐私信息：
- 身份证号
- 手机号
- 邮箱地址
- 详细地址
- 姓名（在特定上下文中）
- 其他敏感信息

**strict_mode=false**：脱敏处理（替换为 `***`）
**strict_mode=true**：直接丢弃包含隐私信息的样本

### 3.3 数据集拆分

- **训练集（train）**：80%
- **验证集（validation）**：10%
- **测试集（test）**：10%

## 4. 输出文件

处理完成后，在 `output_dir` 目录下会生成：

### 基础格式
- `train.csv` / `train.json` - 训练集
- `validation.csv` / `validation.json` - 验证集
- `test.csv` / `test.json` - 测试集

### 聊天格式（如果配置中启用）
- `train_chat.json`
- `validation_chat.json`
- `test_chat.json`

格式示例：
```json
[
  {
    "messages": [
      {"role": "user", "content": "什么是高血压？"},
      {"role": "assistant", "content": "高血压是指..."}
    ]
  }
]
```

### 指令格式（如果配置中启用）
- `train_instructions.json`

格式示例：
```json
[
  {
    "instruction": "什么是高血压？",
    "input": "",
    "output": "高血压是指..."
  }
]
```

### 统计报告
- `filter_report.json` - 包含处理统计信息

## 5. 示例：基础清洗（无标注）

### 5.1 修改配置文件

```yaml
data_file: "/root/autodl-tmp/MedQA/merged_data.json"
output_dir: "/root/autodl-tmp/MedQA/output"

deepseek:
  enable_annotation: false  # 不使用API

privacy_filter:
  enabled: true  # 启用隐私过滤
  strict_mode: false  # 脱敏而不是丢弃

quality_filter:
  enabled: false  # 不启用质量过滤
```

### 5.2 运行

```bash
bash scripts/run_clean.sh
```

### 5.3 预期输出

```
找到 1 个数据文件
原始数据样本数: 173179
格式化后样本数: 173179
处理数据文件: 100%|████████| 1/1
文件处理完成: data.json, 原始样本: 173179, 处理后: 150000+, 过滤: 20000+

训练集: 120000+ 样本
验证集: 15000+ 样本  
测试集: 15000+ 样本
```

## 6. 示例：完整清洗（含标注和质量过滤）

### 6.1 设置API密钥

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

### 6.2 修改配置文件

```yaml
deepseek:
  enable_annotation: true  # 启用API标注
  
quality_filter:
  enabled: true  # 启用质量过滤
  thresholds:
    safety: 6.0  # 安全性最低分
    relevance: 7.0  # 相关性最低分
    overall: 6.5  # 总体平均分最低分
```

### 6.3 运行

```bash
bash scripts/run_clean.sh
```

**注意**：启用标注会调用API，处理速度较慢，建议先用少量数据测试。

## 7. 常见问题

### Q1: 处理后数据为空（n_samples=0）

**原因**：
- 数据格式不正确
- 过滤太严格（strict_mode=true + 大量隐私信息）
- 质量过滤阈值过高

**解决**：
1. 检查原始数据格式
2. 设置 `privacy_filter.strict_mode: false`
3. 禁用质量过滤先测试
4. 查看日志中的过滤统计

### Q2: 处理速度慢

**原因**：
- 启用了DeepSeek标注（需要API调用）
- 数据量太大

**解决**：
1. 先禁用标注（`enable_annotation: false`）快速处理
2. 需要标注时，可以分批处理
3. 调整 `max_workers` 增加并发

### Q3: API调用失败

**原因**：
- API密钥无效
- 网络问题
- API限流

**解决**：
1. 检查API密钥：`echo $DEEPSEEK_API_KEY`
2. 检查配置文件中的 `api_key`
3. 降低 `batch_size` 和 `max_workers`
4. 添加重试机制（代码已内置）

### Q4: 内存不足

**原因**：
- 数据文件太大

**解决**：
1. 分批处理：将大文件拆分成多个小文件
2. 降低 `max_workers`
3. 使用流式处理（需要修改代码）

## 8. 数据质量检查

处理完成后，建议检查：

```bash
# 查看样本数量
wc -l output/*.csv

# 查看前几个样本
head -n 5 output/train.json

# 查看统计报告
cat output/filter_report.json
```

## 9. 进阶：自定义过滤规则

如果需要自定义过滤规则，可以修改：

`src/training/dataset/data_processor.py`

- `PrivacyFilter` 类：自定义隐私检测规则
- `QualityFilter` 类：自定义质量评分标准
- `normalize_qa_format` 方法：自定义数据格式转换

## 10. 性能参考

基于 173,179 样本的实测（单机）：

| 配置 | 时间 | 说明 |
|------|------|------|
| 仅隐私过滤 | ~5分钟 | 不使用API |
| 含标注（batch=10） | ~4-6小时 | 依赖API速度 |
| 含标注+质量过滤 | ~4-6小时 | 主要时间在标注 |

**建议**：
- 开发测试：使用少量数据（1000条），不启用标注
- 正式处理：分批处理，先运行一批观察效果
- 生产环境：考虑使用更快的API或本地模型

## 11. 后续步骤

数据清洗完成后，可以：

1. **训练SFT模型**
   ```bash
   bash train_sft.sh
   ```

2. **构造DPO数据**
   ```bash
   python src/training/scripts/run_dpo_construction.py
   ```

3. **训练DPO模型**
   ```bash
   bash train_dpo.sh
   ```
