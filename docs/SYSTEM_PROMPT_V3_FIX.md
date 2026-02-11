# 系统提示更新说明 v3.0

## 问题诊断

用户反馈模型生成的答案中包含：
1. 转义字符：`\n\n`
2. 格式符号：`✓ "示例内容"`
3. 示例模板：直接复制系统提示中的示例

## 根本原因

之前的系统提示（v2.0）中使用了大量的格式符号作为示例：
```
✓ "根据您的症状，可能是感冒或流感引起的"
✓ "这种情况常见于消化不良，也可能是胃炎的表现"
```

模型在训练/推理时可能会：
1. 把这些符号当作输出格式的一部分
2. 直接复制这些示例内容
3. 学习了这种格式模式

## 解决方案

### 1. 更新系统提示（v3.0 规范版）

**核心改进**：
- ✅ 完全移除所有格式符号（✓ ✗ ## ### 等）
- ✅ 使用纯文本描述和缩进来组织结构
- ✅ 示例改用"回答时应该说："而不是符号列表
- ✅ 清晰的六大模块分层

**特点**：
- 约1567字符，116行
- 结构：一、二、三...（中文数字）
- 小标题：【标题】（方括号）
- 示例格式：
  ```
  回答时应该说：
  - 可能是XX
  - 常见于XX
  
  不要说：
  - 就是XX
  - 确诊为XX
  ```

### 2. 添加后处理函数

在 `run_evaluation.py` 中添加 `clean_generated_response()` 函数：

```python
def clean_generated_response(text: str) -> str:
    """清理生成答案中的格式问题"""
    import re
    
    # 移除转义字符
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '\t')
    
    # 移除格式符号
    text = re.sub(r'[✓✗]\s*"[^"]*"', '', text)
    text = re.sub(r'[✓✗]\s*[^\n]+', '', text)
    
    # 移除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
```

### 3. 统一配置文件格式

将所有YAML配置文件中的 `system_prompt` 从单行转义字符串改为多行格式：

**旧格式（有问题）**：
```yaml
system_prompt: "你是...\n\n一、核心职责\n\n..."
```

**新格式（规范）**：
```yaml
system_prompt: |
  你是一个专业的医疗知识助手...
  
  一、核心职责
  
  1. 提供医学知识科普...
```

### 4. 应用位置

已更新以下所有位置：
1. ✅ `config/system_prompt.yaml` - 主模板
2. ✅ `config/dpo_training_config.yaml` - DPO训练
3. ✅ `config/dpo_construction_config.yaml` - DPO构造
4. ✅ `config/evaluation_config.yaml` - SFT评估
5. ✅ `config/dpo_evaluation_config.yaml` - DPO评估
6. ✅ `src/training/trainer/run_sft.py` - SFT训练脚本
7. ✅ `src/training/scripts/run_evaluation.py` - 评估脚本（添加后处理）

## 使用建议

### 对于现有模型
如果使用旧系统提示训练的模型：
1. 运行评估时会自动应用后处理清理
2. 建议重新训练以彻底解决问题

### 对于新训练
1. 使用更新后的系统提示进行训练
2. 模型不会学到格式符号
3. 生成的答案会更自然、更规范

### 验证方法
```bash
# 检查系统提示版本
python3 << 'EOF'
import yaml
with open('config/system_prompt.yaml', 'r') as f:
    prompt = yaml.safe_load(f)['system_prompt']
    has_v3 = '一、核心职责' in prompt and '【必须遵守的原则】' in prompt
    print(f"系统提示版本: {'✅ v3.0规范版' if has_v3 else '❌ 旧版本'}")
    print(f"长度: {len(prompt)} 字符")
EOF

# 重新评估（会自动清理格式）
python src/training/scripts/run_evaluation.py config/evaluation_config.yaml
```

## 版本历史

- **v1.0**: 简单版本（约200字符）
- **v2.0**: 详细版本（约3000字符，包含✓✗符号，导致格式问题）
- **v3.0**: 规范版本（约1567字符，纯文本，无符号污染）✅

## 文件清单

- `config/system_prompt.yaml` - 主模板
- `docs/system_prompt_application.md` - 应用说明
- `docs/system_prompt_improvement.md` - v2.0改进说明
- `docs/SYSTEM_PROMPT_V3_FIX.md` - 本文档（v3.0修复说明）
