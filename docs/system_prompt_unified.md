# 系统提示统一说明（v3.0 规范版）

## ✅ 已完成统一

所有配置文件和脚本中的系统提示已完全统一为 **v3.0 规范版**。

## 📋 统一范围

### 配置文件（5个）
1. ✅ `config/system_prompt.yaml` - 主模板文件
2. ✅ `config/dpo_training_config.yaml` - DPO训练配置
3. ✅ `config/dpo_construction_config.yaml` - DPO样本构造配置
4. ✅ `config/evaluation_config.yaml` - SFT评估配置
5. ✅ `config/dpo_evaluation_config.yaml` - DPO评估配置

### Python脚本（2个）
6. ✅ `src/training/trainer/run_sft.py` - SFT训练脚本
7. ✅ `src/training/dataset/medical_dataset.py` - 数据集类（动态加载）

## 📊 验证结果

```
文件类型              长度        状态
─────────────────────────────────────────
主模板文件            1567字符    ✅ v3.0
DPO训练配置          1567字符    ✅ v3.0
DPO构造配置          1567字符    ✅ v3.0
SFT评估配置          1567字符    ✅ v3.0
DPO评估配置          1567字符    ✅ v3.0
SFT训练脚本          1567字符    ✅ v3.0
数据集类              动态加载    ✅ v3.0
```

**一致性检查**: ✅ 所有文件完全一致

## 🎯 v3.0 规范版特点

### 结构清晰
- 六大模块：核心职责、回答规范、安全红线、回答示范、质量标准、特殊提醒
- 每个模块层次分明，易于理解

### 纯文本格式
- 完全避免特殊符号（✓、✗、##、### 等）
- 使用中文标点和结构
- 不会被模型误认为格式模板

### 具体可操作
- 提供正面示范："应该说..."
- 提供反面示例："不要说..."
- 明确禁止行为和原因

### 安全优先
- 详细的急诊情况列表
- 明确的就医时机分级
- 强调不确定性处理

## 📝 系统提示内容概览

```
你是一个专业的医疗知识助手，具备全科医学知识，旨在为用户提供准确、安全、实用的医疗健康信息。

一、核心职责
1. 提供医学知识科普和健康信息咨询
2. 分析症状并给出可能的原因（使用不确定性表述）
3. 建议合适的就医科室、检查项目和就医时机
4. 解答用药相关疑问（不开具处方）
5. 提供疾病预防和健康管理建议

二、回答规范
【必须遵守的原则】
1. 使用不确定性表述
2. 可以提供的建议
3. 严格禁止的行为

三、安全红线
【立即建议急诊】9类情况
【建议24小时内就医】7类情况
【可以观察1-2天】4类情况
【不确定时的处理】4条原则

四、回答示范
【症状分析】正反示例
【检查建议】示例
【用药建议】示例
【生活建议】示例

五、回答质量标准
1. 准确性 2. 清晰性 3. 完整性 4. 安全性 5. 实用性

六、特殊提醒
4条关键提醒，强调不能替代医生
```

## 🔄 如何保持同步

### 方法1: 使用同步脚本

```python
import yaml

# 读取主模板
with open('config/system_prompt.yaml', 'r', encoding='utf-8') as f:
    prompt = yaml.safe_load(f)['system_prompt']

# 同步到所有配置文件
for config_file in ['dpo_training_config.yaml', 'evaluation_config.yaml', 
                    'dpo_evaluation_config.yaml', 'dpo_construction_config.yaml']:
    path = f'config/{config_file}'
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['system_prompt'] = prompt
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

# 同步到 run_sft.py
import re
with open('src/training/trainer/run_sft.py', 'r', encoding='utf-8') as f:
    content = f.read()
pattern = r'system_prompt = """.*?"""'
replacement = f'system_prompt = """{prompt}"""'
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
with open('src/training/trainer/run_sft.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ 同步完成")
```

### 方法2: 只修改主模板，其他自动加载

`medical_dataset.py` 已实现自动加载机制：
```python
# 当不指定system_prompt时，自动从config/system_prompt.yaml加载
dataset = MedicalDataset(data_path)  # 自动使用最新版本
```

## ⚠️ 注意事项

1. **修改系统提示时**：
   - 优先修改 `config/system_prompt.yaml`
   - 然后运行同步脚本更新所有文件
   - 验证所有文件一致性

2. **避免格式污染**：
   - 不要使用 ✓、✗、##、### 等符号
   - 不要使用特殊格式，模型可能会复制到输出
   - 使用纯文本、中文标点、缩进结构

3. **保持简洁**：
   - 当前1567字符是经过优化的长度
   - 既包含必要信息，又不会过长
   - 过长会占用太多context

## 📈 版本历史

### v3.0 规范版（当前）
- ✅ 完全纯文本，无符号
- ✅ 1567字符，116行
- ✅ 六大模块结构
- ✅ 已应用到所有7个位置

### v2.1 无格式符号版
- ⚠️ 移除了 ✓ ✗ 符号
- ⚠️ 但结构不够清晰

### v2.0 完善版
- ❌ 包含大量 ✓ ✗ 符号
- ❌ 导致模型输出格式混乱
- ❌ 已废弃

### v1.0 简化版
- ❌ 仅200字符
- ❌ 信息不足
- ❌ 已废弃

## 🎉 总结

✅ **所有config文件和脚本的系统提示已完全统一**
✅ **使用v3.0规范版（1567字符）**
✅ **纯文本格式，避免格式污染**
✅ **结构清晰，安全优先**

---

更新日期：2026-02-01  
版本：v3.0 规范版  
状态：已部署到所有位置
