# DPO 负样本构造 - 代码重构说明

## 📋 重构内容

将 `dpo_negative_constructor.py` 从 692 行精简到 **225 行**，只保留基础组件。

## 🗂️ 文件结构

### 旧版本 (692行)
```
dpo_negative_constructor.py
├── ResponseCandidate (数据类) ✅
├── DPOSample (数据类) ✅
├── JudgeModel (评审模型类) ✅
└── DPONegativeConstructor (构造器类) ❌ 删除
    ├── __init__() - 加载模型
    ├── generate_responses() - 生成回答
    ├── evaluate_and_rank_responses() - 评估排序
    ├── select_chosen_rejected_pair() - 选择对
    ├── construct_dpo_samples() - 主流程
    ├── _load_sft_data()
    ├── _extract_prompt()
    └── _save_dpo_samples()
```

### 新版本 (225行)
```
dpo_negative_constructor.py (基础组件)
├── ResponseCandidate (数据类) ✅
├── DPOSample (数据类) ✅
└── JudgeModel (评审模型类) ✅

run_dpo_construction.py (运行脚本)
├── 导入基础组件
├── 加载SFT模型 (支持LoRA)
├── 生成回答 (多策略)
├── 评估排序 (使用JudgeModel)
├── 选择对 (优化逻辑)
└── 主流程
```

## 🎯 保留的组件

### 1. ResponseCandidate (数据类)
```python
@dataclass
class ResponseCandidate:
    response: str                    # 回答文本
    score: float                     # 综合得分
    hallucination_score: float       # 幻觉分数
    overreach_score: float           # 越权分数
    quality_score: float             # 质量分数
    readability_score: float         # 可读性分数
    details: Dict[str, Any]          # 详细信息
```

**作用**: 存储候选回答及其评分

### 2. DPOSample (数据类)
```python
@dataclass
class DPOSample:
    prompt: str                      # 问题
    chosen: str                      # 好的回答
    rejected: str                    # 差的回答
    chosen_score: float              # chosen得分
    rejected_score: float            # rejected得分
    metadata: Dict[str, Any]         # 元数据
```

**作用**: 存储DPO训练样本对

### 3. JudgeModel (评审模型类)
```python
class JudgeModel:
    def __init__(self, api_key, base_url, model)
    def evaluate_response(prompt, response) -> Dict
    def _call_api(prompt) -> Dict
```

**作用**: 调用DeepSeek API评估回答质量

**评分维度**:
- `hallucination_score` (0-10) - 幻觉检测
- `overreach_score` (0-10) - 越权检测
- `quality_score` (0-10) - 内容质量
- `readability_score` (0-10) - 可读性

## ❌ 删除的组件

### DPONegativeConstructor 类

**删除原因**:
1. ❌ 功能有限，不支持LoRA模型
2. ❌ 生成策略固定，不够灵活
3. ❌ 选择逻辑简单，未优化
4. ❌ 已被 `run_dpo_construction.py` 完全替代

## ✅ 优势

### 旧架构的问题
```python
# 使用 DPONegativeConstructor（不灵活）
constructor = DPONegativeConstructor(
    sft_model_path="...",
    judge_api_key="...",
    num_responses=4,
    temperature=0.8
)
constructor.construct_dpo_samples(...)
```

- ❌ 参数硬编码
- ❌ 不支持LoRA
- ❌ 生成策略单一
- ❌ 选择逻辑固定

### 新架构的优势
```python
# 使用配置文件 + 组件化（灵活）
from dpo_negative_constructor import JudgeModel, ResponseCandidate, DPOSample

judge_model = JudgeModel(api_key=..., base_url=...)
# 自己实现生成逻辑（支持LoRA、多策略）
# 自己实现选择逻辑（Top-K + 安全优先）
```

- ✅ 配置文件驱动
- ✅ 支持LoRA模型
- ✅ 多种生成策略
- ✅ 优化的选择逻辑
- ✅ 组件可复用

## 📦 使用方式

### 导入基础组件
```python
from src.training.dataset.dpo_negative_constructor import (
    ResponseCandidate,  # 候选回答数据类
    DPOSample,         # DPO样本数据类
    JudgeModel         # 评审模型
)
```

### 使用评审模型
```python
# 初始化
judge = JudgeModel(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com/v1"
)

# 评估回答
scores = judge.evaluate_response(
    prompt="什么是高血压？",
    response="高血压是指..."
)

# scores 结构：
{
    "hallucination_score": 2.0,
    "overreach_score": 1.5,
    "quality_score": 8.0,
    "readability_score": 7.5,
    "overall_comment": "...",
    "specific_issues": []
}
```

### 创建数据对象
```python
# 创建候选回答
candidate = ResponseCandidate(
    response="...",
    score=6.5,
    hallucination_score=2.0,
    overreach_score=1.5,
    quality_score=8.0,
    readability_score=7.5,
    details={}
)

# 创建DPO样本
dpo_sample = DPOSample(
    prompt="...",
    chosen="...",
    rejected="...",
    chosen_score=7.5,
    rejected_score=4.0,
    metadata={}
)
```

## 📈 代码行数对比

| 组件 | 旧版本 | 新版本 | 变化 |
|------|--------|--------|------|
| ResponseCandidate | 14行 | 18行 | +4 (增加注释) |
| DPOSample | 13行 | 18行 | +5 (增加注释) |
| JudgeModel | 139行 | 189行 | +50 (增加注释) |
| DPONegativeConstructor | 487行 | **0行** | -487 (删除) |
| main() | 39行 | **0行** | -39 (删除) |
| **总计** | **692行** | **225行** | **-467行 (-67%)** |

## 🎓 设计原则

遵循 **单一职责原则**：
- `dpo_negative_constructor.py` - 只负责提供基础组件
- `run_dpo_construction.py` - 负责实现业务逻辑

这样的设计：
1. ✅ **更清晰** - 职责明确
2. ✅ **更灵活** - 易于扩展
3. ✅ **更可维护** - 代码简洁
4. ✅ **更可复用** - 组件独立

## 🔄 迁移指南

如果有代码使用了旧的 `DPONegativeConstructor`：

### 旧代码
```python
from src.training.dataset.dpo_negative_constructor import DPONegativeConstructor

constructor = DPONegativeConstructor(...)
constructor.construct_dpo_samples(...)
```

### 新代码
```python
# 使用配置文件驱动的脚本
python src/training/scripts/run_dpo_construction.py \
    --config config/dpo_construction_config.yaml
```

或者手动调用：
```bash
bash scripts/run_dpo_construction.sh
```

## 📝 总结

这次重构：
- 🧹 **精简了代码** - 从692行减少到225行
- 🎯 **明确了职责** - 基础组件 vs 业务逻辑
- 🔧 **提高了灵活性** - 配置驱动 + 组件化
- 📦 **保持了兼容** - 导入路径不变
- ✨ **改进了功能** - 支持LoRA + 优化逻辑

**结果**: 更简洁、更灵活、更易维护的代码结构！🎉
