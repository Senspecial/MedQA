# 系统提示应用说明

## 📍 系统提示的存储位置

### 主配置文件
**文件**: `config/system_prompt.yaml`  
**作用**: 完整的医疗AI系统提示词主模板（约3000字符，177行）

这是唯一的系统提示源文件，包含：
- 7个医疗领域的职责覆盖
- 25个推荐做法示例
- 22个禁止做法反例
- 分级就医指导
- 安全红线说明

## 📋 系统提示的应用位置

### 1️⃣ SFT训练（Python内联）
**文件**: `src/training/trainer/run_sft.py` (第32-70行)  
**方式**: 直接在Python代码中内联完整的系统提示字符串  
**用途**: SFT训练时格式化每个训练样本的prompt

```python
system_prompt = """完整的系统提示内容"""
train_ds = MedicalDataset(..., system_prompt=system_prompt)
```

### 2️⃣ DPO训练配置（YAML内联）
**文件**: `config/dpo_training_config.yaml`  
**方式**: YAML配置文件中直接包含 `system_prompt:` 字段  
**用途**: DPO训练时格式化训练数据的prompt

```yaml
system_prompt: |
  完整的系统提示内容
```

### 3️⃣ SFT评估配置（YAML内联）
**文件**: `config/evaluation_config.yaml`  
**方式**: YAML配置文件中直接包含 `system_prompt:` 字段  
**用途**: 评估SFT模型时生成回答的系统提示

### 4️⃣ DPO评估配置（YAML内联）
**文件**: `config/dpo_evaluation_config.yaml`  
**方式**: YAML配置文件中直接包含 `system_prompt:` 字段  
**用途**: 评估DPO模型时生成回答的系统提示

### 5️⃣ MedicalDataset（动态加载）
**文件**: `src/training/dataset/medical_dataset.py`  
**方式**: 从 `config/system_prompt.yaml` 动态加载  
**用途**: 当创建 `MedicalDataset` 且未指定 `system_prompt` 参数时，自动加载

```python
# 自动加载完整版
dataset = MedicalDataset(data_path)  

# 或手动指定
dataset = MedicalDataset(data_path, system_prompt="自定义提示")
```

## 🔄 系统提示的同步机制

### 当前状态（v2.0）
所有配置文件和代码中都已包含**完整的系统提示内容**（约3000字符）

✅ **优点**:
- 每个配置文件独立完整，不依赖外部文件
- 配置文件可以单独使用，便于分享和部署
- 不会出现找不到配置文件的问题

⚠️ **缺点**:
- 多处重复，修改时需要同步更新

### 如何更新系统提示

#### 方法1: 使用Python脚本（推荐）

1. 编辑主模板文件: `config/system_prompt.yaml`
2. 运行同步脚本:

```python
import yaml

# 读取主模板
with open('config/system_prompt.yaml', 'r', encoding='utf-8') as f:
    full_prompt = yaml.safe_load(f)['system_prompt']

# 更新所有配置文件
configs = [
    'config/dpo_training_config.yaml',
    'config/evaluation_config.yaml',
    'config/dpo_evaluation_config.yaml'
]

for config_file in configs:
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['system_prompt'] = full_prompt
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print("✅ 系统提示已同步到所有配置文件")
```

#### 方法2: 手动更新
分别编辑以下5个位置:
1. `config/system_prompt.yaml` - 主模板
2. `src/training/trainer/run_sft.py` - Python字符串
3. `config/dpo_training_config.yaml` - YAML字段
4. `config/evaluation_config.yaml` - YAML字段
5. `config/dpo_evaluation_config.yaml` - YAML字段

## 📊 应用验证

### 验证系统提示是否正确应用:

```python
# 测试脚本
import sys
sys.path.insert(0, '/root/autodl-tmp/MedQA')
from src.training.dataset.medical_dataset import MedicalDataset

# 创建测试数据集
test_data = [{"question": "测试", "answer": "测试"}]
dataset = MedicalDataset(test_data, dataset_type="sft")

# 检查系统提示
print(f"系统提示长度: {len(dataset.system_prompt)} 字符")
print(f"是否为完整版: {'✅' if '核心职责与领域覆盖' in dataset.system_prompt else '❌'}")
print(f"\n前300字符:\n{dataset.system_prompt[:300]}")
```

预期输出:
```
系统提示长度: 2983 字符
是否为完整版: ✅
```

## 🎯 使用场景

### 场景1: SFT训练
```bash
python -m src.training.trainer.run_sft
```
→ 使用 `run_sft.py` 中内联的系统提示

### 场景2: DPO训练
```bash
python src/training/scripts/run_dpo_training.py --config_path config/dpo_training_config.yaml
```
→ 使用 `dpo_training_config.yaml` 中的系统提示

### 场景3: SFT评估
```bash
python src/training/scripts/run_evaluation.py config/evaluation_config.yaml
```
→ 使用 `evaluation_config.yaml` 中的系统提示

### 场景4: DPO评估
```bash
python src/training/scripts/run_evaluation.py config/dpo_evaluation_config.yaml
```
→ 使用 `dpo_evaluation_config.yaml` 中的系统提示

### 场景5: 数据集处理
```python
from src.training.dataset.medical_dataset import MedicalDataset

# 自动加载 config/system_prompt.yaml
dataset = MedicalDataset("data/train.json", dataset_type="sft")

# 或手动指定
custom_prompt = "你是一个医疗助手..."
dataset = MedicalDataset("data/train.json", system_prompt=custom_prompt)
```

## 📝 版本历史

### v2.0（当前版本）- 完善版
- ✅ 扩展到约3000字符、177行
- ✅ 7个医疗领域详细说明
- ✅ 25个推荐做法 + 22个禁止做法
- ✅ 分级就医指导（急诊/24小时/可观察）
- ✅ 已应用到所有5个位置

### v1.0 - 简化版
- 约200字符、10行
- 5条简单原则
- 无具体示例

## 🔧 故障排查

### 问题1: MedicalDataset 使用简化版
**症状**: 创建 MedicalDataset 时，系统提示只有200字符  
**原因**: 找不到 `config/system_prompt.yaml` 文件  
**解决**: 确保在项目根目录运行，或检查 `config/system_prompt.yaml` 是否存在

### 问题2: 评估时使用旧版系统提示
**症状**: 评估结果显示系统提示内容很短  
**原因**: 配置文件未更新  
**解决**: 运行同步脚本更新所有配置文件

### 问题3: 训练和评估使用不同版本
**症状**: 训练通过率和评估通过率差异很大  
**原因**: 系统提示不一致  
**解决**: 确保所有5个位置使用相同的系统提示

## 📚 相关文档

- `docs/system_prompt_improvement.md` - 系统提示完善说明
- `config/system_prompt.yaml` - 主模板文件
- `docs/evaluation_config_guide.md` - 评估配置指南
- `docs/dpo_model_guide.md` - DPO模型指南
