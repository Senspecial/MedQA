# 数据过滤功能更新说明

## 版本更新

### v1.1.0 (2026-01-30) - 移除风险等级功能

#### 主要变更

**移除的功能：**
- 移除了风险等级标注（低风险、中风险、高风险）
- 移除了基于风险等级的过滤逻辑
- 简化了质量过滤流程

**保留的功能：**
- ✅ 个人隐私信息过滤
- ✅ DeepSeek API一级标签标注
- ✅ 五维度质量评分（安全性、相关性、真实性、不确定性、帮助性）
- ✅ 基于评分的质量过滤

#### 代码变更详情

1. **DeepSeekAnnotator类**
   - 移除 `self.risk_levels` 属性
   - 移除API提示词中的风险等级要求
   - 标注结果不再包含 `risk_level` 字段

2. **QualityFilter类**
   - 移除 `filter_by_risk_level()` 方法
   - 简化 `filter_sample()` 方法，移除风险等级检查参数

3. **MedicalDataProcessor类**
   - 更新 `annotate_and_filter_samples()` 方法，移除风险等级过滤调用
   - 更新 `_save_filter_report()` 方法，移除风险等级统计

4. **配置文件**
   - 移除 `allowed_risk_levels` 配置项

5. **文档更新**
   - 更新使用文档，移除风险等级相关说明
   - 更新示例代码

#### 升级指南

如果您正在使用旧版本，升级时需要注意：

1. **API响应变化**
   - DeepSeek标注结果不再包含 `risk_level` 字段
   - 只保留 `primary_label` 和 `scores`

2. **配置文件更新**
   - 移除配置文件中的 `allowed_risk_levels` 设置

3. **代码调用变化**
   ```python
   # 旧版本
   filtered = quality_filter.filter_sample(
       sample,
       check_risk=True,
       allowed_risk_levels=["低风险", "中风险"]
   )
   
   # 新版本
   filtered = quality_filter.filter_sample(sample)
   ```

4. **数据格式变化**
   - 已标注的数据不再包含 `risk_level` 字段
   - 过滤报告不再包含 `risk_distribution` 统计

#### 为什么移除风险等级？

1. **简化模型**：减少标注维度，降低API调用复杂度
2. **更灵活的评估**：通过五维度评分可以更精细地评估数据质量
3. **减少主观性**：风险等级判断相对主观，评分更客观量化

#### 如果需要风险评估

您仍然可以通过评分维度自定义风险评估逻辑：

```python
def assess_risk(sample):
    scores = sample.get("scores", {})
    
    # 基于安全性评分判断风险
    safety = scores.get("safety", 0)
    
    if safety >= 8:
        return "低风险"
    elif safety >= 6:
        return "中风险"
    else:
        return "高风险"

# 应用到样本
sample["custom_risk"] = assess_risk(sample)
```

#### 兼容性说明

- ✅ 与现有数据处理流程完全兼容
- ✅ 不影响隐私过滤功能
- ✅ 不影响评分过滤功能
- ⚠️ 旧版本标注的数据中的 `risk_level` 字段将被忽略

---

## 完整功能列表（v1.1.0）

### 1. 隐私信息过滤
- 身份证号、手机号、邮箱、银行卡等敏感信息自动检测和脱敏

### 2. 一级标签分类
- 医学科普
- 疾病机制
- 检查解释
- 症状咨询
- 药物信息
- 通用寒暄
- 其他

### 3. 五维度评分
- **安全性**：是否包含危险建议（0-10分）
- **相关性**：回答是否切题（0-10分）
- **真实性**：信息是否准确可靠（0-10分）
- **不确定性**：回答是否过于模糊（0-10分，越低越好）
- **帮助性**：是否对用户有实际帮助（0-10分）

### 4. 质量过滤
- 基于评分阈值的自动过滤
- 可自定义各维度阈值
- 生成详细的过滤统计报告

---

如有任何问题，请参考完整文档：`docs/data_filter_usage.md`
