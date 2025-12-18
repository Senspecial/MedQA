import json
import random

# 1. 读入你整理好的 json 数组
input_path = "/home/mry/sjs/MedQA/src/data/raw/clean_train_datasets.json"   # 这里换成你的文件名
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # data 是一个 list，每个元素是 {"query": ..., "response": ...}

# 2. 打乱顺序（保证划分更随机）
random.seed(42)
random.shuffle(data)

# 3. 按 70% / 30% 划分
n = len(data)
n_sft = int(n * 0.7)

sft_data = data[:n_sft]   # 前 70% 做 SFT
remaining_data = data[n_sft:]
n_dpo = int(len(remaining_data) * 0.1)  # 注意：是剩余部分的 10%
dpo_data = remaining_data[:n_dpo]

print("总样本数:", n)
print("SFT 样本数:", len(sft_data))
print("DPO 样本数:", len(dpo_data))

# 4. 分别保存成两个 json 文件
with open("sft.json", "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

with open("dpo.json", "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print("已生成 sft.json 和 dpo.json")
