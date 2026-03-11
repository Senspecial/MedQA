import os
from modelscope import snapshot_download

# 定义模型保存路径
#model_05b_path = "/sakura/sjs/models/Qwen2.5-0.5B"
model_15b_path = "/root/autodl-tmp/MedQA/Qwen3.5-0.8B-Base"

# 自动创建目录（exist_ok=True 表示如果目录已存在也不报错）
#os.makedirs(model_05b_path, exist_ok=True)
os.makedirs(model_15b_path, exist_ok=True)

# print("📁 正在下载 Qwen2.5-0.5B...")
# snapshot_download(
#     'qwen/Qwen2.5-0.5B',
#     cache_dir=model_05b_path
# )

#print("✅ Qwen2.5-0.5B 下载完成！")

print("📁 正在下载 Qwen/Qwen3.5-0.8B-Base...")
snapshot_download(
    'Qwen/Qwen3.5-0.8B-Base',
    cache_dir=model_15b_path
)

print("✅ Qwen3.5-0.8B-Base 下载完成！")
print("🎉 所有模型已保存至 /root/autodl-tmp/MedQA/ 目录。")