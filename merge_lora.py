import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"
LORA_PATH = "/home/mry/sjs/MedQA/save/Qwen2_5-1_5B-GRPO-med-final"  # 你的最终 LoRA
OUT_PATH  = "/home/mry/sjs/MedQA/save/Qwen2_5-1_5B-medqa-merged"

def main():
    # 合并建议用 16/32 位，别用 4bit
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # CPU 也可以，慢一点但能跑
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    # 合并 LoRA 到权重里
    model = model.merge_and_unload()  # 得到普通的 AutoModelForCausalLM 实例

    # 保存合并后的模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.save_pretrained(OUT_PATH)
    tokenizer.save_pretrained(OUT_PATH)

    print("Merged model saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
