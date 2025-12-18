import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer


# 设置Hugging Face token (如果需要访问私有模型或上传结果)
# os.environ["HF_TOKEN"] = "你的HF token"

# 定义模型和数据集
model_name = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"
dataset_name = "/home/mry/sjs/MedQA/src/data/sft_data.json"  # 替换为你的SFT数据集路径

# 使用BitsAndBytes进行量化，节省显存（取消注释以启用4-bit）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 为QLoRA准备模型
model = prepare_model_for_kbit_training(model)

# 定义LoRA配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen2.5 推荐模块
    # 如果你确定旧版用 c_attn/c_proj/w1/w2，也可保留，但新Qwen2架构已改名
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token

# 确保添加了特殊的Qwen指令格式（虽然Qwen2.5通常自带）
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n<|im_start|>user\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'assistant' %}\n<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n{% elif message['role'] == 'system' %}\n<|im_start|>system\n{{ message['content'] }}<|im_end|>\n{% endif %}\n{% endfor %}"

# 加载数据集
dataset = load_dataset("json", data_files=dataset_name)

train_dataset = dataset["train"]

# 定义数据处理函数（假设原始字段为: instruction, input, output）
def preprocess_function(examples):
    messages_list = []
    for i in range(len(examples["instruction"])):
        user_content = f"{examples['instruction'][i]}\n{examples.get('input', [''])[i]}".strip()
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": examples["output"][i]}
        ]
        messages_list.append(messages)
    return {"messages": messages_list}

# 应用数据预处理
processed_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

# （可选）限制训练样本数量用于调试
# processed_dataset = processed_dataset.select(range(1000))
def formatting_func(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


# 定义训练参数
training_args = TrainingArguments(
    output_dir="./save/qwen2_5_1_5b_instruct_sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="tensorboard",
    seed=42,
    push_to_hub=False,
)

# 创建SFT训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    #dataset_text_field="text",  # 实际不用，因为我们用 formatting_func
    formatting_func=formatting_func,
    max_seq_length=2048,
    packing=False,  # 对话长度不一，建议关闭
)

# 开始训练
trainer.train()

# 保存最终模型（仅LoRA权重）
trainer.save_model("./save/qwen2_5_1_5b_instruct_sft/final_lora")

# 如果需要保存到HF Hub
# trainer.push_to_hub()

#python src/training/trainer/sft_from_trl.py
