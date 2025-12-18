import os
import torch
from typing import Dict, List

import verl
from verl.worker.computation.compute_client import LocalComputeClient
from verl.worker.algorithm.sft.sft_worker import SFTWorker, SFTConfig
from verl.worker.algorithm.sft.sft_worker_manager import SFTWorkerManager
from verl.worker.scheduler.preemptible_scheduler import PreemptibleScheduler
from verl.worker.protocol.protocol import Protocol
from verl.trainer.policy.basic_policy_trainer import BasicPolicyTrainer
from verl.trainer.protocol.protocol import Protocol as TrainerProtocol
from verl.dataset.data_source import DataSource, SFTTask
from verl.dataset.shm_dict_dataset import ShmDictDataset
from verl.common.config import RunConfig, ModelConfig
from verl.common.metric_manager import create_metric_manager
from verl.worker.computation.remote_container import RemoteModelContainer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# 配置日志
import logging

logging.basicConfig(level=logging.INFO)

# 设置模型和训练配置
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_path = "/home/mry/sjs/MedQA/src/data/sft_data.json"  # 你的 SFT 数据集路径（格式见下文）
output_dir = "./qwen_sft_output"
os.makedirs(output_dir, exist_ok=True)

# 运行配置
run_config = RunConfig(
    experiment_name="qwen-1.5b-sft-medical",
    run_name="run1",
    seed=42,
    resume=False,
    output_dir=output_dir,
)

# 模型配置
model_config = ModelConfig(
    model_name_or_path=model_name,
    trust_remote_code=True,
)


# 1. 加载 SFT 数据集
def load_sft_dataset():
    """
    加载 SFT 数据集，期望格式为：
    [
        {
            "instruction": "什么是胃食管反流？",
            "input": "",
            "output": "胃食管反流病（GERD）是指..."
        },
        ...
    ]
    """
    import json
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for item in data:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        # 构造对话历史（Qwen 格式）
        messages = [
            {"role": "user", "content": f"{instruction}\n{input_text}".strip()},
            {"role": "assistant", "content": output}
        ]
        tasks.append(SFTTask(messages=messages))

    return tasks


sft_tasks = load_sft_dataset()
data_source = DataSource(tasks=sft_tasks)

# 创建共享内存数据集
dataset = ShmDictDataset()
dataset.add_data(data_source.as_dict())

# 2. 设置模型和 Tokenizer（与 GRPO 脚本一致）

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # Qwen2.5 的关键模块
)


def create_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 确保 chat_template 存在（Qwen 官方已内置，但保险起见）
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}"

    return model, tokenizer


model, tokenizer = create_model_and_tokenizer()

# 3. 配置 SFT 算法参数
sft_config = SFTConfig(
    lr=2e-5,
    mini_batch_size=4,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    use_lora=True,
    lora_config=peft_config.to_dict(),
    response_template="<|im_start|>assistant\n",  # 用于 loss mask
    metrics_to_monitor=[],
)

# 4. 创建 Compute Client 和 Remote Model Container
compute_client = LocalComputeClient(
    device="cuda",
    mixed_precision="fp16",
)

remote_model_container = RemoteModelContainer(
    model=model,
    tokenizer=tokenizer,
    device_map="cuda",
)

# 5. 创建 SFT Worker
sft_worker = SFTWorker(
    config=sft_config,
    compute_client=compute_client,
    dataset=dataset,
    remote_model_container=remote_model_container,
    protocol=Protocol(),
)

# 6. 创建 Worker Manager 和 Scheduler
worker_manager = SFTWorkerManager(
    workers=[sft_worker],
    dataset=dataset,
)

scheduler = PreemptibleScheduler(
    worker_manager=worker_manager,
    epochs=sft_config.num_train_epochs,
    steps_per_epoch=500,  # 可根据数据集大小调整
    eval_every_n_steps=sft_config.eval_steps,
)

# 7. 配置指标管理器
metric_manager = create_metric_manager(
    experiment_name=run_config.experiment_name,
    run_name=run_config.run_name,
    output_dir=run_config.output_dir,
    use_wandb=False,
)

# 8. 创建 Policy Trainer
trainer = BasicPolicyTrainer(
    model=model,
    tokenizer=tokenizer,
    protocol=TrainerProtocol(),
    compute_client=compute_client,
    scheduler=scheduler,
    metric_manager=metric_manager,
    run_config=run_config,
)

# 9. 开始训练
trainer.train()

# 10. 保存最终模型
model.save_pretrained(os.path.join(output_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

print("SFT 训练完成！模型已保存到:", os.path.join(output_dir, "final_model"))

#python src/training/trainer/sft_from_verl.py