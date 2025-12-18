import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["HF_HOME"] = "/sakura/sjs/hf_cache"
import torch

from src.training.dataset.medical_dataset import MedicalDataset
from src.training.trainer.sft_trainer import SFTTrainer


def main():
    model_name_or_path = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"
    output_dir = "./save/medical_sft_qwen2_5_1_5b"

    # 可选：覆盖默认 TrainingArguments
    custom_training_args = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 4,
        "learning_rate": 2e-5,
        "logging_steps": 10,
        "save_steps": 200,
    }

    # 1. 初始化你自己的 SFTTrainer
    trainer = SFTTrainer(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        training_args=custom_training_args,
        use_qlora=True
    )

    # 2. 准备 MedicalDataset（这里按你自己类的构造函数来写）
    medical_dataset = MedicalDataset(
        data_path="/home/mry/sjs/MedQA/sft.json",
        # 其它参数根据 MedicalDataset 的 __init__ 来填
    )

    # 3. 开始训练（内部会自动 load_model_and_tokenizer / create_trainer）
    trainer.train(medical_dataset, eval_split=0.1)


if __name__ == "__main__":
    main()

#python -m src.training.trainer.run_sft
