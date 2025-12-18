import torch
import os

MODEL_PATH = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'





DEEPSPEED_CONFIG_PATH = "./src/config/ds_config.json"
OUTPUT_DIR = "./output"


SFT_MODEL_NAME = "qwen2_medical_sft"
SFT_MODEL_PATH = "/home/mry/sjs/MedQA/save/medical_sft_qwen2_5_1_5b"


DPO_MODEL_NAME = "qwen2_medical_dpo"
DPO_MODEL_PATH = ""


SFT_DPO_MODEL_NAME = ""
SFT_DPO_MODEL_PATH = ""


SFT_GRPO_MODEL_PATH = "/home/mry/sjs/MedQA/save/Qwen2_5-1_5B-GRPO-med-final"
TOKENIZER_PATH = SFT_GRPO_MODEL_PATH