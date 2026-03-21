#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SFT 模型合并（薄包装）

此脚本已被通用的 LoRA 合并工具替代，请直接使用：

    python src/training/scripts/merge_lora_model.py \\
        --base_model <基础模型路径> \\
        --lora_adapter <SFT LoRA 路径> \\
        --output <输出路径>

多 adapter 合并请使用：

    python scripts/merge_adapters.py \\
        --base-model <基础模型路径> \\
        --adapters <CPT adapter> <SFT adapter> \\
        --output <输出路径>
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.training.scripts.merge_lora_model import main

if __name__ == "__main__":
    print("[提示] 此脚本已废弃，请直接使用 src/training/scripts/merge_lora_model.py")
    print("       转发参数到通用合并工具...\n")
    main()
