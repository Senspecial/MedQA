import json  
import numpy as np  
from tqdm import tqdm  
from datasets import load_dataset  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import torch  
from torch.utils.data import DataLoader  
import evaluate
from peft import PeftModel

from src.config.config import (
    MODEL_PATH,
    SFT_MODEL_PATH,
    DPO_MODEL_PATH,
    SFT_DPO_MODEL_PATH,
    SFT_GRPO_MODEL_PATH,
    TOKENIZER_PATH,
)

class MedicalQAEvaluator:  
    def __init__(self, model_path, tokenizer_path, device="cuda:0", base_model_path=None):
        """  
        初始化评估器  
        :param model_path: 微调后的模型路径  
        :param tokenizer_path: 分词器路径  
        :param device: 计算设备 (default: cuda:0)  
        """  
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if base_model_path is None:
                base_model_path = "/sakura/sjs/models/Qwen2.5-1.5B/qwen/Qwen2___5-1___5B"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(self.device)

        try:
            print(f"Trying to load LoRA adapter from: {model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                model_path,
            )
        except Exception as e:
            print(f"[Warning] Failed to load LoRA from {model_path}, use base model only. Error: {e}")
            self.model = base_model
        self.model.eval()

        
        # 初始化评估指标  
        self.bleu = evaluate.load("bleu")  
        self.rouge = evaluate.load("rouge")  
        self.bertscore = evaluate.load("bertscore")  
        
        
        # 生成参数配置  
        self.generation_config = {  
            "max_new_tokens": 512,  
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id  
        }  
        
    
    def format_prompt(self, query):
        """构建模型输入格式"""
        system_prompt = "你是一个专业的医疗助手，请根据你的知识回答以下医疗问题。"
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    
    
    def evaluate_dataset(self, dataset_path, batch_size=4, max_samples=100):  
        """  
        评估数据集  
        :param dataset_path: 数据集路径（本地或HuggingFace）  
        :param batch_size: 批量大小  
        :param max_samples: 最大评估样本数（调试用）  
        :return: 评估指标字典  
        """  
        # 加载数据  
        dataset = load_dataset("json", data_files=dataset_path)["train"]  
        dataset = dataset.select(range(min(max_samples, len(dataset))))  
        dataloader = DataLoader(dataset, batch_size=batch_size)  

        results = {  
            "perplexity": [],  
            "bleu": [],  
            "rouge": [],  
            "bertscore": []  
        }  

        with torch.no_grad():  
            for batch in tqdm(dataloader, desc="Evaluating"):  
                # 生成回答
                query = batch["query"]
                #references = batch["response"]
                prompts = [self.format_prompt(q) for q in query]
                inputs = self.tokenizer(  
                    prompts,  
                    return_tensors="pt",  
                    padding=True,  
                    truncation=True,  
                    max_length=1024  
                ).to(self.device)  
                
                # 生成文本  
                outputs = self.model.generate(  
                    **inputs,
                    **self.generation_config  
                )

                predictions = self.tokenizer.batch_decode(  
                    outputs[:, inputs["input_ids"].shape[1]:],   
                    skip_special_tokens=True  
                )  
                
                # 计算困惑度  
                loss = self.model(  
                    inputs["input_ids"],  
                    labels=inputs["input_ids"]  
                ).loss  
                perplexity = torch.exp(loss).item()  
                results["perplexity"].append(perplexity)  

                # 计算文本相似度指标  
                references = [r.strip() for r in batch["response"]]
                pairs = [
                    (p.strip(), r)
                    for p, r in zip(predictions, references)
                    if len(r) > 0
                ]

                if pairs:
                    preds_bleu, refs_bleu = zip(*pairs)  # 拆成两组列表
                    # evaluate 的 bleu 参考格式是：list[str] + list[list[str]]
                    bleu_result = self.bleu.compute(
                        predictions=list(preds_bleu),
                        references=[[r] for r in refs_bleu],
                    )
                    results["bleu"].append(bleu_result["bleu"])
                
                rouge_scores = self.rouge.compute(  
                    predictions=predictions,  
                    references=references,  
                    rouge_types=["rougeL"],
                    use_aggregator=False,
                )  
                results["rouge"].extend(rouge_scores["rougeL"])
                
                bert_scores = self.bertscore.compute(  
                    predictions=predictions,  
                    references=references,  
                    lang="zh"  
                )  
                results["bertscore"].extend(bert_scores["f1"])  

        # 汇总结果  
        return {  
            "perplexity": np.mean(results["perplexity"]),  
            #"bleu": np.mean(results["bleu"]),
            "rougeL": np.mean(results["rouge"]),  
            "bertscore": np.mean(results["bertscore"])  
        }  

    def save_results(self, results, output_path="sft_grpo_eval_results.json"):
        """保存评估结果"""  
        with open(output_path, "w", encoding="utf-8") as f:  
            json.dump(results, f, ensure_ascii=False, indent=2) 
            
            






# 使用示例  
if __name__ == "__main__":


    evaluator = MedicalQAEvaluator(  
        model_path=SFT_GRPO_MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        base_model_path = None
    )  
    
    results = evaluator.evaluate_dataset(  
        dataset_path="/home/mry/sjs/MedQA/src/data/raw/clean_test_datasets.json",
        batch_size=4,  
        max_samples=300
    )  
    
    print("评估结果：")  
    for k, v in results.items():  
        print(f"{k}: {v:.4f}")  
    
    evaluator.save_results(results)  
    
    
    # 结果解读建议：  
    # - Perplexity < 30：优秀  
    # - BLEU > 0.25：合格  
    # - BERTScore > 0.75：优秀

#python -m src.evaluation.evaluator
