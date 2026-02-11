#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
医疗问答数据集处理工具
用于清洗、转换和增强医疗问答数据，为训练做准备
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union, Set
from pathlib import Path
import re
import jieba
import logging
import hashlib
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime
import time

# MinHash 相似度去重
try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    logger.warning("datasketch 未安装，MinHash去重功能不可用。安装方法: pip install datasketch")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataProcessor")


class PrivacyFilter:
    """个人隐私信息过滤器 - 强制清洗敏感信息"""
    
    def __init__(self, strict_mode: bool = True):
        """
        初始化隐私过滤器
        
        Args:
            strict_mode: 严格模式，检测到隐私信息直接拒绝样本
        """
        self.strict_mode = strict_mode
        
        # ============================================
        # 1️⃣ 身份直接标识
        # ============================================
        
        # 身份证号正则（18位）
        self.id_card_pattern = re.compile(
            r'\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dxX]\b'
        )
        
        # 护照号正则（常见格式）
        self.passport_pattern = re.compile(
            r'\b[EGP]\d{8}\b|'  # 中国护照
            r'\b[A-Z]{1,2}\d{6,9}\b'  # 其他护照
        )
        
        # 手机号正则（11位，1开头）
        self.phone_pattern = re.compile(
            r'\b1[3-9]\d{9}\b'
        )
        
        # 座机号正则（带区号）
        self.landline_pattern = re.compile(
            r'\b0\d{2,3}[-\s]?\d{7,8}\b'
        )
        
        # 微信号正则（字母数字下划线，6-20位）
        self.wechat_pattern = re.compile(
            r'(?:微信号?[:：]?\s*|微信[:：]?\s*|wx[:：]?\s*)([a-zA-Z0-9_-]{6,20})\b',
            re.IGNORECASE
        )
        
        # 个人邮箱正则
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # 社保号/医保号正则（各地格式不同，匹配常见格式）
        self.social_security_pattern = re.compile(
            r'(?:社保号?|医保号?|社会保障号?|医疗保险号?)[:：]?\s*([A-Z0-9]{15,20})\b',
            re.IGNORECASE
        )
        
        # 精确住址（门牌号）
        self.address_pattern = re.compile(
            r'(省|市|区|县|镇|乡|村|街道|路|街|巷|弄|里|苑|园|庄|小区|大厦|广场|中心|大楼)'
            r'[\u4e00-\u9fff\w]*'  # 中文字符和字母数字
            r'\d+号'  # 门牌号
            r'(?:\d+[-一]?\d*栋?)?'  # 可选：栋
            r'(?:\d+[-一]?\d*单元?)?'  # 可选：单元
            r'(?:\d+[-一]?\d*[层楼]?)?'  # 可选：楼层
            r'(?:\d+[-一]?\d*室?)?'  # 可选：房间号
        )
        
        # ============================================
        # 2️⃣ 姓名识别（增强版）
        # ============================================
        
        # 常见姓氏（前100姓）
        common_surnames = (
            '张王李刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董'
            '袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟'
            '熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤'
        )
        
        # 姓名模式：姓氏 + 称谓
        self.name_with_title_pattern = re.compile(
            f'([{common_surnames}])'
            r'(先生|女士|小姐|太太|医生|大夫|师傅|同学|老师|教授|主任|院长|副院长|'
            r'护士|护士长|患者|病人|家属|父亲|母亲|爷爷|奶奶|外公|外婆|哥哥|姐姐|'
            r'弟弟|妹妹|儿子|女儿|老公|老婆|丈夫|妻子)'
        )
        
        # 完整姓名模式：姓氏 + 1-3个名字字符
        self.full_name_pattern = re.compile(
            f'([{common_surnames}])'
            r'[\u4e00-\u9fff]{1,3}'  # 1-3个汉字作为名字
            r'(?=\s|，|。|、|的|说|讲|表示|认为|指出|提到|告诉|问|答)'  # 后面跟标点或动词
        )
        
        # ============================================
        # 3️⃣ 医疗强隐私
        # ============================================
        
        # 病历号、住院号、检查单号（通常是数字+字母组合）
        self.medical_record_pattern = re.compile(
            r'(?:病历号?|住院号?|门诊号?|就诊号?|检查号?|报告号?|单号?)[:：]?\s*([A-Z0-9]{6,20})\b',
            re.IGNORECASE
        )
        
        # 检验报告相关（包含报告二字+日期）
        self.lab_report_pattern = re.compile(
            r'(检验|化验|检查|报告).*?'
            r'(20\d{2}[年\-/\.]\d{1,2}[月\-/\.]\d{1,2}[日号]?)',
            re.IGNORECASE
        )
        
        # 精确时间+医院+科室（可定位个人）
        self.identifiable_medical_pattern = re.compile(
            r'(20\d{2}[年\-/\.]\d{1,2}[月\-/\.]\d{1,2}[日号]?)'  # 精确日期
            r'.*?'
            r'([\u4e00-\u9fff]{2,10}(?:医院|医学院|卫生院|诊所|中心))'  # 医院名
            r'.*?'
            r'([\u4e00-\u9fff]{2,6}(?:科|室|门诊|病房|中心))',  # 科室
            re.IGNORECASE
        )
        
        # 银行卡号（可能用于支付）
        self.bank_card_pattern = re.compile(
            r'\b\d{16,19}\b'
        )
        
        # IP地址
        self.ip_pattern = re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        )
        
        logger.info(f"隐私过滤器初始化完成（严格模式: {strict_mode}）")
    
    def filter_text(self, text: str, mask_char: str = "[隐私信息已脱敏]") -> Tuple[str, List[str]]:
        """
        过滤文本中的个人隐私信息
        
        Args:
            text: 原始文本
            mask_char: 替换字符
            
        Returns:
            过滤后的文本和检测到的隐私类型列表
        """
        if not text:
            return text, []
        
        detected_types = []
        filtered_text = text
        
        # 1️⃣ 身份直接标识
        
        # 身份证号
        if self.id_card_pattern.search(filtered_text):
            filtered_text = self.id_card_pattern.sub(mask_char, filtered_text)
            detected_types.append("身份证号")
        
        # 护照号
        if self.passport_pattern.search(filtered_text):
            filtered_text = self.passport_pattern.sub(mask_char, filtered_text)
            detected_types.append("护照号")
        
        # 手机号
        if self.phone_pattern.search(filtered_text):
            filtered_text = self.phone_pattern.sub(mask_char, filtered_text)
            detected_types.append("手机号")
        
        # 座机号
        if self.landline_pattern.search(filtered_text):
            filtered_text = self.landline_pattern.sub(mask_char, filtered_text)
            detected_types.append("座机号")
        
        # 微信号
        if self.wechat_pattern.search(filtered_text):
            filtered_text = self.wechat_pattern.sub(lambda m: m.group(0).split(':')[-1].split('：')[-1][:2] + mask_char, filtered_text)
            detected_types.append("微信号")
        
        # 邮箱
        if self.email_pattern.search(filtered_text):
            filtered_text = self.email_pattern.sub(mask_char, filtered_text)
            detected_types.append("邮箱")
        
        # 社保号/医保号
        if self.social_security_pattern.search(filtered_text):
            filtered_text = self.social_security_pattern.sub(lambda m: m.group(0).split(':')[-1].split('：')[-1][:2] + mask_char, filtered_text)
            detected_types.append("社保/医保号")
        
        # 精确住址
        if self.address_pattern.search(filtered_text):
            filtered_text = self.address_pattern.sub(mask_char, filtered_text)
            detected_types.append("精确住址")
        
        # 2️⃣ 姓名
        
        # 姓名+称谓
        if self.name_with_title_pattern.search(filtered_text):
            filtered_text = self.name_with_title_pattern.sub(lambda m: m.group(1) + mask_char, filtered_text)
            detected_types.append("姓名（含称谓）")
        
        # 完整姓名（更激进的过滤）
        if self.full_name_pattern.search(filtered_text):
            filtered_text = self.full_name_pattern.sub(lambda m: m.group(1) + mask_char, filtered_text)
            if "姓名" not in [t for t in detected_types if "姓名" in t]:
                detected_types.append("姓名（全名）")
        
        # 3️⃣ 医疗强隐私
        
        # 病历号/住院号/检查单号
        if self.medical_record_pattern.search(filtered_text):
            filtered_text = self.medical_record_pattern.sub(lambda m: m.group(0).split(':')[-1].split('：')[-1][:2] + mask_char, filtered_text)
            detected_types.append("病历号/住院号")
        
        # 检验报告（含日期）
        if self.lab_report_pattern.search(filtered_text):
            filtered_text = self.lab_report_pattern.sub(mask_char, filtered_text)
            detected_types.append("检验报告")
        
        # 精确时间+医院+科室（可定位个人）
        if self.identifiable_medical_pattern.search(filtered_text):
            filtered_text = self.identifiable_medical_pattern.sub(mask_char, filtered_text)
            detected_types.append("可定位医疗信息")
        
        # 其他
        
        # 银行卡号
        if self.bank_card_pattern.search(filtered_text):
            filtered_text = self.bank_card_pattern.sub(mask_char, filtered_text)
            detected_types.append("银行卡号")
        
        # IP地址
        if self.ip_pattern.search(filtered_text):
            filtered_text = self.ip_pattern.sub(mask_char, filtered_text)
            detected_types.append("IP地址")
        
        return filtered_text, detected_types
    
    def has_privacy_info(self, text: str) -> bool:
        """
        检测文本是否包含隐私信息（严格检查）
        
        Args:
            text: 待检测文本
            
        Returns:
            是否包含隐私信息
        """
        if not text:
            return False
        
        # 所有关键隐私模式
        critical_patterns = [
            self.id_card_pattern,
            self.passport_pattern,
            self.phone_pattern,
            self.landline_pattern,
            self.wechat_pattern,
            self.email_pattern,
            self.social_security_pattern,
            self.address_pattern,
            self.medical_record_pattern,
            self.identifiable_medical_pattern,
            self.bank_card_pattern
        ]
        
        # 检查是否匹配任何关键隐私模式
        for pattern in critical_patterns:
            if pattern.search(text):
                return True
        
        # 在严格模式下，也检查姓名
        if self.strict_mode:
            if self.name_with_title_pattern.search(text) or self.full_name_pattern.search(text):
                return True
        
        return False
    
    def get_privacy_report(self, text: str) -> Dict[str, Any]:
        """
        生成详细的隐私检测报告
        
        Args:
            text: 待检测文本
            
        Returns:
            隐私检测报告
        """
        report = {
            "has_privacy": False,
            "privacy_types": [],
            "details": {}
        }
        
        if not text:
            return report
        
        # 检测各类隐私信息
        checks = {
            "身份证号": self.id_card_pattern,
            "护照号": self.passport_pattern,
            "手机号": self.phone_pattern,
            "座机号": self.landline_pattern,
            "微信号": self.wechat_pattern,
            "邮箱": self.email_pattern,
            "社保/医保号": self.social_security_pattern,
            "精确住址": self.address_pattern,
            "姓名（含称谓）": self.name_with_title_pattern,
            "姓名（全名）": self.full_name_pattern,
            "病历号/住院号": self.medical_record_pattern,
            "检验报告": self.lab_report_pattern,
            "可定位医疗信息": self.identifiable_medical_pattern,
            "银行卡号": self.bank_card_pattern,
            "IP地址": self.ip_pattern
        }
        
        for privacy_type, pattern in checks.items():
            matches = pattern.findall(text)
            if matches:
                report["has_privacy"] = True
                report["privacy_types"].append(privacy_type)
                report["details"][privacy_type] = len(matches) if isinstance(matches, list) else 1
        
        return report


class DeepSeekAnnotator:
    """使用DeepSeek API进行数据标注"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek标注器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 一级标签定义
        self.primary_labels = [
            "医学科普",
            "疾病机制", 
            "检查解释",
            "症状咨询",
            "药物信息",
            "通用寒暄",
            "其他"
        ]
        
        logger.info("DeepSeek标注器初始化完成")
    
    def annotate_qa(self, question: str, answer: str, retry: int = 3) -> Optional[Dict[str, Any]]:
        """
        使用DeepSeek API为问答对打标签
        
        Args:
            question: 问题文本
            answer: 回答文本
            retry: 重试次数
            
        Returns:
            标注结果字典
        """
        prompt = f"""请对以下医疗问答进行标注，需要提供：
1. 一级标签（从以下选项中选择一个）：医学科普、疾病机制、检查解释、症状咨询、药物信息、通用寒暄、其他
2. 五个维度的评分（0-10分）：
   - 安全性（是否包含危险建议）
   - 相关性（回答是否切题）
   - 真实性（信息是否准确可靠）
   - 不确定性（回答是否过于模糊）
   - 帮助性（是否对用户有实际帮助）

问题：{question}

回答：{answer}

请以JSON格式返回，格式如下：
{{
  "primary_label": "一级标签",
  "scores": {{
    "safety": 8,
    "relevance": 9,
    "authenticity": 7,
    "uncertainty": 6,
    "helpfulness": 8
  }},
  "reason": "简短说明（可选）"
}}"""

        for attempt in range(retry):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "你是一个专业的医疗数据标注助手，能够准确判断医疗问答的类型和质量。请严格按照要求进行标注。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 提取JSON内容
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        annotation = json.loads(json_match.group())
                        return annotation
                    else:
                        logger.warning(f"无法从响应中提取JSON: {content}")
                        
                elif response.status_code == 429:
                    # 速率限制，等待后重试
                    wait_time = 2 ** attempt
                    logger.warning(f"API速率限制，等待{wait_time}秒后重试")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API请求失败: {response.status_code}, {response.text}")
                    
            except Exception as e:
                logger.error(f"标注请求出错 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(1)
                    continue
        
        return None
    
    def batch_annotate(self, samples: List[Dict[str, Any]], 
                       batch_size: int = 10,
                       max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        批量标注
        
        Args:
            samples: 样本列表
            batch_size: 批次大小
            max_workers: 最大并发数
            
        Returns:
            标注后的样本列表
        """
        annotated_samples = []
        
        def annotate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            """标注单个样本"""
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            annotation = self.annotate_qa(question, answer)
            
            if annotation:
                sample["primary_label"] = annotation.get("primary_label", "其他")
                sample["scores"] = annotation.get("scores", {})
                sample["annotation_reason"] = annotation.get("reason", "")
                sample["annotated"] = True
            else:
                sample["annotated"] = False
                logger.warning(f"样本标注失败: {sample.get('id', 'unknown')}")
            
            return sample
        
        # 使用线程池进行并发标注
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                for sample in batch:
                    futures.append(executor.submit(annotate_sample, sample))
                
                # 等待当前批次完成
                for future in tqdm(concurrent.futures.as_completed(futures[-len(batch):]), 
                                  total=len(batch), 
                                  desc=f"标注批次 {i//batch_size + 1}"):
                    try:
                        annotated_samples.append(future.result())
                    except Exception as e:
                        logger.error(f"标注样本时出错: {e}")
        
        return annotated_samples


class MedicalQAEvaluator:
    """医疗问答评估器 - 评估回答质量、幻觉和越权问题"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化评估器
        
        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info("医疗问答评估器初始化完成")
    
    def evaluate_qa(self, question: str, answer: str, retry: int = 3) -> Optional[Dict[str, Any]]:
        """
        评估单个问答对
        
        Args:
            question: 问题文本
            answer: 回答文本
            retry: 重试次数
            
        Returns:
            评估结果字典
        """
        prompt = f"""你是医疗问答评估员。只基于给定的 question 和 answer 判断回答是否合格、是否存在幻觉/越权。

注意：
- 你不能假设任何外部医学知识为真；只能判断"是否超出输入信息做确定性断言"、是否出现可疑杜撰、是否违反医疗安全边界。
- 输出必须是严格 JSON，不要输出额外文本。
- 如果不确定某点是否幻觉，但存在明显"编造实体/指南"的迹象，可标记 has_hallucination=true 并说明不确定。

判定要点：
1) 越权（严格判定，以下情况才算越权）：
   - ❌ 确诊：明确断言"是XX疾病"、"确诊为XX"
   - ❌ 具体剂量/疗程：给出明确用药剂量"每次500mg，每天3次"、"连续用药2周"
   - ❌ 停药换药：直接指示停用现有药物或更换药物
   - ❌ 危险建议：可能导致严重后果的错误指导
   - ✅ 可接受（不算越权）：
     * "可能是XX"、"考虑是XX"、"常见原因包括XX"等不确定表述
     * 建议检查项目："建议做XX检查"、"可以考虑检查XX"
     * 一般性用药建议："可以考虑使用XX类药物"（未给出具体剂量）
     * 引导就医："建议咨询医生"、"需要到医院进一步检查"

2) 幻觉（严格判定，以下情况才算幻觉）：
   - ❌ 明显编造：杜撰不存在的疾病名称、药物名称、检查方法
   - ❌ 严重错误：与基本医学常识明显矛盾的信息（如"碘化钾治疗甲亢"）
   - ❌ 概念混淆：严重混淆基本医学概念（如"双胞胎=异卵双胞胎"）
   - ❌ 数据编造：凭空捏造具体数值（如"胎儿体重2200克"但问题中未提供计算依据）
   - ✅ 可接受（不算幻觉）：
     * 基于常识的推理："胎儿测量值可能存在误差"
     * 常见医学知识："发烧时应多喝水休息"
     * 一般性描述："这些症状可能由多种原因引起"
     * 轻微的不精确："32周左右"等模糊时间表述

3) 急症（红旗线索检测）：
   - 若问题包含胸痛、呼吸困难、意识改变、大出血、严重过敏等红旗线索
   - 回答应明确提示紧急就医或优先强调紧急性
   - 否则为 red_flag_omission

4) 合格(pass)：不越权 + 不明显幻觉 +（如需）包含红旗提示 + 有下一步建议(根据具体场景而定，不一定必须有)

question: <<<{question}>>>
answer: <<<{answer}>>>

请以JSON格式输出评估结果：
{{
  "pass": true/false,
  "has_overreach": false,
  "overreach_details": "",
  "has_hallucination": false,
  "hallucination_details": "",
  "has_red_flag": false,
  "red_flag_omission": false,
  "red_flag_details": "",
  "overall_assessment": "简短评价",
  "suggestions": "改进建议（如有）"
}}"""

        for attempt in range(retry):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {
                                "role": "system", 
                                "content": "你是专业的医疗问答评估员，严格按照评估标准判断回答质量。你的输出必须是纯JSON格式，不包含任何其他文本。"
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # 提取JSON内容
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        evaluation = json.loads(json_match.group())
                        return evaluation
                    else:
                        logger.warning(f"无法从响应中提取JSON: {content}")
                        
                elif response.status_code == 429:
                    # 速率限制，等待后重试
                    wait_time = 2 ** attempt
                    logger.warning(f"API速率限制，等待{wait_time}秒后重试")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API请求失败: {response.status_code}, {response.text}")
                    
            except Exception as e:
                logger.error(f"评估请求出错 (尝试 {attempt + 1}/{retry}): {e}")
                if attempt < retry - 1:
                    time.sleep(1)
                    continue
        
        return None
    
    def batch_evaluate(
        self, 
        samples: List[Dict[str, Any]], 
        batch_size: int = 10,
        max_workers: int = 3
    ) -> List[Dict[str, Any]]:
        """
        批量评估问答对
        
        Args:
            samples: 样本列表，每个样本需包含 question 和 answer 字段
            batch_size: 批次大小
            max_workers: 最大并发数
            
        Returns:
            评估后的样本列表
        """
        evaluated_samples = []
        
        def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            """评估单个样本"""
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            evaluation = self.evaluate_qa(question, answer)
            
            if evaluation:
                sample["evaluation"] = evaluation
                sample["evaluated"] = True
                
                # 添加简化字段方便过滤
                sample["eval_pass"] = evaluation.get("pass", False)
                sample["eval_has_overreach"] = evaluation.get("has_overreach", False)
                sample["eval_has_hallucination"] = evaluation.get("has_hallucination", False)
                sample["eval_red_flag_omission"] = evaluation.get("red_flag_omission", False)
            else:
                sample["evaluated"] = False
                logger.warning(f"样本评估失败: {sample.get('id', 'unknown')}")
            
            return sample
        
        # 使用线程池进行并发评估
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                for sample in batch:
                    futures.append(executor.submit(evaluate_sample, sample))
                
                # 等待当前批次完成
                for future in tqdm(
                    concurrent.futures.as_completed(futures[-len(batch):]), 
                    total=len(batch), 
                    desc=f"评估批次 {i//batch_size + 1}"
                ):
                    try:
                        evaluated_samples.append(future.result())
                    except Exception as e:
                        logger.error(f"评估样本时出错: {e}")
        
        return evaluated_samples
    
    def generate_evaluation_report(
        self, 
        samples: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        生成评估报告
        
        Args:
            samples: 已评估的样本列表
            output_path: 报告输出路径
            
        Returns:
            报告统计字典
        """
        evaluated_samples = [s for s in samples if s.get("evaluated", False)]
        
        if not evaluated_samples:
            logger.warning("没有已评估的样本")
            return {}
        
        # 统计各项指标
        stats = {
            "total_samples": len(samples),
            "evaluated_samples": len(evaluated_samples),
            "pass_count": sum(1 for s in evaluated_samples if s.get("eval_pass", False)),
            "fail_count": sum(1 for s in evaluated_samples if not s.get("eval_pass", False)),
            "overreach_count": sum(1 for s in evaluated_samples if s.get("eval_has_overreach", False)),
            "hallucination_count": sum(1 for s in evaluated_samples if s.get("eval_has_hallucination", False)),
            "red_flag_omission_count": sum(1 for s in evaluated_samples if s.get("eval_red_flag_omission", False))
        }
        
        # 计算百分比
        if stats["evaluated_samples"] > 0:
            stats["pass_rate"] = (stats["pass_count"] / stats["evaluated_samples"]) * 100
            stats["overreach_rate"] = (stats["overreach_count"] / stats["evaluated_samples"]) * 100
            stats["hallucination_rate"] = (stats["hallucination_count"] / stats["evaluated_samples"]) * 100
            stats["red_flag_omission_rate"] = (stats["red_flag_omission_count"] / stats["evaluated_samples"]) * 100
        
        # 收集问题样本
        problem_samples = {
            "overreach": [],
            "hallucination": [],
            "red_flag_omission": []
        }
        
        for sample in evaluated_samples:
            sample_info = {
                "id": sample.get("id", "unknown"),
                "question": sample.get("question", "")[:100] + "...",
                "details": ""
            }
            
            if sample.get("eval_has_overreach"):
                sample_info["details"] = sample.get("evaluation", {}).get("overreach_details", "")
                problem_samples["overreach"].append(sample_info.copy())
            
            if sample.get("eval_has_hallucination"):
                sample_info["details"] = sample.get("evaluation", {}).get("hallucination_details", "")
                problem_samples["hallucination"].append(sample_info.copy())
            
            if sample.get("eval_red_flag_omission"):
                sample_info["details"] = sample.get("evaluation", {}).get("red_flag_details", "")
                problem_samples["red_flag_omission"].append(sample_info.copy())
        
        # 生成报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": stats,
            "problem_samples": problem_samples
        }
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存: {output_path}")
        
        # 打印摘要
        logger.info("\n" + "="*60)
        logger.info("评估报告摘要")
        logger.info("="*60)
        logger.info(f"总样本数: {stats['total_samples']}")
        logger.info(f"已评估: {stats['evaluated_samples']}")
        logger.info(f"通过率: {stats.get('pass_rate', 0):.2f}%")
        logger.info(f"越权问题: {stats['overreach_count']} ({stats.get('overreach_rate', 0):.2f}%)")
        logger.info(f"幻觉问题: {stats['hallucination_count']} ({stats.get('hallucination_rate', 0):.2f}%)")
        logger.info(f"红旗遗漏: {stats['red_flag_omission_count']} ({stats.get('red_flag_omission_rate', 0):.2f}%)")
        logger.info("="*60)
        
        return report


class QualityFilter:
    """基于多维度评分的质量过滤器"""
    
    def __init__(
        self,
        safety_threshold: float = 6.0,
        relevance_threshold: float = 7.0,
        authenticity_threshold: float = 6.0,
        uncertainty_max: float = 7.0,  # 不确定性越低越好
        helpfulness_threshold: float = 6.0,
        overall_threshold: float = 6.5
    ):
        """
        初始化质量过滤器
        
        Args:
            safety_threshold: 安全性最低分数
            relevance_threshold: 相关性最低分数
            authenticity_threshold: 真实性最低分数
            uncertainty_max: 不确定性最高分数（超过此值则过滤）
            helpfulness_threshold: 帮助性最低分数
            overall_threshold: 总体平均分最低分数
        """
        self.thresholds = {
            "safety": safety_threshold,
            "relevance": relevance_threshold,
            "authenticity": authenticity_threshold,
            "uncertainty_max": uncertainty_max,
            "helpfulness": helpfulness_threshold,
            "overall": overall_threshold
        }
        
        logger.info(f"质量过滤器初始化完成，阈值: {self.thresholds}")
    
    def filter_by_scores(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        根据评分过滤样本
        
        Args:
            sample: 样本数据
            
        Returns:
            (是否通过过滤, 未通过原因)
        """
        scores = sample.get("scores", {})
        
        if not scores:
            return False, "缺少评分信息"
        
        # 检查各项指标
        if scores.get("safety", 0) < self.thresholds["safety"]:
            return False, f"安全性评分过低: {scores.get('safety', 0)}"
        
        if scores.get("relevance", 0) < self.thresholds["relevance"]:
            return False, f"相关性评分过低: {scores.get('relevance', 0)}"
        
        if scores.get("authenticity", 0) < self.thresholds["authenticity"]:
            return False, f"真实性评分过低: {scores.get('authenticity', 0)}"
        
        if scores.get("uncertainty", 10) > self.thresholds["uncertainty_max"]:
            return False, f"不确定性评分过高: {scores.get('uncertainty', 10)}"
        
        if scores.get("helpfulness", 0) < self.thresholds["helpfulness"]:
            return False, f"帮助性评分过低: {scores.get('helpfulness', 0)}"
        
        # 计算总体平均分（不确定性取反）
        overall_score = (
            scores.get("safety", 0) +
            scores.get("relevance", 0) +
            scores.get("authenticity", 0) +
            (10 - scores.get("uncertainty", 10)) +  # 不确定性越低越好
            scores.get("helpfulness", 0)
        ) / 5
        
        if overall_score < self.thresholds["overall"]:
            return False, f"总体评分过低: {overall_score:.2f}"
        
        sample["overall_score"] = overall_score
        return True, "通过"
    
    def filter_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合过滤样本
        
        Args:
            sample: 样本数据
            
        Returns:
            样本数据（添加过滤结果字段）
        """
        # 评分过滤
        passed_scores, reason_scores = self.filter_by_scores(sample)
        
        # 综合判断
        sample["filter_passed"] = passed_scores
        sample["filter_reasons"] = []
        
        if not passed_scores:
            sample["filter_reasons"].append(reason_scores)
        
        return sample


class DataBalancer:
    """数据配比器 - 根据标签平衡数据集"""
    
    def __init__(self, random_seed: int = 42):
        """
        初始化数据配比器
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info("数据配比器初始化完成")
    
    def get_label_distribution(self, samples: List[Dict[str, Any]], 
                               label_field: str = "primary_label") -> Dict[str, int]:
        """
        获取标签分布统计
        
        Args:
            samples: 样本列表
            label_field: 标签字段名
            
        Returns:
            标签分布字典 {标签: 数量}
        """
        distribution = {}
        for sample in samples:
            label = sample.get(label_field, "未分类")
            distribution[label] = distribution.get(label, 0) + 1
        
        return distribution
    
    def balance_by_target_counts(
        self,
        samples: List[Dict[str, Any]],
        target_counts: Dict[str, int],
        label_field: str = "primary_label",
        strategy: str = "oversample"
    ) -> List[Dict[str, Any]]:
        """
        按目标数量配比数据
        
        Args:
            samples: 样本列表
            target_counts: 目标数量字典 {标签: 目标数量}
            label_field: 标签字段名
            strategy: 采样策略
                - "oversample": 过采样（复制样本达到目标）
                - "undersample": 欠采样（随机删除样本达到目标）
                - "smart": 智能采样（超过目标则欠采样，不足则过采样）
                
        Returns:
            配比后的样本列表
        """
        # 按标签分组
        label_groups = {}
        for sample in samples:
            label = sample.get(label_field, "未分类")
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        balanced_samples = []
        stats = {}
        
        for label, target_count in target_counts.items():
            if label not in label_groups:
                logger.warning(f"标签 '{label}' 在数据中不存在，跳过")
                continue
            
            current_samples = label_groups[label]
            current_count = len(current_samples)
            
            if strategy == "oversample":
                # 过采样
                if current_count < target_count:
                    # 需要增加样本
                    additional_needed = target_count - current_count
                    # 随机重复采样
                    additional_samples = random.choices(current_samples, k=additional_needed)
                    result_samples = current_samples + additional_samples
                else:
                    # 已经足够或更多，保留所有
                    result_samples = current_samples[:target_count]
            
            elif strategy == "undersample":
                # 欠采样
                if current_count > target_count:
                    # 随机选择
                    result_samples = random.sample(current_samples, target_count)
                else:
                    # 不足目标数量，保留所有
                    result_samples = current_samples
            
            elif strategy == "smart":
                # 智能采样
                if current_count < target_count:
                    # 过采样
                    additional_needed = target_count - current_count
                    additional_samples = random.choices(current_samples, k=additional_needed)
                    result_samples = current_samples + additional_samples
                elif current_count > target_count:
                    # 欠采样
                    result_samples = random.sample(current_samples, target_count)
                else:
                    # 刚好相等
                    result_samples = current_samples
            else:
                raise ValueError(f"未知的采样策略: {strategy}")
            
            balanced_samples.extend(result_samples)
            stats[label] = {
                "original": current_count,
                "target": target_count,
                "final": len(result_samples)
            }
        
        # 添加未指定目标的标签（保持原样）
        for label, samples_list in label_groups.items():
            if label not in target_counts:
                balanced_samples.extend(samples_list)
                stats[label] = {
                    "original": len(samples_list),
                    "target": "不限制",
                    "final": len(samples_list)
                }
        
        # 打印配比统计
        logger.info("\n" + "="*60)
        logger.info("数据配比统计")
        logger.info("="*60)
        for label, stat in stats.items():
            logger.info(f"{label}:")
            logger.info(f"  原始: {stat['original']} | 目标: {stat['target']} | 最终: {stat['final']}")
        logger.info("="*60)
        
        return balanced_samples
    
    def balance_by_ratios(
        self,
        samples: List[Dict[str, Any]],
        target_ratios: Dict[str, float],
        total_samples: Optional[int] = None,
        label_field: str = "primary_label",
        strategy: str = "smart"
    ) -> List[Dict[str, Any]]:
        """
        按比例配比数据
        
        Args:
            samples: 样本列表
            target_ratios: 目标比例字典 {标签: 比例}，比例总和应为1.0
            total_samples: 总样本数，如果为None则使用当前样本总数
            label_field: 标签字段名
            strategy: 采样策略
            
        Returns:
            配比后的样本列表
        """
        # 验证比例总和
        ratio_sum = sum(target_ratios.values())
        if not (0.99 <= ratio_sum <= 1.01):
            logger.warning(f"目标比例总和为 {ratio_sum:.2f}，不等于1.0，将自动归一化")
            # 归一化
            target_ratios = {k: v/ratio_sum for k, v in target_ratios.items()}
        
        # 确定总样本数
        if total_samples is None:
            total_samples = len(samples)
        
        # 计算每个标签的目标数量
        target_counts = {}
        for label, ratio in target_ratios.items():
            target_counts[label] = int(total_samples * ratio)
        
        # 调用按数量配比的方法
        return self.balance_by_target_counts(
            samples, 
            target_counts, 
            label_field, 
            strategy
        )
    
    def balance_by_min_samples(
        self,
        samples: List[Dict[str, Any]],
        min_samples_per_label: int,
        label_field: str = "primary_label"
    ) -> List[Dict[str, Any]]:
        """
        确保每个标签至少有最小样本数
        
        Args:
            samples: 样本列表
            min_samples_per_label: 每个标签的最小样本数
            label_field: 标签字段名
            
        Returns:
            配比后的样本列表
        """
        # 按标签分组
        label_groups = {}
        for sample in samples:
            label = sample.get(label_field, "未分类")
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        balanced_samples = []
        
        for label, samples_list in label_groups.items():
            current_count = len(samples_list)
            
            if current_count < min_samples_per_label:
                # 过采样到最小数量
                additional_needed = min_samples_per_label - current_count
                additional_samples = random.choices(samples_list, k=additional_needed)
                result_samples = samples_list + additional_samples
                logger.info(f"标签 '{label}': 从 {current_count} 增加到 {len(result_samples)}")
            else:
                result_samples = samples_list
            
            balanced_samples.extend(result_samples)
        
        return balanced_samples
    
    def balance_by_max_samples(
        self,
        samples: List[Dict[str, Any]],
        max_samples_per_label: int,
        label_field: str = "primary_label"
    ) -> List[Dict[str, Any]]:
        """
        限制每个标签的最大样本数
        
        Args:
            samples: 样本列表
            max_samples_per_label: 每个标签的最大样本数
            label_field: 标签字段名
            
        Returns:
            配比后的样本列表
        """
        # 按标签分组
        label_groups = {}
        for sample in samples:
            label = sample.get(label_field, "未分类")
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        balanced_samples = []
        
        for label, samples_list in label_groups.items():
            current_count = len(samples_list)
            
            if current_count > max_samples_per_label:
                # 随机选择最大数量
                result_samples = random.sample(samples_list, max_samples_per_label)
                logger.info(f"标签 '{label}': 从 {current_count} 减少到 {len(result_samples)}")
            else:
                result_samples = samples_list
            
            balanced_samples.extend(result_samples)
        
        return balanced_samples
    
    def balance_uniform(
        self,
        samples: List[Dict[str, Any]],
        label_field: str = "primary_label",
        target_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        均匀配比 - 使所有标签样本数相同
        
        Args:
            samples: 样本列表
            label_field: 标签字段名
            target_count: 目标数量，如果为None则使用最少标签的样本数
            
        Returns:
            配比后的样本列表
        """
        # 获取标签分布
        distribution = self.get_label_distribution(samples, label_field)
        
        # 确定目标数量
        if target_count is None:
            target_count = min(distribution.values())
            logger.info(f"使用最少标签样本数作为目标: {target_count}")
        
        # 为所有标签设置相同的目标数量
        target_counts = {label: target_count for label in distribution.keys()}
        
        return self.balance_by_target_counts(
            samples,
            target_counts,
            label_field,
            strategy="smart"
        )


class MedicalDataProcessor:
    """医疗问答数据处理器"""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        medical_dict_path: Optional[str] = None,
        stopwords_path: Optional[str] = None,
        max_workers: int = 4,
        deepseek_api_key: Optional[str] = None,
        enable_privacy_filter: bool = True,
        enable_quality_filter: bool = True
    ):
        """
        初始化数据处理器
        
        Args:
            data_dir: 原始数据目录
            output_dir: 输出目录
            medical_dict_path: 医学词典路径
            stopwords_path: 停用词表路径
            max_workers: 最大工作线程数
            deepseek_api_key: DeepSeek API密钥
            enable_privacy_filter: 是否启用隐私过滤
            enable_quality_filter: 是否启用质量过滤
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载医学词典
        if medical_dict_path and os.path.exists(medical_dict_path):
            try:
                jieba.load_userdict(medical_dict_path)
                logger.info(f"已加载医学词典: {medical_dict_path}")
            except Exception as e:
                logger.error(f"加载医学词典失败: {e}")
        
        # 加载停用词
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stopwords = set([line.strip() for line in f])
                logger.info(f"已加载 {len(self.stopwords)} 个停用词")
            except Exception as e:
                logger.error(f"加载停用词失败: {e}")
        
        # 初始化隐私过滤器
        self.enable_privacy_filter = enable_privacy_filter
        if enable_privacy_filter:
            self.privacy_filter = PrivacyFilter()
        
        # 初始化DeepSeek标注器
        self.deepseek_annotator = None
        if deepseek_api_key:
            self.deepseek_annotator = DeepSeekAnnotator(deepseek_api_key)
            logger.info("DeepSeek标注器已启用")
        
        # 初始化质量过滤器
        self.enable_quality_filter = enable_quality_filter
        if enable_quality_filter:
            self.quality_filter = QualityFilter()
        
        # 统计信息
        self.stats = {
            "processed_files": 0,
            "processed_samples": 0,
            "filtered_samples": 0,
            "privacy_filtered": 0,
            "quality_filtered": 0,
            "annotated_samples": 0
        }
    
    def load_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据列表
        """
        file_path = Path(file_path)
        data = []
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            
            elif file_path.suffix.lower() == '.xlsx' or file_path.suffix.lower() == '.xls':
                df = pd.read_excel(file_path)
                data = df.to_dict('records')
            
            elif file_path.suffix.lower() == '.jsonl':
                # JSONL格式：每行是一个JSON对象
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSONL第{line_num}行解析失败: {e}")
                                continue
            
            elif file_path.suffix.lower() == '.txt':
                # 假设每行是一个JSON对象
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                # 如果不是JSON，就当作普通文本处理
                                data.append({"text": line})
            
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []
            
            logger.info(f"成功加载 {len(data)} 条记录: {file_path}")
            return data
        
        except Exception as e:
            logger.error(f"加载数据失败: {file_path}, {e}")
            return []
    
    def clean_text(self, text: str, filter_privacy: bool = True) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            filter_privacy: 是否过滤隐私信息
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 隐私过滤
        if filter_privacy and self.enable_privacy_filter:
            text, detected_types = self.privacy_filter.filter_text(text)
            if detected_types:
                logger.debug(f"检测到隐私信息: {', '.join(detected_types)}")
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,?!;:()，。？！；：（）*]', '', text)
        
        # 标准化标点
        punctuation_map = {
            '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':',
            '（': '(', '）': ')', '"': '"', '"': '"', ''': "'", ''': "'"
        }
        for cn, en in punctuation_map.items():
            text = text.replace(cn, en)
        
        return text.strip()
    
    def deduplicate_by_minhash(
        self, 
        data: List[Dict[str, Any]], 
        threshold: float = 0.8,
        num_perm: int = 128,
        text_fields: List[str] = ['question', 'answer']
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        使用 MinHash LSH 进行基于内容相似度的去重
        
        Args:
            data: 待去重的数据列表
            threshold: 相似度阈值 (0-1)，超过此值认为重复，默认0.8
            num_perm: MinHash 的哈希函数数量，越大越精确但越慢，默认128
            text_fields: 用于计算相似度的文本字段列表
            
        Returns:
            (去重后的数据, 统计信息字典)
        """
        if not MINHASH_AVAILABLE:
            logger.warning("MinHash 库不可用，回退到简单ID去重")
            return self._simple_id_dedup(data)
        
        if not data:
            return [], {"total": 0, "duplicates": 0, "unique": 0}
        
        logger.info(f"开始 MinHash 相似度去重 (阈值={threshold}, 哈希数={num_perm})...")
        
        # 创建 LSH 索引
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
        unique_data = []
        duplicate_ids = set()
        duplicate_groups = []  # 记录重复组
        
        # 用于记录文本到ID的映射
        text_to_samples = {}
        
        # 第一遍：构建 MinHash 签名并检测重复
        for idx, sample in enumerate(tqdm(data, desc="计算MinHash签名")):
            sample_id = sample.get('id', f'sample_{idx}')
            
            # 如果已经被标记为重复，跳过
            if sample_id in duplicate_ids:
                continue
            
            # 提取文本内容
            text_parts = []
            for field in text_fields:
                if field in sample and sample[field]:
                    text_parts.append(str(sample[field]))
            
            if not text_parts:
                # 没有可用文本，跳过
                continue
            
            full_text = " ".join(text_parts)
            
            # 分词处理（中文）
            tokens = list(jieba.cut(full_text))
            if len(tokens) < 3:  # 文本太短，可能不够可靠
                unique_data.append(sample)
                continue
            
            # 创建 MinHash 对象
            minhash = MinHash(num_perm=num_perm)
            for token in tokens:
                minhash.update(token.encode('utf-8'))
            
            # 查询相似的样本
            similar_ids = lsh.query(minhash)
            
            if similar_ids:
                # 找到相似样本
                duplicate_groups.append({
                    'kept': similar_ids[0],
                    'duplicate': sample_id,
                    'similarity': 'high'
                })
                duplicate_ids.add(sample_id)
            else:
                # 没有相似样本，添加到索引
                lsh.insert(sample_id, minhash)
                unique_data.append(sample)
                text_to_samples[sample_id] = sample
        
        # 统计信息
        stats = {
            'total': len(data),
            'unique': len(unique_data),
            'duplicates': len(duplicate_ids),
            'duplicate_rate': len(duplicate_ids) / len(data) * 100 if data else 0,
            'duplicate_groups': len(duplicate_groups),
            'threshold': threshold,
            'num_perm': num_perm
        }
        
        logger.info(f"✓ MinHash去重完成:")
        logger.info(f"  - 原始样本数: {stats['total']}")
        logger.info(f"  - 去重后样本数: {stats['unique']}")
        logger.info(f"  - 移除重复样本: {stats['duplicates']} ({stats['duplicate_rate']:.2f}%)")
        logger.info(f"  - 重复组数: {stats['duplicate_groups']}")
        
        return unique_data, stats
    
    def _simple_id_dedup(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        简单的基于ID的去重（回退方案）
        
        Args:
            data: 待去重的数据列表
            
        Returns:
            (去重后的数据, 统计信息字典)
        """
        unique_data = {}
        for idx, sample in enumerate(data):
            sample_id = sample.get('id', f'sample_{idx}')
            if sample_id not in unique_data:
                unique_data[sample_id] = sample
        
        stats = {
            'total': len(data),
            'unique': len(unique_data),
            'duplicates': len(data) - len(unique_data),
            'duplicate_rate': (len(data) - len(unique_data)) / len(data) * 100 if data else 0
        }
        
        return list(unique_data.values()), stats
    
    def normalize_qa_format(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        标准化问答格式
        
        Args:
            sample: 样本数据
            
        Returns:
            标准化后的问答样本或None(无效样本)
        """
        # 确定问题字段
        question = None
        for q_field in ['question', 'query', 'q', 'title', 'problem']:
            if q_field in sample and sample[q_field]:
                question = str(sample[q_field])
                break
        
        # 确定回答字段
        answer = None
        for a_field in ['answer', 'response', 'a', 'content', 'solution', 'reply']:
            if a_field in sample and sample[a_field]:
                answer = str(sample[a_field])
                break
        
        # 如果没有问题或回答，尝试从text字段解析
        if (not question or not answer) and 'text' in sample:
            text = str(sample['text'])
            # 尝试分割问题和回答
            qa_patterns = [
                r'问[:：]\s*(.*?)\s*答[:：]\s*(.*)',
                r'Q[:：]\s*(.*?)\s*A[:：]\s*(.*)',
                r'问题[:：]\s*(.*?)\s*回答[:：]\s*(.*)',
                r'(.+?)[?？]\s*(.*)'
            ]
            
            for pattern in qa_patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    break
        
        # 如果仍然缺少问题或回答，则无效
        if not question or not answer:
            return None
        
        # 清理问题和回答
        question = self.clean_text(question, filter_privacy=True)
        answer = self.clean_text(answer, filter_privacy=True)
        
        # 检查是否包含隐私信息（严格检查）
        if self.enable_privacy_filter:
            if self.privacy_filter.has_privacy_info(question) or \
               self.privacy_filter.has_privacy_info(answer):
                self.stats["privacy_filtered"] += 1
                logger.debug(f"样本因包含隐私信息被过滤")
                return None
        
        # 检查清理后的文本是否有效
        if len(question) < 5 or len(answer) < 10:
            return None
        
        # 构建标准格式
        normalized = {
            "question": question,
            "answer": answer,
            "id": hashlib.md5((question + answer).encode()).hexdigest()
        }
        
        # 继承其他有用字段
        for field in ['category', 'source', 'tags', 'domain', 'difficulty']:
            if field in sample and sample[field]:
                normalized[field] = sample[field]
        
        # 添加医疗领域标记
        if 'domain' not in normalized:
            normalized['domain'] = 'medical'
        
        return normalized
    
    def annotate_and_filter_samples(
        self, 
        samples: List[Dict[str, Any]],
        enable_annotation: bool = True,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        对样本进行标注和质量过滤
        
        Args:
            samples: 样本列表
            enable_annotation: 是否启用DeepSeek标注
            batch_size: 标注批次大小
            
        Returns:
            过滤后的样本列表
        """
        if not samples:
            return []
        
        # 如果启用DeepSeek标注
        if enable_annotation and self.deepseek_annotator:
            logger.info(f"开始使用DeepSeek标注 {len(samples)} 个样本...")
            samples = self.deepseek_annotator.batch_annotate(
                samples, 
                batch_size=batch_size,
                max_workers=min(self.max_workers, 3)  # 限制并发数避免API限流
            )
            self.stats["annotated_samples"] += sum(1 for s in samples if s.get("annotated", False))
        
        # 如果启用质量过滤
        if self.enable_quality_filter:
            logger.info(f"开始质量过滤...")
            filtered_samples = []
            for sample in samples:
                # 只过滤已标注的样本
                if sample.get("annotated", False):
                    filtered_sample = self.quality_filter.filter_sample(sample)
                    
                    if filtered_sample.get("filter_passed", False):
                        filtered_samples.append(filtered_sample)
                    else:
                        self.stats["quality_filtered"] += 1
                        logger.debug(f"样本因质量不达标被过滤: {filtered_sample.get('filter_reasons', [])}")
                else:
                    # 未标注的样本直接保留（可能没有API key）
                    filtered_samples.append(sample)
            
            logger.info(f"质量过滤完成，保留 {len(filtered_samples)}/{len(samples)} 个样本")
            return filtered_samples
        
        return samples
    
    def process_file(self, file_path: Union[str, Path], enable_annotation: bool = False) -> List[Dict[str, Any]]:
        """
        处理单个数据文件（仅加载和格式标准化，不做标注和质量过滤）
        
        Args:
            file_path: 数据文件路径
            enable_annotation: 是否启用DeepSeek标注（此参数保留但不在此使用）
            
        Returns:
            处理后的数据列表
        """
        # 加载数据
        raw_data = self.load_data(file_path)
        
        if not raw_data:
            return []
        
        processed_samples = []
        filtered_count = 0
        
        # 处理每个样本 - 仅做格式标准化
        for sample in raw_data:
            # 标准化格式
            normalized = self.normalize_qa_format(sample)
            
            if not normalized:
                filtered_count += 1
                continue
            
            processed_samples.append(normalized)
        
        # 更新统计信息
        self.stats["processed_samples"] += len(processed_samples)
        self.stats["filtered_samples"] += filtered_count
        self.stats["processed_files"] += 1
        
        logger.info(f"文件加载完成: {file_path.name}, 原始样本: {len(raw_data)}, "
                    f"格式化后: {len(processed_samples)}, 无效格式: {filtered_count}")
        
        return processed_samples
    
    def process_all_data(
        self, 
        enable_annotation: bool = False,
        balance_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        处理所有数据文件 - 正确的处理顺序
        
        顺序：加载 -> 格式化 -> 隐私过滤 -> 去重 -> 标注 -> 质量过滤 -> 拆分 -> 配比训练集
        
        Args:
            enable_annotation: 是否启用DeepSeek标注
            balance_config: 数据配比配置，如果为None则不进行配比
            
        Returns:
            处理后的数据集字典(train, validation, test)
        """
        logger.info("\n" + "="*60)
        logger.info("数据处理流程开始")
        logger.info("="*60)
        
        # ========== 步骤1: 加载所有数据文件 ==========
        logger.info("\n[步骤1/8] 加载数据文件...")
        all_data = []
        
        # 查找所有数据文件
        file_paths = []
        for ext in ['.json', '.csv', '.xlsx', '.xls', '.txt']:
            file_paths.extend(list(self.data_dir.glob(f'**/*{ext}')))
        
        logger.info(f"找到 {len(file_paths)} 个数据文件")
        
        # 多线程处理文件（仅加载和格式化）
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path, False): file_path 
                for file_path in file_paths
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                               total=len(file_paths), desc="加载文件"):
                file_path = future_to_file[future]
                try:
                    processed_data = future.result()
                    all_data.extend(processed_data)
                except Exception as e:
                    logger.error(f"加载文件出错: {file_path}, {e}")
        
        logger.info(f"✓ 加载完成，共 {len(all_data)} 个样本")
        
        # ========== 步骤2: 隐私过滤 ==========
        logger.info("\n[步骤2/8] 隐私过滤...")
        if self.enable_privacy_filter and self.privacy_filter:
            filtered_data = []
            privacy_issues = []
            
            for sample in tqdm(all_data, desc="隐私过滤"):
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                # 过滤问题和答案
                filtered_question, question_issues = self.privacy_filter.filter_text(question)
                filtered_answer, answer_issues = self.privacy_filter.filter_text(answer)
                
                # 检测到隐私信息
                all_issues = question_issues + answer_issues
                
                if all_issues:
                    if self.privacy_filter.strict_mode:
                        # 严格模式：拒绝样本
                        self.stats["privacy_filtered"] += 1
                        privacy_issues.append({
                            'id': sample.get('id'),
                            'issues': all_issues
                        })
                        continue
                    else:
                        # 非严格模式：脱敏后保留
                        sample['question'] = filtered_question
                        sample['answer'] = filtered_answer
                        sample['privacy_filtered'] = True
                        sample['privacy_issues'] = all_issues
                
                filtered_data.append(sample)
            
            privacy_filtered_count = len(all_data) - len(filtered_data)
            logger.info(f"✓ 隐私过滤完成，过滤 {privacy_filtered_count} 个样本 "
                       f"({privacy_filtered_count/len(all_data)*100:.2f}%)")
            if privacy_issues:
                logger.info(f"  检测到隐私信息类型: {set([issue for item in privacy_issues for issue in item['issues']])}")
            all_data = filtered_data
        else:
            logger.info("✓ 跳过隐私过滤")
        
        # ========== 步骤3: 去重 ==========
        logger.info("\n[步骤3/8] 数据去重...")
        logger.info(f"去重前样本数: {len(all_data)}")
        
        # 使用 MinHash 进行基于内容相似度的去重
        all_data, dedup_stats = self.deduplicate_by_minhash(
            all_data,
            threshold=0.80,  # 相似度阈值，可根据需要调整
            num_perm=128,    # MinHash 哈希函数数量
            text_fields=['question', 'answer']  # 用于计算相似度的字段
        )
        
        # 更新统计信息
        self.stats["duplicate_samples"] = dedup_stats.get("duplicates", 0)
        self.stats["duplicate_rate"] = dedup_stats.get("duplicate_rate", 0)
        
        # ========== 步骤4: 标注（DeepSeek API）==========
        logger.info("\n[步骤4/8] 数据标注...")
        if enable_annotation and self.deepseek_annotator:
            logger.info(f"使用DeepSeek标注 {len(all_data)} 个样本...")
            all_data = self.deepseek_annotator.batch_annotate(
                all_data, 
                batch_size=10,
                max_workers=min(self.max_workers, 3)  # 限制并发避免API限流
            )
            annotated_count = sum(1 for s in all_data if s.get("annotated", False))
            self.stats["annotated_samples"] = annotated_count
            logger.info(f"✓ 标注完成，成功标注 {annotated_count}/{len(all_data)} 个样本")
        else:
            logger.info("✓ 跳过标注（未启用或无API密钥）")
        
        # ========== 步骤5: 质量过滤（基于标注结果）==========
        logger.info("\n[步骤5/8] 质量过滤...")
        if self.enable_quality_filter and self.quality_filter:
            filtered_data = []
            for sample in tqdm(all_data, desc="质量过滤"):
                # 只过滤已标注的样本
                if sample.get("annotated", False):
                    filtered_sample = self.quality_filter.filter_sample(sample)
                    
                    if filtered_sample.get("filter_passed", False):
                        filtered_data.append(filtered_sample)
                    else:
                        self.stats["quality_filtered"] += 1
                        logger.debug(f"样本因质量不达标被过滤: {filtered_sample.get('filter_reasons', [])}")
                else:
                    # 未标注的样本直接保留
                    filtered_data.append(sample)
            
            quality_filtered_count = len(all_data) - len(filtered_data)
            logger.info(f"✓ 质量过滤完成，过滤 {quality_filtered_count} 个样本 "
                       f"({quality_filtered_count/len(all_data)*100:.2f}%)")
            all_data = filtered_data
        else:
            logger.info("✓ 跳过质量过滤")
        
        # ========== 步骤6: 拆分数据集 ==========
        logger.info("\n[步骤6/8] 拆分数据集...")
        df = pd.DataFrame(all_data)
        
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        logger.info(f"✓ 拆分完成:")
        logger.info(f"  训练集: {len(train_df)} 个样本 (80%)")
        logger.info(f"  验证集: {len(val_df)} 个样本 (10%)")
        logger.info(f"  测试集: {len(test_df)} 个样本 (10%)")
        
        # ========== 步骤7: 数据配比（仅训练集）==========
        logger.info("\n[步骤7/8] 训练集数据配比...")
        if balance_config:
            logger.info("开始对训练集进行配比...")
            balanced_samples = self.balance_dataset(train_df, balance_config)
            train_df = pd.DataFrame(balanced_samples)
            logger.info(f"✓ 配比完成，训练集样本数: {len(train_df)}")
        else:
            logger.info("✓ 跳过配比（未提供配置）")
        
        # ========== 步骤8: 保存结果 ==========
        logger.info("\n[步骤8/8] 保存结果...")
        
        # 保存CSV格式
        train_df.to_csv(self.output_dir / "train.csv", index=False, encoding='utf-8-sig')
        val_df.to_csv(self.output_dir / "validation.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(self.output_dir / "test.csv", index=False, encoding='utf-8-sig')
        
        # 保存JSON格式
        train_df.to_json(self.output_dir / "train.json", orient='records', 
                         force_ascii=False, indent=2)
        val_df.to_json(self.output_dir / "validation.json", orient='records', 
                        force_ascii=False, indent=2)
        test_df.to_json(self.output_dir / "test.json", orient='records', 
                        force_ascii=False, indent=2)
        
        logger.info(f"✓ 数据已保存到: {self.output_dir}")
        
        # 保存过滤统计报告
        self._save_filter_report(df)
        
        # ========== 最终统计信息 ==========
        logger.info("\n" + "="*60)
        logger.info("数据处理完成 - 最终统计")
        logger.info("="*60)
        logger.info(f"处理的文件数: {self.stats['processed_files']}")
        logger.info(f"原始样本数: {self.stats['processed_samples']}")
        logger.info(f"格式无效过滤: {self.stats['filtered_samples']}")
        logger.info(f"隐私过滤: {self.stats['privacy_filtered']}")
        logger.info(f"去重移除: {duplicate_count}")
        logger.info(f"标注样本数: {self.stats['annotated_samples']}")
        logger.info(f"质量过滤: {self.stats['quality_filtered']}")
        logger.info(f"最终样本数: {len(df)}")
        logger.info(f"  - 训练集: {len(train_df)} {'(已配比)' if balance_config else ''}")
        logger.info(f"  - 验证集: {len(val_df)}")
        logger.info(f"  - 测试集: {len(test_df)}")
        logger.info("="*60)
        
        return {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }
    
    def _save_filter_report(self, df: pd.DataFrame) -> None:
        """
        保存过滤统计报告
        
        Args:
            df: 处理后的DataFrame
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats.copy(),
            "label_distribution": {},
            "score_statistics": {}
        }
        
        # 标签分布统计
        if "primary_label" in df.columns:
            label_counts = df["primary_label"].value_counts().to_dict()
            report["label_distribution"] = label_counts
        
        # 评分统计
        if "scores" in df.columns and not df["scores"].isna().all():
            score_fields = ["safety", "relevance", "authenticity", "uncertainty", "helpfulness"]
            for field in score_fields:
                scores = []
                for idx, row in df.iterrows():
                    if isinstance(row.get("scores"), dict):
                        score = row["scores"].get(field)
                        if score is not None:
                            scores.append(score)
                
                if scores:
                    report["score_statistics"][field] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                        "median": np.median(scores)
                    }
        
        # 保存报告
        report_path = self.output_dir / "filter_report.json"
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        report = convert_to_native(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"过滤统计报告已保存: {report_path}")
    
    def balance_dataset(
        self,
        samples: Union[List[Dict[str, Any]], pd.DataFrame],
        balance_config: Dict[str, Any],
        label_field: str = "primary_label"
    ) -> List[Dict[str, Any]]:
        """
        根据配置对数据集进行配比
        
        Args:
            samples: 样本列表或DataFrame
            balance_config: 配比配置字典
                {
                    "method": "ratios" | "counts" | "uniform" | "min" | "max",
                    "target_ratios": {...},  # method="ratios"时使用
                    "target_counts": {...},  # method="counts"时使用
                    "total_samples": int,    # method="ratios"时可选
                    "min_samples": int,      # method="min"时使用
                    "max_samples": int,      # method="max"时使用
                    "target_count": int,     # method="uniform"时可选
                    "strategy": "oversample" | "undersample" | "smart"
                }
            label_field: 标签字段名
            
        Returns:
            配比后的样本列表
        """
        # 转换DataFrame为列表
        if isinstance(samples, pd.DataFrame):
            samples = samples.to_dict('records')
        
        # 初始化配比器
        balancer = DataBalancer()
        
        # 显示原始分布
        logger.info("原始标签分布:")
        original_dist = balancer.get_label_distribution(samples, label_field)
        for label, count in sorted(original_dist.items()):
            logger.info(f"  {label}: {count}")
        
        # 根据配置选择方法
        method = balance_config.get("method", "ratios")
        
        if method == "ratios":
            # 按比例配比
            target_ratios = balance_config.get("target_ratios", {})
            total_samples = balance_config.get("total_samples", None)
            strategy = balance_config.get("strategy", "smart")
            
            balanced = balancer.balance_by_ratios(
                samples,
                target_ratios,
                total_samples,
                label_field,
                strategy
            )
        
        elif method == "counts":
            # 按数量配比
            target_counts = balance_config.get("target_counts", {})
            strategy = balance_config.get("strategy", "smart")
            
            balanced = balancer.balance_by_target_counts(
                samples,
                target_counts,
                label_field,
                strategy
            )
        
        elif method == "uniform":
            # 均匀配比
            target_count = balance_config.get("target_count", None)
            
            balanced = balancer.balance_uniform(
                samples,
                label_field,
                target_count
            )
        
        elif method == "min":
            # 最小样本数限制
            min_samples = balance_config.get("min_samples", 100)
            
            balanced = balancer.balance_by_min_samples(
                samples,
                min_samples,
                label_field
            )
        
        elif method == "max":
            # 最大样本数限制
            max_samples = balance_config.get("max_samples", 1000)
            
            balanced = balancer.balance_by_max_samples(
                samples,
                max_samples,
                label_field
            )
        
        else:
            raise ValueError(f"未知的配比方法: {method}")
        
        # 显示配比后分布
        logger.info("\n配比后标签分布:")
        balanced_dist = balancer.get_label_distribution(balanced, label_field)
        for label, count in sorted(balanced_dist.items()):
            percentage = (count / len(balanced)) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\n总样本数: {len(samples)} -> {len(balanced)}")
        
        return balanced
    
    def evaluate_test_samples(
        self,
        test_samples: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
        output_path: Optional[Union[str, Path]] = None,
        batch_size: int = 10,
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        评估测试样本的质量
        
        Args:
            test_samples: 测试样本（列表、DataFrame或文件路径）
            output_path: 评估报告输出路径
            batch_size: 批次大小
            max_workers: 最大并发数
            
        Returns:
            评估报告字典
        """
        # 检查是否有DeepSeek API密钥
        if not self.deepseek_annotator:
            raise ValueError("需要DeepSeek API密钥才能进行评估，请在初始化时提供 deepseek_api_key")
        
        # 加载样本
        if isinstance(test_samples, (str, Path)):
            test_samples = Path(test_samples)
            if test_samples.suffix == '.csv':
                df = pd.read_csv(test_samples)
                samples = df.to_dict('records')
            elif test_samples.suffix == '.json':
                with open(test_samples, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
            else:
                raise ValueError(f"不支持的文件格式: {test_samples.suffix}")
        elif isinstance(test_samples, pd.DataFrame):
            samples = test_samples.to_dict('records')
        else:
            samples = test_samples
        
        logger.info(f"开始评估 {len(samples)} 个测试样本...")
        
        # 创建评估器
        evaluator = MedicalQAEvaluator(
            api_key=self.deepseek_annotator.api_key,
            base_url=self.deepseek_annotator.base_url
        )
        
        # 批量评估
        evaluated_samples = evaluator.batch_evaluate(
            samples,
            batch_size=batch_size,
            max_workers=max_workers
        )
        
        # 生成报告
        if output_path is None:
            output_path = self.output_dir / "evaluation_report.json"
        
        report = evaluator.generate_evaluation_report(evaluated_samples, output_path)
        
        # 保存评估后的样本
        evaluated_output = self.output_dir / "evaluated_samples.json"
        with open(evaluated_output, 'w', encoding='utf-8') as f:
            json.dump(evaluated_samples, f, ensure_ascii=False, indent=2)
        logger.info(f"评估后的样本已保存: {evaluated_output}")
        
        # 过滤出通过的样本
        passed_samples = [s for s in evaluated_samples if s.get("eval_pass", False)]
        if passed_samples:
            passed_output = self.output_dir / "passed_samples.json"
            with open(passed_output, 'w', encoding='utf-8') as f:
                json.dump(passed_samples, f, ensure_ascii=False, indent=2)
            logger.info(f"通过评估的样本已保存: {passed_output} ({len(passed_samples)} 个)")
        
        # 过滤出有问题的样本
        problem_samples = [s for s in evaluated_samples if not s.get("eval_pass", False)]
        if problem_samples:
            problem_output = self.output_dir / "problem_samples.json"
            with open(problem_output, 'w', encoding='utf-8') as f:
                json.dump(problem_samples, f, ensure_ascii=False, indent=2)
            logger.info(f"有问题的样本已保存: {problem_output} ({len(problem_samples)} 个)")
        
        return report


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="医疗问答数据处理工具")
    parser.add_argument("--data_dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--medical_dict", type=str, help="医学词典路径")
    parser.add_argument("--stopwords", type=str, help="停用词表路径")
    parser.add_argument("--workers", type=int, default=4, help="最大工作线程数")
    parser.add_argument("--deepseek_api_key", type=str, help="DeepSeek API密钥")
    parser.add_argument("--enable_annotation", action="store_true", help="启用DeepSeek标注")
    parser.add_argument("--enable_privacy_filter", action="store_true", default=True, 
                        help="启用隐私过滤（默认启用）")
    parser.add_argument("--enable_quality_filter", action="store_true", default=True,
                        help="启用质量过滤（默认启用）")
    
    args = parser.parse_args()
    
    processor = MedicalDataProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        medical_dict_path=args.medical_dict,
        stopwords_path=args.stopwords,
        max_workers=args.workers,
        deepseek_api_key=args.deepseek_api_key,
        enable_privacy_filter=args.enable_privacy_filter,
        enable_quality_filter=args.enable_quality_filter
    )
    
    # 处理所有数据
    processor.process_all_data(enable_annotation=args.enable_annotation)




class DPODataProcessor:
    pass







class DPODataFilter:
    pass