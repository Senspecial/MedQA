# utils/text_utils.py

import re
import unicodedata
import hashlib
import random
import string
import jieba
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import json
from collections import Counter
import difflib
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextNormalizer:
    """文本标准化工具"""
    
    @staticmethod
    def normalize_unicode(text: str, form: str = "NFKC") -> str:
        """Unicode标准化"""
        return unicodedata.normalize(form, text)
    
    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """移除多余空格"""
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空格
        return text.strip()
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """移除HTML标签"""
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """移除URL"""
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    
    @staticmethod
    def remove_emojis(text: str) -> str:
        """移除表情符号"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # 表情符号
            "\U0001F300-\U0001F5FF"  # 符号和象形文字
            "\U0001F680-\U0001F6FF"  # 交通和地图符号
            "\U0001F700-\U0001F77F"  # 炼金术符号
            "\U0001F780-\U0001F7FF"  # 几何形状
            "\U0001F800-\U0001F8FF"  # 补充箭头
            "\U0001F900-\U0001F9FF"  # 补充符号和象形文字
            "\U0001FA00-\U0001FA6F"  # 国际象棋符号
            "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展
            "\U00002702-\U000027B0"  # 装饰符号
            "\U000024C2-\U0001F251" 
            "]+"
        )
        return emoji_pattern.sub('', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """标准化空白字符"""
        # 将各种空白字符（制表符、换行符等）转换为空格
        text = re.sub(r'[\s\t\n\r\f\v]+', ' ', text)
        return text.strip()
    
    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """标准化标点符号"""
        # 统一中文和英文标点
        punctuation_map = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
            '—': '-'
        }
        
        for cn_punct, en_punct in punctuation_map.items():
            text = text.replace(cn_punct, en_punct)
        
        return text
    
    @staticmethod
    def full_to_half_width(text: str) -> str:
        """全角转半角"""
        result = ""
        for char in text:
            code = ord(char)
            # 全角空格直接转换
            if code == 0x3000:
                code = 0x0020
            # 全角字符（除空格外）根据关系转换
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            result += chr(code)
        return result
    
    @staticmethod
    def normalize_text(
        text: str,
        lowercase: bool = True,
        remove_html: bool = True,
        remove_url: bool = True,
        remove_emoji: bool = False,
        normalize_unicode: bool = True,
        normalize_spaces: bool = True,
        normalize_punct: bool = False,
        to_half_width: bool = False
    ) -> str:
        """综合文本标准化"""
        if normalize_unicode:
            text = TextNormalizer.normalize_unicode(text)
        
        if remove_html:
            text = TextNormalizer.remove_html_tags(text)
        
        if remove_url:
            text = TextNormalizer.remove_urls(text)
        
        if remove_emoji:
            text = TextNormalizer.remove_emojis(text)
        
        if normalize_spaces:
            text = TextNormalizer.normalize_whitespace(text)
        
        if normalize_punct:
            text = TextNormalizer.normalize_punctuation(text)
        
        if to_half_width:
            text = TextNormalizer.full_to_half_width(text)
        
        if lowercase:
            text = text.lower()
        
        return text


class TextSegmenter:
    """文本分词工具"""
    
    def __init__(self, user_dict_path: Optional[str] = None):
        """初始化分词工具"""
        if user_dict_path:
            try:
                jieba.load_userdict(user_dict_path)
                logger.info(f"Loaded user dictionary from {user_dict_path}")
            except Exception as e:
                logger.error(f"Failed to load user dictionary: {e}")
    
    def segment(self, text: str, cut_all: bool = False, HMM: bool = True) -> List[str]:
        """分词"""
        return list(jieba.cut(text, cut_all=cut_all, HMM=HMM))
    
    def segment_precise(self, text: str) -> List[str]:
        """精确模式分词"""
        return list(jieba.cut(text, cut_all=False))
    
    def segment_full(self, text: str) -> List[str]:
        """全模式分词"""
        return list(jieba.cut(text, cut_all=True))
    
    def segment_search(self, text: str) -> List[str]:
        """搜索引擎模式分词"""
        return list(jieba.cut_for_search(text))
    
    def extract_keywords(self, text: str, topK: int = 20, withWeight: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """提取关键词"""
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=topK, withWeight=withWeight)
    
    def extract_tfidf_keywords(self, text: str, topK: int = 20, withWeight: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """使用TFIDF提取关键词"""
        import jieba.analyse
        return jieba.analyse.extract_tags(text, topK=topK, withWeight=withWeight)
    
    def extract_textrank_keywords(self, text: str, topK: int = 20, withWeight: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """使用TextRank提取关键词"""
        import jieba.analyse
        return jieba.analyse.textrank(text, topK=topK, withWeight=withWeight)
    
    def add_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None) -> None:
        """添加新词到词典"""
        jieba.add_word(word, freq, tag)
    
    def get_sentence_segments(self, text: str) -> List[str]:
        """获取句子分段"""
        # 使用标点符号分割句子
        pattern = r'[。！？!?]+'
        segments = re.split(pattern, text)
        return [segment.strip() for segment in segments if segment.strip()]
    
    def segment_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """分词并返回词性标注"""
        import jieba.posseg as pseg
        return [(w, p) for w, p in pseg.cut(text)]
    
    def get_stopwords(self, stopwords_path: str) -> Set[str]:
        """从文件加载停用词"""
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
            return set()
    
    def remove_stopwords(self, tokens: List[str], stopwords: Set[str]) -> List[str]:
        """移除停用词"""
        return [token for token in tokens if token not in stopwords]


class TextHasher:
    """文本哈希工具"""
    
    @staticmethod
    def md5(text: str) -> str:
        """计算MD5哈希"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha1(text: str) -> str:
        """计算SHA1哈希"""
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def sha256(text: str) -> str:
        """计算SHA256哈希"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_hash(text: str, algorithm: str = "md5") -> str:
        """根据指定算法计算哈希"""
        if algorithm == "md5":
            return TextHasher.md5(text)
        elif algorithm == "sha1":
            return TextHasher.sha1(text)
        elif algorithm == "sha256":
            return TextHasher.sha256(text)
        else:
            raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    @staticmethod
    def get_simhash(text: str) -> str:
        """计算SimHash (用于近似文本重复检测)"""
        try:
            import simhash
            return str(simhash.Simhash(text.split()).value)
        except ImportError:
            logger.warning("simhash 模块未安装，无法计算 SimHash")
            return TextHasher.md5(text)


class TextGenerator:
    """文本生成工具"""
    
    @staticmethod
    def generate_random_string(length: int, include_digits: bool = True, 
                             include_letters: bool = True, include_punctuation: bool = False) -> str:
        """生成随机字符串"""
        chars = ""
        
        if include_letters:
            chars += string.ascii_letters
        if include_digits:
            chars += string.digits
        if include_punctuation:
            chars += string.punctuation
        
        if not chars:
            raise ValueError("至少需要包含一种字符类型")
        
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def generate_uuid(as_hex: bool = True) -> str:
        """生成UUID"""
        import uuid
        if as_hex:
            return uuid.uuid4().hex
        else:
            return str(uuid.uuid4())
    
    @staticmethod
    def generate_random_sentence(word_count: int = 10, lang: str = "en") -> str:
        """生成随机句子"""
        if lang == "en":
            # 英文随机单词
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                   "a", "an", "and", "in", "on", "at", "with", "by", "to", "from",
                   "for", "of", "is", "was", "am", "are", "were", "be", "being",
                   "been", "have", "has", "had", "do", "does", "did", "can", "could",
                   "will", "would", "shall", "should", "may", "might", "must"]
        elif lang == "zh":
            # 中文随机词
            words = ["我", "你", "他", "她", "它", "我们", "你们", "他们",
                   "这", "那", "这里", "那里", "这个", "那个", "今天", "明天",
                   "昨天", "早上", "中午", "晚上", "春", "夏", "秋", "冬",
                   "去", "来", "走", "跑", "跳", "看", "说", "听", "吃", "喝"]
        else:
            raise ValueError(f"不支持的语言: {lang}")
        
        # 生成随机单词序列
        sentence = " ".join(random.choice(words) for _ in range(word_count))
        
        # 首字母大写并添加句号
        if lang == "en":
            return sentence[0].upper() + sentence[1:] + "."
        else:
            return sentence + "。"
    
    @staticmethod
    def generate_lorem_ipsum(paragraphs: int = 1, sentences_per_paragraph: int = 5) -> str:
        """生成Lorem Ipsum假文"""
        lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
        lorem_sentences = [s.strip() for s in lorem_ipsum.split('.') if s.strip()]
        
        result = []
        for p in range(paragraphs):
            paragraph_sentences = []
            for s in range(sentences_per_paragraph):
                sentence_idx = (p * sentences_per_paragraph + s) % len(lorem_sentences)
                paragraph_sentences.append(lorem_sentences[sentence_idx] + ".")
            result.append(" ".join(paragraph_sentences))
        
        return "\n\n".join(result)
    
    @staticmethod
    def generate_chinese_lorem(paragraphs: int = 1, sentences_per_paragraph: int = 5) -> str:
        """生成中文假文"""
        chinese_lorem = "人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。人人有资格享有本宣言所载的一切权利和自由，不分种族、肤色、性别、语言、宗教、政治或其他见解、国籍或社会出身、财产、出生或其他身份等任何区别。并且不应当由于一个人所属的国家或领土的政治的、行政的或者国际的地位之不同而有所区别，无论该领土是独立领土、托管领土、非自治领土或者处于其他任何主权受限制的情况之下。"
        chinese_sentences = [s.strip() for s in chinese_lorem.split('。') if s.strip()]
        
        result = []
        for p in range(paragraphs):
            paragraph_sentences = []
            for s in range(sentences_per_paragraph):
                sentence_idx = (p * sentences_per_paragraph + s) % len(chinese_sentences)
                paragraph_sentences.append(chinese_sentences[sentence_idx] + "。")
            result.append("".join(paragraph_sentences))
        
        return "\n\n".join(result)


class TextAnalyzer:
    """文本分析工具"""
    
    @staticmethod
    def count_words(text: str) -> int:
        """计算单词数量"""
        words = text.split()
        return len(words)
    
    @staticmethod
    def count_chinese_words(text: str) -> int:
        """计算中文词数（使用jieba分词）"""
        words = jieba.cut(text)
        return len(list(words))
    
    @staticmethod
    def count_characters(text: str, include_spaces: bool = False) -> int:
        """计算字符数"""
        if include_spaces:
            return len(text)
        else:
            return len(text.replace(" ", ""))
    
    @staticmethod
    def count_sentences(text: str) -> int:
        """计算句子数"""
        # 英文和中文的句子结束符
        pattern = r'[.。!！?？]+(?:\s|$)'
        sentences = re.split(pattern, text)
        # 过滤空字符串
        sentences = [s for s in sentences if s.strip()]
        return len(sentences)
    
    @staticmethod
    def get_word_frequency(text: str, top_n: Optional[int] = None) -> Dict[str, int]:
        """获取单词频率"""
        words = text.split()
        counter = Counter(words)
        
        if top_n:
            return dict(counter.most_common(top_n))
        return dict(counter)
    
    @staticmethod
    def get_chinese_word_frequency(text: str, top_n: Optional[int] = None) -> Dict[str, int]:
        """获取中文词频率"""
        words = jieba.cut(text)
        counter = Counter(words)
        
        if top_n:
            return dict(counter.most_common(top_n))
        return dict(counter)
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str, method: str = "jaccard") -> float:
        """计算文本相似度"""
        if method == "jaccard":
            # Jaccard相似度：交集大小除以并集大小
            set1 = set(text1.split())
            set2 = set(text2.split())
            
            if not set1 and not set2:
                return 1.0
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union
        
        elif method == "cosine":
            # 余弦相似度的简单实现
            from collections import Counter
            vec1 = Counter(text1.split())
            vec2 = Counter(text2.split())
            
            intersection = set(vec1.keys()) & set(vec2.keys())
            
            numerator = sum([vec1[x] * vec2[x] for x in intersection])
            
            sum1 = sum([vec1[x]**2 for x in vec1.keys()])
            sum2 = sum([vec2[x]**2 for x in vec2.keys()])
            denominator = (sum1 * sum2) ** 0.5
            
            if not denominator:
                return 0.0
            return numerator / denominator
        
        elif method == "levenshtein":
            # 莱文斯坦距离
            try:
                import Levenshtein
                distance = Levenshtein.distance(text1, text2)
                max_len = max(len(text1), len(text2))
                
                if max_len == 0:
                    return 1.0
                
                return 1 - (distance / max_len)
            except ImportError:
                logger.warning("Levenshtein 模块未安装，回退到序列匹配")
                method = "sequence"
        
        if method == "sequence":
            # 序列匹配比率
            matcher = difflib.SequenceMatcher(None, text1, text2)
            return matcher.ratio()
        
        else:
            raise ValueError(f"不支持的相似度计算方法: {method}")
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """提取文本中的数字"""
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(match) for match in matches]
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """提取文本中的邮箱地址"""
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """提取文本中的URL"""
        pattern = r'https?://[^\s]+'
        return re.findall(pattern, text)
    
    @staticmethod
    def extract_chinese(text: str) -> str:
        """提取中文字符"""
        pattern = r'[\u4e00-\u9fa5]+'
        matches = re.findall(pattern, text)
        return ''.join(matches)
    
    @staticmethod
    def is_chinese_text(text: str, threshold: float = 0.5) -> bool:
        """判断是否为中文文本"""
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        chinese_ratio = len(chinese_chars) / len(text) if text else 0
        return chinese_ratio >= threshold
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """提取文本中的日期"""
        # 匹配常见日期格式
        patterns = [
            r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',  # 2023-01-01, 2023年01月01日
            r'\d{1,2}[-/月]\d{1,2}[-/日号]?\s*,?\s*\d{4}',  # 01-01-2023, 01月01日, 2023
            r'\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'  # 1st Jan 2023
        ]
        
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            results.extend(matches)
            
        return results
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, float]:
        """简单的情感分析"""
        try:
            from snownlp import SnowNLP
            s = SnowNLP(text)
            return {
                "positive_prob": s.sentiments,
                "negative_prob": 1 - s.sentiments
            }
        except ImportError:
            logger.warning("SnowNLP 模块未安装，无法进行情感分析")
            # 返回中性情感
            return {
                "positive_prob": 0.5,
                "negative_prob": 0.5
            }
    
    @staticmethod
    def get_text_readability(text: str, lang: str = "zh") -> Dict[str, float]:
        """计算文本可读性指标"""
        result = {}
        
        if lang == "zh":
            # 中文文本可读性
            sentences = TextAnalyzer.count_sentences(text)
            words = TextAnalyzer.count_chinese_words(text)
            chars = TextAnalyzer.count_characters(text, include_spaces=False)
            
            # 平均句长
            if sentences > 0:
                avg_sentence_length = words / sentences
            else:
                avg_sentence_length = 0
            
            # 平均词长
            if words > 0:
                avg_word_length = chars / words
            else:
                avg_word_length = 0
            
            # 简化的可读性指标
            readability = 100 - (avg_sentence_length * 0.6 + avg_word_length * 5)
            
            result = {
                "avg_sentence_length": avg_sentence_length,
                "avg_word_length": avg_word_length,
                "readability": max(0, min(100, readability))  # 限制在0-100范围内
            }
            
        else:
            # 英文文本可读性 (简化的弗莱奇易读性指数)
            sentences = TextAnalyzer.count_sentences(text)
            words = TextAnalyzer.count_words(text)
            syllables = TextAnalyzer._count_syllables(text)
            
            if words > 0 and sentences > 0:
                flesch_reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
                flesch_kincaid_grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
            else:
                flesch_reading_ease = 0
                flesch_kincaid_grade = 0
            
            result = {
                "flesch_reading_ease": max(0, min(100, flesch_reading_ease)),
                "flesch_kincaid_grade": max(0, flesch_kincaid_grade)
            }
        
        return result
    
    @staticmethod
    def _count_syllables(text: str) -> int:
        """计算英文文本的音节数（估计值）"""
        # 简化的计算方法
        text = text.lower()
        text = re.sub(r'[^a-z]', ' ', text)
        words = text.split()
        
        syllable_count = 0
        for word in words:
            word = word.strip()
            if not word:
                continue
                
            # 特殊情况: 单词以'e'结尾但不是'le'
            if word.endswith('e') and not word.endswith('le'):
                word = word[:-1]
                
            # 计算元音组
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # 确保每个单词至少有一个音节
            if count == 0:
                count = 1
                
            syllable_count += count
            
        return syllable_count


class MedicalTextProcessor:
    """医疗文本处理工具"""
    
    def __init__(self, medical_dict_path: Optional[str] = None):
        """初始化医疗文本处理器"""
        self.segmenter = TextSegmenter(user_dict_path=medical_dict_path)
        
        # 常见医学单位词表
        self.medical_units = {
            "mg", "g", "kg", "ml", "l", "mmol", "mol", "mmHg", "kPa", "mm",
            "cm", "m", "IU", "U", "μg", "ng", "pg", "mmol/L", "g/L", "mEq/L"
        }
        
        # 常见医疗术语缩写映射
        self.medical_abbr = {
            "BP": "血压",
            "HR": "心率",
            "RR": "呼吸频率",
            "BT": "体温",
            "SpO2": "血氧饱和度",
            "WBC": "白细胞计数",
            "RBC": "红细胞计数",
            "Hb": "血红蛋白",
            "T": "体温",
            "P": "脉搏",
            "R": "呼吸"
        }
        
        # 加载医学词典
        if medical_dict_path:
            self._load_medical_dict(medical_dict_path)
    
    def _load_medical_dict(self, dict_path: str) -> None:
        """加载医学词典"""
        try:
            jieba.load_userdict(dict_path)
            logger.info(f"已加载医学词典: {dict_path}")
        except Exception as e:
            logger.error(f"加载医学词典失败: {e}")
    
    def segment_medical_text(self, text: str) -> List[str]:
        """分词医疗文本"""
        return self.segmenter.segment(text)
    
    def extract_medical_terms(self, text: str, term_dict: Optional[Dict[str, str]] = None) -> List[str]:
        """提取医学术语"""
        # 如果提供了术语词典，使用词典匹配
        if term_dict:
            terms = []
            for term in term_dict.keys():
                if term in text:
                    terms.append(term)
            return terms
        
        # 否则使用关键词提取
        return self.segmenter.extract_keywords(text)
    
    def normalize_medical_unit(self, text: str) -> str:
        """标准化医疗单位"""
        # 匹配数字和单位的模式
        number_unit_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:/[a-zA-Z]+)?)'
        
        def replace_unit(match):
            number, unit = match.groups()
            # 检查单位是否在医学单位列表中
            if unit.lower() in self.medical_units:
                # 返回标准格式：数字和单位之间有一个空格
                return f"{number} {unit}"
            return match.group(0)
        
        # 替换文本中的单位格式
        return re.sub(number_unit_pattern, replace_unit, text)
    
    def expand_medical_abbreviations(self, text: str) -> str:
        """展开医学缩写"""
        expanded_text = text
        
        # 替换缩写
        for abbr, full in self.medical_abbr.items():
            # 只替换独立的缩写词（前后有空格或标点）
            pattern = r'(?<![a-zA-Z])' + re.escape(abbr) + r'(?![a-zA-Z])'
            expanded_text = re.sub(pattern, f"{abbr}({full})", expanded_text)
        
        return expanded_text
    
    def extract_medical_values(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """提取医疗数值"""
        results = {}
        
        # 提取各种医疗数值
        # 1. 血压
        bp_pattern = r'BP[：:]\s*(\d+)/(\d+)\s*(?:mmHg)?'
        bp_matches = re.finditer(bp_pattern, text)
        bp_results = []
        
        for match in bp_matches:
            systolic = int(match.group(1))
            diastolic = int(match.group(2))
            bp_results.append({
                "systolic": systolic,
                "diastolic": diastolic,
                "unit": "mmHg"
            })
        
        if bp_results:
            results["blood_pressure"] = bp_results
        
        # 2. 体温
        temp_pattern = r'(?:T|BT|体温)[：:]\s*(\d+\.?\d*)\s*(?:℃|度)'
        temp_matches = re.finditer(temp_pattern, text)
        temp_results = []
        
        for match in temp_matches:
            temperature = float(match.group(1))
            temp_results.append({
                "value": temperature,
                "unit": "℃"
            })
        
        if temp_results:
            results["temperature"] = temp_results
        
        # 3. 血糖
        glucose_pattern = r'血糖[：:]\s*(\d+\.?\d*)\s*(?:mmol/L)'
        glucose_matches = re.finditer(glucose_pattern, text)
        glucose_results = []
        
        for match in glucose_matches:
            glucose = float(match.group(1))
            glucose_results.append({
                "value": glucose,
                "unit": "mmol/L"
            })
        
        if glucose_results:
            results["blood_glucose"] = glucose_results
        
        # 4. 一般匹配测试数值
        general_pattern = r'(\w+)\s*(?:[:：])\s*(\d+\.?\d*)\s*(?:[a-zA-Z]+(?:/[a-zA-Z]+)?)?'
        general_matches = re.finditer(general_pattern, text)
        
        for match in general_matches:
            test_name = match.group(1)
            value = float(match.group(2))
            
            # 排除已经处理过的项目
            if test_name not in ["BP", "T", "BT", "体温", "血糖"]:
                if test_name not in results:
                    results[test_name] = []
                
                results[test_name].append({
                    "value": value
                })
        
        return results
    
    def format_medical_report(self, report_data: Dict[str, Any], template: Optional[str] = None) -> str:
        """格式化医疗报告数据"""
        if not template:
            template = (
                "医疗报告\n"
                "==============================\n"
                "{date}\n\n"
                "患者信息：\n"
                "姓名: {patient_name}\n"
                "性别: {patient_gender}\n"
                "年龄: {patient_age}\n"
                "ID: {patient_id}\n\n"
                "检查结果：\n"
                "{examination_results}\n\n"
                "诊断：\n"
                "{diagnosis}\n\n"
                "建议：\n"
                "{recommendations}\n"
                "==============================\n"
                "医师: {doctor_name}"
            )
        
        # 格式化检查结果
        examination_results = ""
        if "examination_results" in report_data and isinstance(report_data["examination_results"], dict):
            for test, results in report_data["examination_results"].items():
                if isinstance(results, list):
                    for result in results:
                        value = result.get("value", "")
                        unit = result.get("unit", "")
                        examination_results += f"- {test}: {value}{' ' + unit if unit else ''}\n"
                else:
                    examination_results += f"- {test}: {results}\n"
        else:
            examination_results = report_data.get("examination_results", "无")
        
        # 使用模板替换字段
        formatted_report = template
        for key, value in report_data.items():
            if key != "examination_results":  # 已经单独处理
                formatted_report = formatted_report.replace("{" + key + "}", str(value))
        
        # 替换检查结果
        formatted_report = formatted_report.replace("{examination_results}", examination_results)
        
        # 替换任何遗漏的字段
        formatted_report = re.sub(r'\{[^}]+\}', '无', formatted_report)
        
        return formatted_report
    
    def detect_adverse_conditions(self, text: str, threshold_dict: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict[str, Any]]:
        """检测异常医疗指标"""
        # 默认阈值
        default_thresholds = {
            "blood_pressure": {"systolic_min": 90, "systolic_max": 140, "diastolic_min": 60, "diastolic_max": 90},
            "temperature": {"min": 36.0, "max": 37.3},
            "blood_glucose": {"min": 3.9, "max": 6.1}
        }
        
        # 合并用户提供的阈值
        thresholds = default_thresholds.copy()
        if threshold_dict:
            for key, value in threshold_dict.items():
                if key in thresholds:
                    thresholds[key].update(value)
                else:
                    thresholds[key] = value
        
        # 提取医疗数值
        medical_values = self.extract_medical_values(text)
        
        # 检测异常
        abnormal_findings = []
        
        # 血压异常
        if "blood_pressure" in medical_values:
            for bp in medical_values["blood_pressure"]:
                systolic = bp.get("systolic")
                diastolic = bp.get("diastolic")
                
                if systolic and diastolic:
                    systolic_min = thresholds["blood_pressure"]["systolic_min"]
                    systolic_max = thresholds["blood_pressure"]["systolic_max"]
                    diastolic_min = thresholds["blood_pressure"]["diastolic_min"]
                    diastolic_max = thresholds["blood_pressure"]["diastolic_max"]
                    
                    if systolic < systolic_min or systolic > systolic_max or diastolic < diastolic_min or diastolic > diastolic_max:
                        abnormal_findings.append({
                            "type": "blood_pressure",
                            "value": f"{systolic}/{diastolic} mmHg",
                            "normal_range": f"{systolic_min}-{systolic_max}/{diastolic_min}-{diastolic_max} mmHg",
                            "is_high": systolic > systolic_max or diastolic > diastolic_max,
                            "is_low": systolic < systolic_min or diastolic < diastolic_min
                        })
        
        # 体温异常
        if "temperature" in medical_values:
            for temp in medical_values["temperature"]:
                value = temp.get("value")
                
                if value:
                    min_temp = thresholds["temperature"]["min"]
                    max_temp = thresholds["temperature"]["max"]
                    
                    if value < min_temp or value > max_temp:
                        abnormal_findings.append({
                            "type": "temperature",
                            "value": f"{value} ℃",
                            "normal_range": f"{min_temp}-{max_temp} ℃",
                            "is_high": value > max_temp,
                            "is_low": value < min_temp
                        })
        
        # 血糖异常
        if "blood_glucose" in medical_values:
            for glucose in medical_values["blood_glucose"]:
                value = glucose.get("value")
                
                if value:
                    min_glucose = thresholds["blood_glucose"]["min"]
                    max_glucose = thresholds["blood_glucose"]["max"]
                    
                    if value < min_glucose or value > max_glucose:
                        abnormal_findings.append({
                            "type": "blood_glucose",
                            "value": f"{value} mmol/L",
                            "normal_range": f"{min_glucose}-{max_glucose} mmol/L",
                            "is_high": value > max_glucose,
                            "is_low": value < min_glucose
                        })
        
        # 检查其他可能的异常项目
        for test_name, values in medical_values.items():
            if test_name not in ["blood_pressure", "temperature", "blood_glucose"] and test_name in thresholds:
                for value_obj in values:
                    value = value_obj.get("value")
                    
                    if value:
                        min_val = thresholds[test_name].get("min")
                        max_val = thresholds[test_name].get("max")
                        
                        if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                            abnormal_findings.append({
                                "type": test_name,
                                "value": str(value),
                                "normal_range": f"{min_val if min_val is not None else '?'}-{max_val if max_val is not None else '?'}",
                                "is_high": max_val is not None and value > max_val,
                                "is_low": min_val is not None and value < min_val
                            })
        
        return abnormal_findings
    
    def extract_diagnoses(self, text: str) -> List[str]:
        """从医疗文本中提取诊断信息"""
        diagnoses = []
        
        # 常见的诊断部分标识符
        diagnosis_patterns = [
            r'诊断[：:]\s*(.+?)(?=\n|$)',
            r'初步诊断[：:]\s*(.+?)(?=\n|$)',
            r'临床诊断[：:]\s*(.+?)(?=\n|$)',
            r'出院诊断[：:]\s*(.+?)(?=\n|$)'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                diagnosis = match.group(1).strip()
                if diagnosis:
                    # 分割多个诊断
                    if "；" in diagnosis:
                        diagnoses.extend([d.strip() for d in diagnosis.split("；") if d.strip()])
                    elif ";" in diagnosis:
                        diagnoses.extend([d.strip() for d in diagnosis.split(";") if d.strip()])
                    elif "，" in diagnosis:
                        diagnoses.extend([d.strip() for d in diagnosis.split("，") if d.strip()])
                    elif "," in diagnosis:
                        diagnoses.extend([d.strip() for d in diagnosis.split(",") if d.strip()])
                    else:
                        diagnoses.append(diagnosis)
        
        # 去重
        return list(dict.fromkeys(diagnoses))
    
    def extract_medications(self, text: str) -> List[Dict[str, Any]]:
        """从医疗文本中提取药物信息"""
        medications = []
        
        # 匹配药物名称、剂量和频率
        med_pattern = r'(?:用药|处方|医嘱)[：:]\s*(.+?)(?=\n|$)'
        med_matches = re.finditer(med_pattern, text)
        
        for med_match in med_matches:
            med_text = med_match.group(1).strip()
            
            # 分割多个药物
            med_items = re.split(r'[,，;；]', med_text)
            
            for item in med_items:
                item = item.strip()
                if not item:
                    continue
                
                # 尝试提取药物名称、剂量和频率
                name_dose_pattern = r'([^\d]+?)\s*(\d+\.?\d*\s*(?:mg|g|ml|mcg|μg|IU|U|单位))?(?:\s*([^，,]*?[每日天一二三四五六七八九十]+次))?'
                match = re.search(name_dose_pattern, item)
                
                if match:
                    name = match.group(1).strip() if match.group(1) else item
                    dose = match.group(2).strip() if match.group(2) else None
                    frequency = match.group(3).strip() if match.group(3) else None
                    
                    medications.append({
                        "name": name,
                        "dose": dose,
                        "frequency": frequency,
                        "raw_text": item
                    })
                else:
                    # 如果无法解析，添加原始文本
                    medications.append({
                        "name": item,
                        "raw_text": item
                    })
        
        # 如果没有找到正式的药物部分，尝试直接从文本中提取药物名
        if not medications:
            # 常见中西药物名列表
            common_meds = [
                "阿司匹林", "布洛芬", "对乙酰氨基酚", "奥美拉唑", "二甲双胍", "利尿酸", 
                "辛伐他汀", "氨氯地平", "氯沙坦", "甲状腺素", "胰岛素", "地塞米松",
                "西药", "中药", "抗生素", "消炎药", "止痛药", "退烧药", "血压药"
            ]
            
            for med in common_meds:
                if med in text:
                    # 尝试提取完整的药物信息
                    context = re.search(f"{med}[^，。,.\n]+", text)
                    if context:
                        medications.append({
                            "name": med,
                            "raw_text": context.group(0).strip()
                        })
        
        return medications
    
    def extract_patient_info(self, text: str) -> Dict[str, Any]:
        """从医疗文本中提取患者基本信息"""
        info = {}
        
        # 提取姓名
        name_pattern = r'(?:姓名|患者)[：:]\s*([^\s,，。.]+'
        name_match = re.search(name_pattern, text)
        if name_match:
            info["name"] = name_match.group(1).strip()
        
        # 提取性别
        gender_pattern = r'性别[：:]\s*(男|女)'
        gender_match = re.search(gender_pattern, text)
        if gender_match:
            info["gender"] = gender_match.group(1).strip()
        
        # 提取年龄
        age_pattern = r'年龄[：:]\s*(\d+)(?:\s*岁)?'
        age_match = re.search(age_pattern, text)
        if age_match:
            info["age"] = int(age_match.group(1).strip())
        
        # 提取患者ID
        id_pattern = r'(?:ID|病号|编号|档案号)[：:]\s*([A-Za-z0-9-]+)'
        id_match = re.search(id_pattern, text)
        if id_match:
            info["patient_id"] = id_match.group(1).strip()
        
        # 提取联系方式
        phone_pattern = r'(?:电话|手机|联系方式)[：:]\s*(\d{3,}[-\s]?\d{4,}(?:[-\s]?\d{4})?)'
        phone_match = re.search(phone_pattern, text)
        if phone_match:
            info["phone"] = phone_match.group(1).strip()
        
        # 提取就诊日期
        date_pattern = r'(?:日期|就诊日期|门诊日期)[：:]\s*(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)'
        date_match = re.search(date_pattern, text)
        if date_match:
            info["visit_date"] = date_match.group(1).strip()
        
        return info
    
    def generate_medical_question_embeddings(self, questions: List[str]) -> Dict[str, List[str]]:
        """
        为医疗问题生成语义相关的问题变体嵌入
        用于增强医疗问答系统的检索
        """
        question_variants = {}
        
        for question in questions:
            variants = []
            
            # 添加原始问题
            variants.append(question)
            
            # 提取关键词
            keywords = self.segmenter.extract_keywords(question, topK=5)
            
            # 变体1: 使用关键词构建问题
            if keywords:
                keywords_question = "关于" + "和".join(keywords[:3]) + "的信息"
                variants.append(keywords_question)
            
            # 变体2: 更改问句形式
            # 将"是什么"改为"有哪些"
            variants.append(re.sub(r'是什么', r'有哪些', question))
            # 将"如何"改为"怎么样"
            variants.append(re.sub(r'如何', r'怎么样', question))
            
            # 变体3: 添加医疗术语
            # 将"头痛"改为"头痛(头疼)"
            variants.append(re.sub(r'头痛', r'头痛(头疼)', question))
            # 将"感冒"改为"感冒(上呼吸道感染)"
            variants.append(re.sub(r'感冒', r'感冒(上呼吸道感染)', question))
            
            # 存储变体
            question_variants[question] = list(set(variants))
        
        return question_variants
    
    def classify_medical_text(self, text: str) -> Dict[str, float]:
        """
        对医疗文本进行分类
        返回各类别的置信度分数
        """
        # 医疗文本分类特征词
        category_features = {
            "症状描述": ["疼痛", "不适", "感觉", "出现", "症状", "感到"],
            "疾病信息": ["疾病", "病因", "病理", "机制", "影响", "发病率"],
            "治疗方案": ["治疗", "手术", "药物", "方案", "效果", "疗法", "用药"],
            "预防保健": ["预防", "保健", "护理", "饮食", "建议", "生活方式"],
            "检查检验": ["检查", "化验", "影像", "结果", "报告", "CT", "核磁", "B超"]
        }
        
        # 计算每个类别的分数
        category_scores = {}
        
        for category, features in category_features.items():
            score = 0
            for feature in features:
                if feature in text:
                    score += 1
            
            # 归一化分数
            category_scores[category] = score / len(features)
        
        # 确保分数总和为1
        total_score = sum(category_scores.values())
        if total_score > 0:
            for category in category_scores:
                category_scores[category] /= total_score
        
        return category_scores


# 使用示例
if __name__ == "__main__":
    # 文本标准化示例
    text = "这是一个测试文本，包含HTML标签<p>和URL http://example.com，还有emoji😊。"
    normalizer = TextNormalizer()
    normalized_text = normalizer.normalize_text(text)
    print(f"标准化文本: {normalized_text}")
    
    # 分词示例
    segmenter = TextSegmenter()
    tokens = segmenter.segment("我今天去医院看了医生，医生说我的血压有点高。")
    print(f"分词结果: {tokens}")
    
    # 医疗文本处理示例
    medical_processor = MedicalTextProcessor()
    medical_text = "患者男，45岁，血压BP: 145/95 mmHg，体温T: 37.6℃，血糖: 6.5 mmol/L。"
    extracted_values = medical_processor.extract_medical_values(medical_text)
    print(f"提取的医疗数值: {extracted_values}")
    
    abnormal_findings = medical_processor.detect_adverse_conditions(medical_text)
    print(f"检测到的异常状况: {abnormal_findings}")