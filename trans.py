import json

input_path = "/home/mry/sjs/MedQA/src/data/raw/train_datasets.jsonl"   # Huatuo 样式 jsonl
output_path = "/home/mry/sjs/MedQA/src/data/raw/clean_train_datasets.json"  # 输出：纯字符串 QA json（数组）

result = []

def squash_to_str(x):
    """
    把值拍扁成一个字符串：
    - 如果是 list，就一直取第一个，直到不是 list 为止
    - 如果最后是 str，就返回；否则返回空字符串
    """
    while isinstance(x, list):
        if not x:
            return ""
        x = x[0]
    return x if isinstance(x, str) else ""

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        questions = obj.get("questions") or []
        answers = obj.get("answers") or []

        # 没有问或没有答就跳过
        if not questions or not answers:
            continue

        # 1) 处理 questions：拍扁成字符串
        query = squash_to_str(questions)

        # 2) 处理 answers：
        #    - 如果是 list，就对每个元素拍扁成字符串再拼接
        #    - 如果是 str，直接用
        if isinstance(answers, list):
            pieces = []
            for a in answers:
                if isinstance(a, list):
                    txt = squash_to_str(a)
                else:
                    txt = a if isinstance(a, str) else ""
                if txt:
                    pieces.append(txt)
            response = "\n".join(pieces)
        else:
            response = answers if isinstance(answers, str) else ""

        # 简单过滤空字符串
        query = query.strip()
        response = response.strip()
        if not query or not response:
            continue

        result.append({
            "query": query,
            "response": response
        })

print(f"转换完成，共 {len(result)} 条样本")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"已保存到 {output_path}")
