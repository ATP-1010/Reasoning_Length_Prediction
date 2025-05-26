from datasets import load_dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm  # 加载进度条
import re
from itertools import islice

def extract_think_and_rest(text):
    # 提取所有 <think>...</think> 中的内容
    think_blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    
    # 找到最后一个 </think> 的位置
    last_think_end = 0
    for match in re.finditer(r"</think>", text):
        last_think_end = match.end()

    # 获取 </think> 后的剩余文本
    rest_text = text[last_think_end:].strip() if last_think_end else text.strip()

    return think_blocks, rest_text

# 加载 OpenR1-Math-220k 数据集
# dataset = load_dataset("open-r1/OpenR1-Math-220k", split="train")
# dataset = load_dataset("open-r1/OpenThoughts-114k-Code_decontaminated", split="train")
# dataset = load_dataset("FreedomIntelligence/Medical-R1-Distill-Data", split="train")
dataset = load_dataset("glaiveai/reasoning-v1-20m", split="train")

# 选择 tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324")

# 构造新数据
records = []

# 用 tqdm 包装数据集，加进度条
for item in tqdm(islice(dataset, 200000), desc="Processing samples"):
    problem = item.get("question", "").strip()
    solution = item.get("response", "").strip()
    reasoning, solution = extract_think_and_rest(solution)
    if len(reasoning) == 0:
        continue

    input_token_count = len(tokenizer.tokenize(problem))
    output_token_count = len(tokenizer.tokenize(solution))

    generation_list = item.get("reasoning (reasoning_content)", "")
    generation = generation_list.strip()

    # 拼接 prompt（input）
    prompt = (
        f"Problem: {problem}\n"
        f"Answer: {solution}\n"
        f"The problem has {input_token_count} tokens, and the answer has {output_token_count} tokens."
    )

    # 统计 reasoning 的 token 数
    reasoning_token_count = len(tokenizer.tokenize(reasoning[0]))

    response = f"{reasoning_token_count}."

    # 保存样本
    records.append({
        "prompt": prompt,
        "response": response
    })

# 保存为 JSON 文件
with open("general_dataset.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
