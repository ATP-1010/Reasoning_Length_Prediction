# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# import json
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm

# # ==== 配置 ====
# model_name = "qwen3b-lora-finetuned-50000-2"
# # model_name = "Qwen/Qwen2.5-3b"
# data_path = "./datasets/openr1_reasoning_token_dataset.json"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ==== 加载模型和tokenizer ====
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3b")
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# # ==== 加载数据 ====
# with open(data_path, 'r', encoding='utf-8') as f:
#     dataset = json.load(f)

# # ==== 提取数字函数 ====
# def extract_number(text):
#     match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
#     return float(match[0]) if match else None

# # ==== 初始化统计 ====
# errors = []
# accurate_count = 0
# total_count = 0

# # ==== 推理与评估 ====
# for item in tqdm(dataset[-100:]):
#     prompt = (
#         "You are an AI reasoning analyzer. Given a math problem and its final answer, "
#         "estimate how many tokens were used in the detailed reasoning process that led to the answer."
#         + item['prompt'] +
#         "The approximate number of tokens in the reasoning process is: "
#     )
#     true_number = extract_number(item['response'])
#     if true_number is None or true_number == 0:
#         continue

#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     found_accurate = False
#     prediction_errors = []

#     for _ in range(5):  # 5 次采样生成
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=30,
#             do_sample=True,
#             temperature=1.0,
#             top_p=0.95,
#             pad_token_id=tokenizer.eos_token_id
#         )

#         output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#         pred_number = extract_number(output_text)

#         print(f"\n[Generated] {output_text.strip()}")

#         if pred_number is not None:
#             error_ratio = abs(pred_number - true_number) / abs(true_number)
#             if error_ratio <= 0.5:
#                 found_accurate = True
#             else:
#                 prediction_errors.append(abs(pred_number - true_number))

#     print(f"One of the predictions accurate: {found_accurate} | True: {true_number}")

#     total_count += 1
#     if found_accurate:
#         accurate_count += 1
#     else:
#         errors.extend(prediction_errors if prediction_errors else [float('inf')])

# # ==== 结果统计 ====
# if total_count > 0:
#     mean_error = sum(errors) / len(errors) if errors else 0.0
#     accuracy = accurate_count / total_count
#     print(f"\n[Repeat-5 Any-Correct] Average numeric error (on incorrect cases): {mean_error:.4f}")
#     print(f"[Repeat-5 Any-Correct] Accuracy with at least one pass ≤ 25% error: {accuracy * 100:.2f}%")
# else:
#     print("No valid numeric predictions found.")


# --------------------------------------------------------------------------------------------------------------


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ==== 配置 ====
model_name = "qwen3b-lora-finetuned-50000-2-withlength/epoch1"
data_path = "./datasets/openr1_reasoning_token_dataset.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== 加载模型和tokenizer ====
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3b")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# ==== 加载数据 ====
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# ==== 提取数字函数 ====
def extract_number(text):
    match = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return float(match[0]) if match else None

# ==== 初始化统计 ====
errors = []
error_ratios = []
accurate_count = 0
total_count = 0

# ==== 推理与评估 ====
for item in tqdm(dataset[:100]):
    prompt = (
        "You are an AI reasoning analyzer. Given a math problem and the model output together with their token length, estimate how many tokens were used in the detailed reasoning process that led to the answer:\n"
        + item['prompt'] +
        "The approximate number of tokens in the reasoning process is: "
    )
    true_number = extract_number(item['response'])
    if true_number is None or true_number == 0:
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # === greedy search ===
    output_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    pred_number = extract_number(output_text)

    print(f"\n[Generated] {output_text.strip()} | True: {true_number}")

    total_count += 1
    if pred_number is not None:
        abs_error = abs(pred_number - true_number)
        error_ratio = abs_error / abs(true_number)

        errors.append(abs_error)
        error_ratios.append(error_ratio)

        if error_ratio <= 0.25:
            accurate_count += 1
    else:
        errors.append(float('inf'))
        error_ratios.append(float('inf'))

# ==== 结果统计 ====
if total_count > 0:
    mean_error = sum(errors) / len(errors)
    mean_ratio = sum(error_ratios) / len(error_ratios)
    accuracy = accurate_count / total_count
    print(f"\n[Greedy Search] Average numeric error (all cases): {mean_error:.4f}")
    print(f"[Greedy Search] Average relative error (all cases): {mean_ratio * 100:.2f}%")
    print(f"[Greedy Search] Accuracy within 25% error margin: {accuracy * 100:.2f}%")
else:
    print("No valid numeric predictions found.")
