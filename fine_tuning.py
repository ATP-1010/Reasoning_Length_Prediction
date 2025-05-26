import os
import json
import re
import torch
import gc
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# ==== 配置路径 ====
DATA_FILE = "./datasets/general_dataset.json"
BASE_MODEL = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./qwen3b-lora-finetuned-190k-general"

# ==== 加载数据 ====
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==== 数据预处理 ====
def tokenize(example):
    prompt = example["prompt"]
    response = example["response"]

    instruction = "You are an AI reasoning analyzer. Given a math problem and the model output together with their token length, estimate how many tokens were used in the detailed reasoning process that led to the answer:\n"
    response_prefix = "\nThe approximate number of tokens in the reasoning process is: "
    full_text = instruction + prompt + response_prefix + response

    inputs = tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)
    input_ids = inputs["input_ids"]

    prefix_ids = tokenizer(instruction + prompt + response_prefix, add_special_tokens=False)["input_ids"]
    start_index = len(prefix_ids)

    labels = [-100] * len(input_ids)
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    for j, token_id in enumerate(response_ids):
        if start_index + j < len(labels):
            labels[start_index + j] = token_id

    inputs["labels"] = labels
    return inputs

# ==== 初始化 tokenizer 和 model ====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, torch_dtype=torch.float16)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
))

model.print_trainable_parameters()  # 查看是否只训练了LoRA层


# ==== 加载数据集 ====
raw_data = load_data(DATA_FILE)[:190000]
dataset = Dataset.from_list(raw_data).map(tokenize, remove_columns=["prompt", "response"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=data_collator)

# ==== Accelerate 初始化 ====
accelerator = Accelerator()
device = accelerator.device
model, dataloader, optimizer = accelerator.prepare(
    model,
    dataloader,
    torch.optim.AdamW(model.parameters(), lr=1e-5)
)

# ==== 训练循环（无评估） ====
epochs = 2
model.train()
for epoch in range(epochs):
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    # 每轮保存模型
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_path = os.path.join(OUTPUT_DIR, f"epoch{epoch+1}")
    unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)
    tokenizer.save_pretrained(save_path)

    # 每轮之后释放显存
    torch.cuda.empty_cache()
    gc.collect()

# ==== 最终保存模型 ====
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(OUTPUT_DIR, save_function=accelerator.save)
tokenizer.save_pretrained(OUTPUT_DIR)
