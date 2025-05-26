import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_SDP_BACKEND"] = "math" 
os.environ["PYTORCH_SDP_ENABLE_FLASH_ATTENTION"] = "0"
os.environ["PYTORCH_SDP_ENABLE_MEM_EFFICIENT"]   = "0"
os.environ["PYTORCH_SDP_ENABLE_math"]            = "1"   # 启用回退

import json
import torch
import torch.nn as nn
import re
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    TrainerCallback, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ==== 设置路径 ====
DATA_FILE = "./datasets/openr1_reasoning_token_dataset.json"
BASE_MODEL = "Qwen/Qwen2.5-3B"
OUTPUT_DIR = "./qwen3b-lora-90000-classification-epoch5"

# ==== 构建分类桶 ====
def bin_token_count(value):
    if value <= 2000:
        return 0
    elif value <= 4000:
        return 1
    elif value <= 6000:
        return 2
    elif value <= 10000:
        return 3
    else:
        return 4

NUM_CLASSES = 5  # 对应 5 个类别

# ==== 加载数据 ====
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==== 初始化 tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ==== 输入构造 + 分类标签 ====
def tokenize(example):
    prompt = example["prompt"]
    response = example["response"]
    
    response_token_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    n_prompt = len(prompt_token_ids)
    n_response = len(response_token_ids)

    context = (
        f"System note: The input has {n_prompt} tokens and the output has {n_response} tokens.\n"
        "You are an AI reasoning analyzer. Given a math problem and its final answer, "
        "predict the reasoning token bucket (e.g., 0-500, 500-1000...)."
    )

    full_text = context + "\n" + prompt
    inputs = tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)

    # true_token_count = int(response)
    true_token_count = int(float(response))
    class_label = bin_token_count(true_token_count)

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "classification_label": class_label
    }

# ==== 模型结构：添加分类头 ====
class CausalLMWithClassifier(nn.Module):
    def __init__(self, base_model, num_classes=NUM_CLASSES):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(self.base.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, classification_label=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled = outputs.hidden_states[-1][:, -1, :]  # 最后一个 token 的 hidden state
        class_logits = self.classifier(pooled)

        loss = None
        if classification_label is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(class_logits, classification_label)

        return {"logits": class_logits, "loss": loss}

# ==== 自动评估 callback：分类准确率 ====
class EvalCallback(TrainerCallback):
    def __init__(self, tokenizer, data_path):
        self.tokenizer = tokenizer
        self.data_path = data_path

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print("\n--- Running Evaluation ---")

        model.eval()
        device = model.base.device if hasattr(model, "base") else model.device

        with open(self.data_path, 'r', encoding='utf-8') as file:
            eval_data = json.load(file)

        correct = 0
        total = 0

        for item in tqdm(eval_data[50000:50200]):
            prompt = item["prompt"]
            response = item["response"]
            true_class = bin_token_count(int(float(response)))

            n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            n_response = len(tokenizer(response, add_special_tokens=False)["input_ids"])

            context = (
                f"System note: The input has {n_prompt} tokens and the output has {n_response} tokens.\n"
                "You are an AI reasoning analyzer. Given a math problem and its final answer, "
                "predict the reasoning token bucket (e.g., 0-500, 500-1000...)."
            )
            full_prompt = context + "\n" + prompt
            inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                class_logits = model(**inputs)["logits"]
                pred_class = class_logits.argmax(dim=-1).item()

            correct += int(pred_class == true_class)
            total += 1

        accuracy = correct / total
        print(f"✅ Evaluation done. Accuracy: {accuracy:.4f}")

# ==== 训练器重写：只训练分类任务 ====
class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("classification_label")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            classification_label=labels
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# ==== 模型加载并注入 LoRA ====
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
base_model = prepare_model_for_kbit_training(base_model)
base_model = get_peft_model(base_model, LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "gate_proj", "up_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
))

model = CausalLMWithClassifier(base_model)

# ==== 处理数据集 ====
raw_data = load_data(DATA_FILE)[:90000]
dataset = Dataset.from_list(raw_data).map(tokenize, remove_columns=["prompt", "response"])

# ==== 训练参数 ====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=5000,
    save_strategy="no",
    report_to="none",
)

# ==== 启动训练 ====
trainer = ClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    callbacks=[EvalCallback(tokenizer=tokenizer, data_path=DATA_FILE)]
)
trainer.train()

# ==== 保存最终模型 ====
model.base.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# 保存分类头权重
torch.save(model.classifier.state_dict(), os.path.join(OUTPUT_DIR, "classifier_head.pt"))

# （可选）完整保存整个模型（包含分类头和base）
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "entire_model.pt"))
