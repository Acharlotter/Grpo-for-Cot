# train_grpo.py
import tempfile
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import json
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model,PeftModel
import re
import os, torch
import os
import swanlab
import numpy as np
from swanlab.integration.transformers import SwanLabCallback
# local_rank = int(os.environ["LOCAL_RANK"])
# print(f"rank {local_rank} -> physical GPU {local_rank} "
#       f"({torch.cuda.get_device_name()})")

import math

# 斜率字典：左闭右开区间，斜率逐段变大
SLOPE = {
    (0, 300):    0.0005,   # 0-300 字：轻罚
    (300, 600):  0.0008,   # 300-600：中罚
    (600, 900):  0.0015,   # 600-900：重罚
    (900, float('inf')): 0.0025  # >900：超重罚
}

def linear_segment_penalty(length, slope_dict):
    """逐段线性累加惩罚"""
    penalty = 0.0
    for (low, high), k in slope_dict.items():
        if length <= low:
            continue
        seg_len = min(length, high) - low
        penalty += seg_len * k
    penalty=max(penalty, -2.0)
    return penalty


data = []
with open("/hy-tmp/RL/code/data/dianjin_data/qwen_RL.jsonl", "r", encoding="utf-8") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            data.append(json.loads(ln))


# Define the reward function, which rewards completions that are close to 20 characters
def reward_fn(completions, **kwargs):
    refs = kwargs.get("reference", [None] * len(completions))
    outs = [compute_reward(c, r) for c, r in zip(completions, refs)]
    rewards, ok_list, lens = zip(*outs)

    # ---- 新增：每步上传一次 ----
    swanlab.log({
        "acc/answer":     np.mean(ok_list),
        "think_len/mean": np.mean(lens),
    })
    # ----------------------------

    return list(rewards)   # GRPO 只拿这个

def clean_answer(ans):
    ans = ans.strip()
    for pre in ["答案是", "答案:", "答案：", "the answer is", "answer:"]:
        if ans.lower().startswith(pre.lower()):
            ans = ans[len(pre):].strip()
    return ans.strip('"\' ')

def extract_answer(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    text=text[0]
    text=text['content']
    m = re.search(r'\\boxed\{{1,2}(.+?)\}{1,2}', text)
    if m:
        return m.group(1).strip()
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else text.strip()

def is_correct(pred, ref):
    pred, ref = clean_answer(pred), clean_answer(ref)
    if pred == ref:
        return True
    # A/B/C/D
    if len(pred) == len(ref) == 1 and pred.upper() in "ABCD" and ref.upper() in "ABCD":
        return pred.upper() == ref.upper()
    # 多选
    if all(c.upper() in "ABCD" for c in pred.replace(",", "")) and \
       all(c.upper() in "ABCD" for c in ref.replace(",", "")):
        return set(pred.upper()) == set(ref.upper())
    # 数值
    try:
        return abs(float(pred) - float(ref)) / max(abs(float(ref)), 1e-10) < 0.001
    except ValueError:
        return False

def compute_reward(response, reference=None, alpha=1.0, beta=1.0):

    # 1. 正确性
    answer = extract_answer(response)
    acc = 2.1 if is_correct(answer, reference) else -2.1 if reference else 0.0

    # 2. 思考链长度
    content = response[0]['content']
    think_match = re.search(r'<think>(.*?)</think>', content, flags=re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    cur_len = len(think_text)
    #concise = -len(think_text) / 1000
    
    # 3. 长度动态惩罚
    concise = -linear_segment_penalty(cur_len, SLOPE) * beta


    # 3. 返回三元组：reward 给 GRPO，is_ok & len 给日志
    is_ok = float(is_correct(answer, reference))
    return alpha * acc + beta * concise, is_ok, len(think_text)


training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.05,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16=True,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory,对每一条训练数据生成的候选结果数量
    max_prompt_length = 2048,
    max_completion_length = 2048,
    max_steps = 1000,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "/hy-tmp/RL_outputs/outputs2",
)

base = AutoModelForCausalLM.from_pretrained(
    '/hy-tmp/RL_model/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a',
    #device_map="auto",     #多卡不能'auto'
)
tokenizer = AutoTokenizer.from_pretrained('/hy-tmp/RL_model/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a')

peftmodel = PeftModel.from_pretrained(base, "/hy-tmp/RL/code/lora_model/dianjin24000/checkpoint-2500").merge_and_unload()


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# 由于 transformers 必须重新 load，用临时目录即可
# with tempfile.TemporaryDirectory() as tmp:
#     peftmodel.save_pretrained(tmp)               # 保存合并权重
#     model_4bit = AutoModelForCausalLM.from_pretrained(
#         tmp,
#         quantization_config=bnb_config,       # 重新 4-bit 加载
#         #torch_dtype=torch.bfloat16,
#         device_map="auto",
#     )
swanlab_callback = SwanLabCallback(
    project="Qwen3-4B-grpo",
    experiment_name="Qwen3-4B",
    description="使用通义千问Qwen3-4B模型在dianjing金融数据集上做强化学习grpo。",
    config={
        "model": "Qwen/Qwen3-4B",
        "dataset": "/hy-tmp/RL/code/data/dianjin_data/qwen_RL.jsonl",
        "train_data_number": len(data),
        "lora_rank": 32,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    }
)

model = get_peft_model(peftmodel, lora_config)
model.print_trainable_parameters()


trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=data,
    callbacks=[swanlab_callback],
)

trainer.train()