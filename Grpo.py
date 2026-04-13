"""
GRPO 训练脚本（金融 / 会计问答：正确 + 简洁）
"""
import os
import torch
import json
import re
import logging
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead

# ---------------- 日志 ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- GRPO 超参 ---------------- #
grpo_config = GRPOConfig(
    # use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,

    # bf16 = is_bfloat16_supported(),
    # fp16 = not is_bfloat16_supported(),
    bf16=True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory,对每一条训练数据生成的候选结果数量
    max_prompt_length = 1024,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 750,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)

# ---------------- 模型加载 ---------------- #
def load_sft_model(model_path, lora_path, device="cuda:0", use_4bit=False):
    is_peft = os.path.exists(os.path.join(lora_path, "adapter_config.json"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    load_kwargs = {"device_map": device, "trust_remote_code": True}
    if use_4bit:
        load_kwargs.update(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if is_peft:
        base = AutoModelForCausalLM.from_pretrained(model_path)
        model = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
        #model = model.merge_and_unload() 
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    return model, tokenizer

# ---------------- 奖励/工具函数 ---------------- #
# def prepare_for_grpo(model):
#     # 将模型包装为带有价值头的模型，GRPO不需要价值头！！！！
#     model.config.return_dict = True
#     model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
#     if not hasattr(model, "warnings_issued"):
#         model.warnings_issued = {}
#     if not hasattr(model, "add_model_tags"):
#         model.add_model_tags = model.pretrained_model.add_model_tags
#     model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
    
#     return model

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

def compute_reward(response, reference=None, alpha=1.0, beta=0.01):
    answer = extract_answer(response)
    concise = -len(response) / 10000
    acc = 1.0 if is_correct(answer, reference) else -1.0 if reference else 0.0
    return alpha * acc + beta * concise

# ---------------- 生成回答 ---------------- #
# def generate_responses(model, tokenizer, queries, max_new=512, temp=0.7, top_p=0.9):
#     responses = []
#     sys_msg = "请你扮演一位金融和会计领域专家，你要给出思考过程和最终答案，答案用 $\\boxed{答案}$ 输出。"
#     for q in queries:
#         msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}]
#         prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#         with torch.no_grad():
#             outs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new,
#                 do_sample=True,
#                 temperature=temp,
#                 top_p=top_p,
#                 pad_token_id=tokenizer.pad_token_id
#             )
#         resp = tokenizer.decode(outs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#         responses.append(resp)
#     return responses

# ---------------- 奖励适配器（GRPO 要求 list[float]） ---------------- #
def reward_fn(prompts, completions, **kwargs):
    refs = kwargs.get("reference", [None] * len(completions))
    reward=[compute_reward(c, r) for c, r in zip(completions, refs)]
    return reward

# ---------------- GRPO 训练入口 ---------------- #
def grpo_training(model_path,
                  dataset_path,
                  lora_path,
                  num_epochs=1,
                  use_4bit=False,
                  output_dir="./output/grpo"):
    model, tokenizer = load_sft_model(model_path, lora_path=lora_path, use_4bit=use_4bit)

    # 读数据： [{"prompt": "...", "reference": "..."}, ...]
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                data.append(json.loads(ln))

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=data,
        reward_funcs=reward_fn,
    )

    logger.info("开始 GRPO 训练")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"训练完成，模型保存在 {output_dir}")
    return trainer.model

# ---------------- 命令行入口 ---------------- #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/hy-tmp/RL_model/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a", help="模型名称或路径")
    parser.add_argument("--dataset_path", type=str, default="/hy-tmp/RL/code/data/dianjin_data/qwen_RL.jsonl", help="数据集路径")
    parser.add_argument("--lora_path", type=str, default="/hy-tmp/RL/code/sft_final/dianjin", help="LoRA模型路径")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--output_dir", default="./output/grpo")
    args = parser.parse_args()

    grpo_training(
        model_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        lora_path=args.lora_path,
        num_epochs=args.num_epochs,
        use_4bit=args.use_4bit,
        output_dir=args.output_dir,
    )