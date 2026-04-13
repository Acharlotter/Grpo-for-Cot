import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import numpy as np
import re
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PPO参数
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    #optimize_cuda_cache=True,
    #early_stopping=True,
    #target_kl=0.1,
    #kl_penalty="kl",
    seed=42,
    #use_score_scaling=True,
    #use_score_norm=True,
    #score_clip=None,
)

def load_sft_model(model_path, device="auto", use_4bit=False):
    # 检查是否是PEFT模型
    is_peft_model = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 设置加载参数
    load_kwargs = {
        "device_map": device,
        "trust_remote_code": True
    }
    
    if use_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        load_kwargs["bnb_4bit_use_double_quant"] = True
        load_kwargs["bnb_4bit_quant_type"] = "nf4"
    
    # 加载模型
    if is_peft_model:
        # 获取基础模型路径
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **load_kwargs
        )
        
        # 加载PEFT模型
        model = PeftModel.from_pretrained(
            base_model,
            model_path
        )
    else:
        # 直接加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )
    
    return model, tokenizer


def prepare_for_ppo(model):
    # 将模型包装为带有价值头的模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    return model


def compute_reward(response, reference=None, alpha=1.0, beta=0.01):
    # 提取答案
    answer = extract_answer(response)
    
    # 计算简洁性奖励
    conciseness_reward = -len(response) / 10000  # 归一化
    
    # 如果有参考答案，计算准确性奖励
    if reference is not None:
        accuracy_reward = 1.0 if is_answer_correct(answer, reference) else -1.0
    else:
        # 如果没有参考答案，只考虑简洁性
        accuracy_reward = 0.0
        alpha = 0.0
    
    # 计算总奖励
    reward = alpha * accuracy_reward + beta * conciseness_reward
    
    return reward

def extract_answer(response):
    # 尝试使用####分隔符提取答案
    if "####" in response:
        answer = response.split("####")[-1].strip()
        return answer
    
    # 尝试使用$\\boxed{答案}$格式提取答案
    boxed_pattern = r'\$\\boxed\{(.*?)\}\$'
    boxed_match = re.search(boxed_pattern, response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 尝试使用最后一行作为答案
    lines = response.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return response.strip()

def is_answer_correct(predicted, reference):
    # 清理和标准化答案
    predicted = clean_answer(predicted)
    reference = clean_answer(reference)
    
    # 直接匹配
    if predicted == reference:
        return True
    
    # 选择题匹配（A/B/C/D）
    if len(predicted) == 1 and predicted.upper() in "ABCD" and reference.upper() in "ABCD":
        return predicted.upper() == reference.upper()
    
    # 多选题匹配（如"A,C"或"AC"）
    if all(c.upper() in "ABCD" for c in predicted.replace(",", "")) and all(c.upper() in "ABCD" for c in reference.replace(",", "")):
        pred_choices = set(c.upper() for c in predicted if c.upper() in "ABCD")
        ref_choices = set(c.upper() for c in reference if c.upper() in "ABCD")
        return pred_choices == ref_choices
    
    # 数值匹配（允许一定的误差）
    try:
        pred_num = float(predicted)
        ref_num = float(reference)
        # 允许0.1%的相对误差
        return abs(pred_num - ref_num) / max(abs(ref_num), 1e-10) < 0.001
    except ValueError:
        pass
    
    return False

def clean_answer(answer):
    # 移除空白字符
    answer = answer.strip()
    
    # 移除常见的前缀
    prefixes = ["答案是", "答案:", "答案：", "the answer is", "answer:"]
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # 移除引号
    answer = answer.strip('"\'')
    
    return answer


def generate_responses(model, tokenizer, queries, max_new_tokens=4096, temperature=0.7, top_p=0.9):
    responses = []
    
    for query in queries:
        # 构建消息
        messages = [
            {"role": "system", "content": "请你扮演一位金融和会计领域专家，你会面临用户提出的一些问题，你要给出解决问题的思考过程和最终答案。你要首先在头脑中思考推理过程，然后向用户提供答案。最后，答案要用 $\\boxed{答案}$的形式输出。"},
            {"role": "user", "content": query}
        ]
        
        # 将消息转换为模型输入
        input_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses

def compute_rewards(responses, references=None):
    rewards = []
    
    for i, response in enumerate(responses):
        reference = references[i] if references is not None else None
        reward = compute_reward(response, reference)
        rewards.append(reward)
    
    return rewards


def rlhf_training(model, tokenizer, train_dataset, num_train_epochs=1, learning_rate=1e-5, output_dir="./output/rlhf"):
    # 准备PPO训练
    model = prepare_for_ppo(model)
    
    # 创建PPO训练器
    ppo_trainer = PPOTrainer(
        #config=ppo_config,
        model=model,
        #tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=None,
    )
    
    # 训练循环
    for epoch in range(num_train_epochs):
        logger.info(f"开始第{epoch+1}轮训练")
        
        # 遍历数据集
        for i, batch in enumerate(ppo_trainer.dataloader):
            # 获取查询
            queries = [item["question"] for item in batch]
            
            # 生成回答
            responses = generate_responses(model, tokenizer, queries)
            
            # 计算奖励
            rewards = compute_rewards(responses)
            
            # 更新模型
            stats = ppo_trainer.step(queries, responses, rewards)
            
            # 打印统计信息
            if i % 10 == 0:
                logger.info(f"Batch {i}, 平均奖励: {np.mean(rewards):.4f}, KL散度: {stats['kl']:.4f}")
        
        # 保存模型
        ppo_trainer.save_model(os.path.join(output_dir, f"checkpoint-{epoch+1}"))
    
    # 保存最终模型
    ppo_trainer.save_model(os.path.join(output_dir, "final"))
    
    return model


if __name__ == "__main__":
    # 设置模型路径和数据集路径
    model_path = "C:\\AAa_char\\tianchi-competition3\\TC3\\models--Qwen--Qwen3-4B\\snapshots\\531c80e289d6cff3a7cd8c0db8110231d23a6f7a"
    dataset_path = "C:\\AAa_char\\tianchi-competition3\\TC3\\input.json"
    
    # 加载模型和分词器
    model, tokenizer = load_sft_model(model_path, use_4bit=True)
    
    # 加载训练数据集（假设是JSON格式）
    with open(dataset_path, "r", encoding="utf-8") as jsonfile:
        train_dataset = json.load(jsonfile)

    # 开始RLHF训练
    rlhf_training(model, tokenizer, train_dataset, num_train_epochs=3, output_dir="./output/rlhf")