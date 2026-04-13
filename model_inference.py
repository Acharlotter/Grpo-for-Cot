"""
模型加载和推理模块

此模块提供了加载Qwen3-4b模型和进行推理的功能。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union
import re
import time
import os

class ModelInference:
    """
    模型推理类，用于加载模型和进行推理。
    """
    
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-4B"):
        """
        初始化模型推理类。
        
        参数:
            model_name_or_path: 模型名称或路径，默认为"Qwen/Qwen3-4B-Chat"
        """
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self) -> None:
        """
        加载模型和分词器。
        """
        print(f"正在加载模型 {self.model_name_or_path}...")
        start_time = time.time()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        end_time = time.time()
        print(f"模型加载完成，耗时 {end_time - start_time:.2f} 秒")
        print(f"模型运行设备: {self.device}")
        
    def generate(self, 
                prompt: Dict[str, str], 
                max_new_tokens: int = 4096, 
                temperature: float = 0.7,
                top_p: float = 0.9,
                num_return_sequences: int = 1) -> List[str]:
        """
        生成回答。
        
        参数:
            prompt: 包含system和user字段的字典
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            num_return_sequences: 返回序列数量
            
        返回:
            生成的回答列表
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # 构建消息
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]}
        ]
        
        # 将消息转换为模型输入
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 解码输出
        responses = []
        for output in outputs:
            response = self.tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def extract_answer(self, response: str) -> str:
        """
        从回答中提取最终答案。
        
        参数:
            response: 模型生成的回答
            
        返回:
            提取的答案
        """
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
    
    def batch_inference(self, 
                       prompts: List[Dict[str, str]], 
                       max_new_tokens: int = 4096,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       num_samples: int = 5) -> List[Tuple[int, int, str]]:
        """
        批量推理。
        
        参数:
            prompts: prompt列表
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            num_samples: 每个问题生成的样本数量
            
        返回:
            结果列表，每个元素为(id, sample_id, output)的元组
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"正在处理问题 {i+1}/{len(prompts)}...")
            
            for j in range(num_samples):
                responses = self.generate(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1
                )
                
                response = responses[0]
                results.append((i, j, response))
                
                print(f"  样本 {j+1}/{num_samples} 完成，长度: {len(response)}")
        
        return results
    
    def save_model(self, save_path: str) -> None:
        """
        保存模型。
        
        参数:
            save_path: 保存路径
        """
        if self.model is None:
            print("模型未加载，无法保存")
            return
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(save_path)
        
        # 保存分词器
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        
        print(f"模型已保存到 {save_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化模型推理类
    inference = ModelInference()
    
    # 加载模型
    inference.load_model()
    
    # 准备prompt
    prompt = {
        "system": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Return the answer at the end of the response after a separator ####.",
        "user": "请你扮演一位金融和会计领域专家，你会面临用户提出的一些问题，你要给出解决问题的思考过程和最终答案。你要首先在头脑中思考推理过程，然后向用户提供答案。最后，答案要用 $\\boxed{答案}$的形式输出。\n问题：\n关于财产清查结果的会计处理，下列说法中错误的是（　　）。\nA.财产清查产生的损溢，企业应于期末前查明原因，并根据企业的管理权限，经股东会或董事会，或经理（厂长）会议或类似机构批准后，在期末结账前处理完毕\nB.财产清查产生的损溢，如果在期末结账前尚未经批准，在对外提供财务报表时，先按相关规定进行处理，并在附注中作出说明，其后批准处理的金额与已处理的金额不一致的，调整财务报表相关项目的当期数\nC.对于财产清查中发现的问题，应核实情况，调查分析产生的原因\nD.根据'清查结果报告表''盘点报告表'等数据资料，填制记账凭证，记入有关账簿，使账簿记录与实际盘存数相符"
    }
    
    # 生成回答
    responses = inference.generate(prompt)
    
    # 打印回答
    print(f"生成的回答:\n{responses[0]}")
    
    # 提取答案
    answer = inference.extract_answer(responses[0])
    print(f"提取的答案: {answer}")

