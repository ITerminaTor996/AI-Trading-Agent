import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
from openai import OpenAI # 导入OpenAI库，用于调用DeepSeek API
from .base_agent import BaseAgent

class DecisionAgent(BaseAgent):
    """
    负责与大型语言模型交互，根据Prompt生成交易决策。
    """
    def __init__(self, api_key: str, model_name: str, base_url: str):
        """
        初始化DecisionAgent。

        :param api_key: 用于API调用的密钥。
        :param model_name: 要使用的模型名称。
        :param base_url: API的基础URL。
        """
        super().__init__(name="AI决策Agent")
        if not api_key:
            raise ValueError("错误：API密钥未提供。")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        print(f"[{self.name}] 使用模型: {model_name}, Base URL: {base_url}")

    def generate_decision(self, prompt: str) -> dict:
        """
        根据给定的Prompt，调用LLM生成交易决策。

        :param prompt: 包含市场数据、风险偏好和指令的完整Prompt字符串。
        :return: LLM返回的JSON格式决策（字典）。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # 确保确定性输出
                max_tokens=500 # 适当增加token限制以确保完整输出
            )
            
            # 解析LLM的响应文本
            response_text = response.choices[0].message.content.strip()
            
            # 尝试解析JSON，处理LLM可能在JSON前后添加的```json或```
            if response_text.startswith('```json') and response_text.endswith('```'):
                response_text = response_text[len('```json'):-len('```')].strip()
            
            decision = json.loads(response_text)
            return decision
        except Exception as e:
            print(f"[{self.name}] 调用LLM或解析响应时发生错误: {e}")
            response_text = locals().get('response_text', 'N/A')
            print(f"[{self.name}] LLM原始响应: {response_text}")
            return {"decision": "错误", "reason": f"LLM调用失败或响应解析错误: {e}", "confidence": 0.0}

# 示例用法 (用于测试)
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import app_config

    mock_prompt = """
你是一个专业的金融交易员。根据以下数据做出交易决策：

历史市场示例：
日期：2023-01-01，开盘价：100.00，收盘价：102.00，成交量：1000。

当前市场数据：
日期：2023-01-02，开盘价：102.00，收盘价：103.00，成交量：1200。

用户风险偏好：中性型

请严格按照以下JSON格式输出你的决策，不要包含任何额外文字或解释：
```json
{
  "decision": "买入/卖出/持有",
  "reason": "详细的决策原因",
  "confidence": 0.0
}
"""

    try:
        decision_agent = DecisionAgent(
            api_key=app_config.DEEPSEEK_API_KEY,
            model_name=app_config.DEEPSEEK_MODEL_NAME,
            base_url=app_config.DEEPSEEK_BASE_URL
        )
        
        print(f"\n--- {decision_agent.name} 模拟LLM调用测试 ---")
        if not app_config.DEEPSEEK_API_KEY:
            print("警告: DEEPSEEK_API_KEY 未在 app_config.py 或 .env 文件中设置，测试将失败。")
        
        decision = decision_agent.generate_decision(mock_prompt)
        print("LLM决策:", decision)

    except ValueError as ve:
        print(ve)
    except ImportError:
        print("错误：无法导入app_config模块。请确保 `app_config.py` 文件在项目根目录，")
        print("并且你正在从项目根目录运行此脚本，或者已经将项目根目录添加到了PYTHONPATH。")
    except Exception as e:
        print(f"发生错误: {e}")
