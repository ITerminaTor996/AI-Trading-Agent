import sys
sys.stdout.reconfigure(encoding='utf-8')

class PromptTemplates:
    """
    管理所有用于LLM交互的Prompt模板。
    """

    @staticmethod
    def trading_decision_prompt(
        examples: str, 
        current_data: str, 
        risk_preference: str = "中性",
        news_text: str = "" # 新增新闻文本参数
    ) -> str:
        """
        生成用于交易决策的Prompt。

        :param examples: 历史K线数据转换而来的文本示例。
        :param current_data: 当前K线数据转换而来的文本描述。
        :param risk_preference: 用户的风险偏好（例如: '保守', '稳健', '激进', '中性'）。
        :param news_text: 最新的相关新闻文本。
        :return: 完整的Prompt字符串。
        """
        prompt_template = """
你是一个专业的金融交易员，拥有丰富的市场分析经验和严格的风险控制意识。你的任务是根据提供的市场数据、技术指标、新闻信息和用户的风险偏好，生成一个交易决策（买入/卖出/持有）。

请严格遵循以下步骤进行分析和决策：
1.  **分析技术指标**：数据中包含了关键的技术指标，请务必仔细解读：
    *   `20日均线 (MA20)`：短期趋势的判断依据。价格在均线上方通常视为看涨信号。
    *   `14日RSI (RSI14)`：相对强弱指数，衡量市场超买（通常>70）或超卖（通常<30）的状态。
    *   `MACD` 和 `MACD信号线`: 趋势跟踪动量指标。MACD线上穿信号线（金叉）是看涨信号，下穿（死叉）是看跌信号。
2.  **分析历史与当前K线**：结合历史示例，评估当前市场走势和潜在机会/风险。
3.  **分析新闻信息**：分析新闻内容对市场可能产生的情绪和基本面影响。
4.  **结合用户风险偏好**：
    *   如果用户偏好'保守'：优先考虑资金安全，只有在技术指标和基本面都非常明确时才进行操作。
    *   如果用户偏好'稳健'：在风险可控的前提下，寻求技术指标和基本面支持的合理收益。
    *   如果用户偏好'激进'：愿意承担较高风险以追求高回报，可以尝试基于动量和短期信号的操作。
    *   如果用户偏好'中性'：在风险和收益之间寻求平衡。
5.  **综合决策**：综合以上所有信息，给出最终的交易决策、详细的决策原因和你的信心指数。

历史市场示例（供参考，请勿直接复制决策）：
{examples}

当前市场数据：
{current_data}

用户风险偏好：{risk_preference}型

{news_section}请严格按照以下JSON格式输出你的决策，不要包含任何额外文字或解释：
```json
{{
  "decision": "买入/卖出/持有",
  "reason": "详细的决策原因，请结合技术指标（MA20, RSI, MACD）、K线走势、新闻信息和用户风险偏好进行分析。",
  "confidence": 0.0 # 你的决策信心指数，范围0.0到1.0，0.0表示完全不确定，1.0表示非常确定。
}}
"""
        
        news_section = f"相关新闻：\n{news_text}\n" if news_text else ""

        return prompt_template.format(
            examples=examples,
            current_data=current_data,
            risk_preference=risk_preference,
            news_section=news_section
        )

# 示例用法 (用于测试)
if __name__ == '__main__':
    # 模拟DataAgent的输出
    mock_examples = "日期：2023-01-01，开盘价：100.00，收盘价：102.00，成交量：1000。 20日均线：101.00。 14日RSI：60.50。 MACD：0.50。 MACD信号线：0.45。"
    mock_current_data = "日期：2023-01-03，开盘价：101.50，收盘价：103.00，成交量：1200。 20日均线：101.50。 14日RSI：65.00。 MACD：0.60。 MACD信号线：0.50。"
    mock_news_text = "【新闻】标题: 某公司发布利好消息\n内容: 公司宣布重大技术突破。"
    
    # 测试不同风险偏好
    prompt_conservative = PromptTemplates.trading_decision_prompt(
        examples=mock_examples,
        current_data=mock_current_data,
        risk_preference="保守",
        news_text=mock_news_text
    )
    print("--- 保守型用户Prompt ---")
    print(prompt_conservative)
    
    prompt_aggressive = PromptTemplates.trading_decision_prompt(
        examples=mock_examples,
        current_data=mock_current_data,
        risk_preference="激进",
        news_text=mock_news_text
    )
    print("\n--- 激进型用户Prompt ---")
    print(prompt_aggressive)