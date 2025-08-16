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
        portfolio_status: str,
        news_text: str,
        financial_summary_text: str, # 新增财务摘要参数
        risk_preference: str = "中性"
    ) -> str:
        """
        生成用于交易决策的Prompt。

        :param examples: 历史K线数据转换而来的文本示例。
        :param current_data: 当前K线数据转换而来的文本描述。
        :param portfolio_status: 当前的持仓状态文本。
        :param news_text: 最新的相关新闻文本。
        :param financial_summary_text: 公司的财务摘要文本。
        :param risk_preference: 用户的风险偏好。
        :return: 完整的Prompt字符串。
        """
        
        json_format_example = """```json
{
  "decision": "买入/卖出/持有",
  "reason": "详细的决策原因，请结合你的当前持仓、基本面、所有技术指标、K线走势、新闻信息和用户风险偏好进行分析。",
  "confidence": 0.0,
  "trade_percent": 0.0
}
```"""

        news_section = f"相关新闻：\n{news_text}\n" if news_text and news_text.strip() else ""
        financial_section = f"\n基本面财务摘要：\n{financial_summary_text}\n" if financial_summary_text and financial_summary_text.strip() else ""

        prompt_template = """
你是一位顶级的金融分析师，同时具备深厚的技术分析功底和严谨的基本面分析能力。你的任务是综合运用所有信息，为一笔交易生成包含具体交易规模的决策。

请严格遵循以下步骤进行分析和决策：
1.  **评估当前持仓**：首先审视你的持仓状态，这是你决策的出发点。
2.  **分析基本面**: 查看公司的财务摘要，评估其估值（如PE）和盈利能力（如EPS），判断其长期投资价值。
3.  **分析技术指标**：解读所有技术指标，判断短期市场趋势和动量。
4.  **分析新闻信息**：分析新闻内容对市场可能产生的情绪和短期影响。
5.  **结合用户风险偏好**：根据风险偏好调整你的决策和交易规模。
6.  **决定交易规模 (trade_percent)**: 这是关键一步。根据你对基本面、技术面和新闻面的综合信心来决定交易规模。
    *   如果决策是 **买入**: `trade_percent` 表示你建议使用多少比例的**可用现金**来建仓 (0.0到1.0之间)。
    *   如果决策是 **卖出**: `trade_percent` 表示你建议卖出多少比例的**现有持仓** (0.0到1.0之间)。
    *   如果决策是 **持有**: `trade_percent` 应为 0.0。
7.  **综合决策**：综合以上所有信息，给出最终的交易决策、详细的决策原因、你的信心指数以及建议的交易规模百分比。

当前持仓状态：
{portfolio_status}
{financial_section}
{news_section}
历史市场示例（供参考，请勿直接复制决策）：
{examples}

当前市场数据：
{current_data}

用户风险偏好：{risk_preference}型

请严格按照以下JSON格式输出你的决策，不要包含任何额外文字或解释：
{json_format_example}
"""
        # 使用.format()来填充模板
        return prompt_template.format(
            portfolio_status=portfolio_status,
            financial_section=financial_section,
            news_section=news_section,
            examples=examples,
            current_data=current_data,
            risk_preference=risk_preference,
            json_format_example=json_format_example
        )

# 示例用法 (用于测试)
if __name__ == '__main__':
    mock_examples = "日期：2023-01-01，开盘价：100.00，收盘价：102.00..."
    mock_current_data = "日期：2023-01-03，开盘价：101.50，收盘价：103.00..."
    mock_news_text = "【新闻】标题: 某公司发布利好消息..."
    mock_portfolio_status = "当前无持仓。"
    mock_financial_summary = "- 公司名称: Apple Inc.\n- 市值: 2.7T\n- 市盈率(PE): 28.50"

    prompt = PromptTemplates.trading_decision_prompt(
        examples=mock_examples,
        current_data=mock_current_data,
        portfolio_status=mock_portfolio_status,
        news_text=mock_news_text,
        financial_summary_text=mock_financial_summary,
        risk_preference="稳健"
    )
    print("--- 完整Prompt示例 ---")
    print(prompt)