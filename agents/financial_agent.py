import sys
sys.stdout.reconfigure(encoding='utf-8')

import yfinance as yf
import asyncio
from .base_agent import BaseAgent

class FinancialAgent(BaseAgent):
    """
    负责获取和处理公司的基本面财务数据。
    """
    def __init__(self):
        super().__init__(name="基本面分析Agent")

    def _get_financial_summary_sync(self, ticker: str) -> dict:
        """
        同步方法：获取指定股票的财务摘要信息。
        """
        try:
            print(f"[{self.name}] 正在从yfinance获取 {ticker} 的财务摘要...")
            stock = yf.Ticker(ticker)
            # .info返回一个包含大量信息的字典
            return stock.info
        except Exception as e:
            print(f"[{self.name}] 错误：获取 {ticker} 的财务摘要时发生异常: {e}")
            return {}

    async def get_financial_summary(self, ticker: str) -> dict:
        """
        异步方法：在独立的线程中运行同步的数据获取方法。
        """
        return await asyncio.to_thread(self._get_financial_summary_sync, ticker)

    def summary_to_text(self, summary: dict) -> str:
        """
        将财务摘要字典转换为LLM可读的自然语言描述。
        """
        if not summary:
            return "没有可用的财务摘要信息。"

        # 挑选一些关键指标进行展示
        key_metrics = {
            "longName": "公司名称",
            "marketCap": "市值",
            "trailingPE": "市盈率(PE)",
            "trailingEps": "每股收益(EPS)",
            "dividendYield": "股息率",
            "beta": "贝塔系数",
            "dayHigh": "当日最高价",
            "dayLow": "当日最低价",
            "fiftyTwoWeekHigh": "52周最高价",
            "fiftyTwoWeekLow": "52周最低价",
            "volume": "成交量"
        }

        text_parts = ["基本面财务摘要："]
        for key, label in key_metrics.items():
            value = summary.get(key)
            if value is not None:
                if isinstance(value, (int, float)):
                    # 对市值等大数字进行格式化
                    if "市值" in label and value > 1_000_000_000:
                        value_str = f"{value / 1_000_000_000:.2f}B" # 十亿
                    elif isinstance(value, float):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = f"{value:,}"
                else:
                    value_str = str(value)
                text_parts.append(f"- {label}: {value_str}")
        
        return "\n".join(text_parts)

# 示例用法 (用于测试)
async def main_test():
    financial_agent = FinancialAgent()
    summary_data = await financial_agent.get_financial_summary("AAPL")
    
    if summary_data:
        summary_text = financial_agent.summary_to_text(summary_data)
        print("\n--- AAPL财务摘要转文本示例 ---")
        print(summary_text)
    else:
        print("未能获取财务摘要。")

if __name__ == '__main__':
    asyncio.run(main_test())