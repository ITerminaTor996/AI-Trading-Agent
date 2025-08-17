import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import yfinance as yf
import ta
import asyncio
from .base_agent import BaseAgent

class DataAgent(BaseAgent):
    """
    负责处理和转换金融数据，为LLM提供文本格式的输入。
    """
    def __init__(self, ticker: str):
        """
        初始化DataAgent。

        :param ticker: 股票代码 (例如: 'AAPL', 'MSFT')。
        """
        super().__init__(name="数据处理Agent")
        self.ticker = ticker.upper()
        self.data = None

    def _load_data_sync(self):
        """
        同步方法：从yfinance加载指定股票代码的K线数据，并计算技术指标。
        """
        try:
            print(f"[{self.name}] 正在从yfinance加载 {self.ticker} 的数据...")
            # 增加数据周期到100天，为技术指标计算提供更充足的数据
            df = yf.download(self.ticker, period="100d", auto_adjust=True)
            if df.empty:
                print(f"[{self.name}] 错误：未能从yfinance加载 {self.ticker} 的数据，DataFrame为空。")
                self.data = pd.DataFrame()
                return self.data

            # --- 核心修正：检查并压平yfinance可能返回的多级索引列名 ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.reset_index(inplace=True)
            df.rename(columns={
                'Date': 'Date', 'Open': 'Open', 'High': 'High',
                'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
            }, inplace=True)

            # --- 计算技术指标 ---
            print(f"[{self.name}] 正在计算技术指标...")
            close_prices = df["Close"].squeeze()
            high_prices = df["High"].squeeze()
            low_prices = df["Low"].squeeze()

            # 原有指标
            df["MA20"] = ta.trend.sma_indicator(close_prices, window=20, fillna=True)
            df["RSI14"] = ta.momentum.rsi(close_prices, window=14, fillna=True)
            macd = ta.trend.MACD(close_prices, window_fast=12, window_slow=26, window_sign=9, fillna=True)
            df["MACD"] = macd.macd()
            df["MACD_signal"] = macd.macd_signal()

            # 新增指标
            # 布林带
            bollinger = ta.volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)
            df["Bollinger_High"] = bollinger.bollinger_hband()
            df["Bollinger_Low"] = bollinger.bollinger_lband()
            # 随机振荡器
            stoch = ta.momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True)
            df["Stoch_K"] = stoch.stoch()
            df["Stoch_D"] = stoch.stoch_signal()
            # 平均真实范围 (ATR)
            df["ATR"] = ta.volatility.average_true_range(high_prices, low_prices, close_prices, window=14, fillna=True)

            self.data = df.tail(60) # 仅保留最近60天的数据用于分析
            print(f"[{self.name}] 数据加载和技术指标计算成功，共 {len(self.data)} 条记录。")
        except Exception as e:
            print(f"[{self.name}] 错误：加载或处理 {self.ticker} 数据时发生异常: {e}")
            self.data = pd.DataFrame()
        return self.data

    async def load_data(self):
        """
        异步方法：在独立的线程中运行同步的数据加载和处理方法。
        """
        return await asyncio.to_thread(self._load_data_sync)

    def kline_to_text(self, kline_data: pd.DataFrame) -> str:
        """
        将K线数据的DataFrame（包含技术指标）转换为自然语言描述。
        """
        if kline_data.empty:
            return "没有可用的K线数据。"

        example_texts = []

        def get_formatted_value(series_or_scalar, is_int=False):
            val = series_or_scalar
            if isinstance(val, pd.Series):
                if val.empty: return 'N/A' # Handle empty Series
                try:
                    val = val.item()
                except ValueError:
                    return 'N/A'
            
            if isinstance(val, str):
                return val
            if pd.isna(val): return 'N/A' # Handle NaN/NaT

            if is_int:
                return f"{int(val)}"
            return f"{val:.2f}"

        for _, row in kline_data.iterrows():
            date_val = row['Date']
            if isinstance(date_val, pd.Series):
                date_val = date_val.item()
            date = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else 'N/A'
            
            text = (
                f"日期：{date}，"
                f"开盘价：{get_formatted_value(row['Open'])}，"
                f"最高价：{get_formatted_value(row['High'])}，"
                f"最低价：{get_formatted_value(row['Low'])}，"
                f"收盘价：{get_formatted_value(row['Close'])}，"
                f"成交量：{get_formatted_value(row['Volume'], is_int=True)}。"
                f" 20日均线：{get_formatted_value(row.get('MA20', 'N/A'))}。"
                f" 14日RSI：{get_formatted_value(row.get('RSI14', 'N/A'))}。"
                f" MACD：{get_formatted_value(row.get('MACD', 'N/A'))}。"
                f" MACD信号线：{get_formatted_value(row.get('MACD_signal', 'N/A'))}。"
                f" 布林带上轨：{get_formatted_value(row.get('Bollinger_High', 'N/A'))}。"
                f" 布林带下轨：{get_formatted_value(row.get('Bollinger_Low', 'N/A'))}。"
                f" 随机指标K线：{get_formatted_value(row.get('Stoch_K', 'N/A'))}。"
                f" 随机指标D线：{get_formatted_value(row.get('Stoch_D', 'N/A'))}。"
                f" 平均真实范围ATR：{get_formatted_value(row.get('ATR', 'N/A'))}。"
            )
            example_texts.append(text)
        
        return "\n".join(example_texts)

# 使用示例 (用于测试)
async def main_test():
    # 测试使用AAPL股票代码
    data_agent = DataAgent('AAPL')
    df = await data_agent.load_data()

    if df is not None and not df.empty:
        sample_df = df.tail(5)
        
        text_description = data_agent.kline_to_text(sample_df)
        
        print("\n--- K线数据转文本示例 ---")
        print(text_description)
    else:
        print("\n--- 测试失败：未能成功加载或处理数据 ---")

if __name__ == '__main__':
    asyncio.run(main_test())
