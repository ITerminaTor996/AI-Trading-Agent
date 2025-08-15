import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import asyncio
import yfinance as yf
import ta
from dotenv import load_dotenv
from agents.data_agent import DataAgent
from agents.decision_agent import DecisionAgent
from agents.news_agent import NewsAgent
from prompts.prompt_templates import PromptTemplates
from tqdm import tqdm # 导入tqdm用于显示进度条

# Load environment variables from .env file
load_dotenv()

# --- 从 main.py 移植过来的辅助函数 ---
def calculate_price_change(df_slice: pd.DataFrame) -> float:
    if df_slice.empty or len(df_slice) < 2:
        return 0.0
    # 使用 .item() 从Series中提取标量值，防止ValueError
    first_open = df_slice['Open'].iloc[0].item()
    last_close = df_slice['Close'].iloc[-1].item()
    if first_open == 0: return 0.0
    return (last_close - first_open) / first_open

def find_similar_examples(
    full_df: pd.DataFrame,
    current_change: float,
    window_size: int = 5,
    num_examples: int = 3,
    similarity_threshold: float = 0.1
) -> list[pd.DataFrame]:
    similar_examples = []
    search_end_idx = len(full_df) - (2 * window_size)
    if search_end_idx < window_size: return []

    for i in range(search_end_idx, window_size - 1, -1):
        window_slice = full_df.iloc[i - window_size : i]
        if len(window_slice) != window_size: continue
        window_change = calculate_price_change(window_slice)
        if abs(current_change - window_change) <= similarity_threshold:
            similar_examples.append((abs(current_change - window_change), window_slice))
    
    similar_examples.sort(key=lambda x: x[0])
    return [ex[1] for ex in similar_examples[:num_examples]]

class Portfolio:
    """
    一个简单的投资组合模拟器。
    """
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}
        self.trades = []

    def get_value(self, current_prices):
        stock_value = sum(shares * current_prices.get(ticker, 0) for ticker, shares in self.holdings.items())
        return self.cash + stock_value

    def execute_trade(self, ticker, decision, price, shares=1):
        if decision == "买入":
            if self.cash >= price * shares:
                self.cash -= price * shares
                self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
                self.trades.append({"date": pd.Timestamp.now(), "type": "BUY", "price": price, "shares": shares})
        elif decision == "卖出":
            if self.holdings.get(ticker, 0) >= shares:
                self.cash += price * shares
                self.holdings[ticker] -= shares
                self.trades.append({"date": pd.Timestamp.now(), "type": "SELL", "price": price, "shares": shares})

async def run_backtest(ticker: str, days_to_backtest: int, initial_capital: float, risk_preference: str, news_fetch_interval: int, report_interval_days: int):
    print(f"--- 开始对 {ticker} 进行回测 ---")
    print(f"回测周期: 最近 {days_to_backtest} 个交易日")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"风险偏好: {risk_preference}")
    print(f"新闻获取频率: 每 {news_fetch_interval} 天")
    print(f"中间报告频率: 每 {report_interval_days} 天")
    print("----------------------------------")

    try:
        decision_agent = DecisionAgent()
        news_agent = NewsAgent()
        data_agent_for_text = DataAgent(ticker)
    except ValueError as e:
        print(e)
        return

    portfolio = Portfolio(initial_capital)
    WINDOW_SIZE = 5
    data_load_days = days_to_backtest + 60
    print(f"正在加载最近 {data_load_days} 天的历史数据...")
    full_historical_data = yf.download(ticker, period=f"{data_load_days}d", auto_adjust=True, progress=False)
    if full_historical_data.empty or len(full_historical_data) < data_load_days - 5:
        print("错误：无法加载足够的回测数据。")
        return
    print("历史数据加载完成。")
    full_historical_data.reset_index(inplace=True) # 将Date索引转为列

    # --- 为整个历史数据集计算技术指标 ---
    print("正在为历史数据计算技术指标...")
    close_prices = full_historical_data["Close"].squeeze()
    full_historical_data["MA20"] = ta.trend.sma_indicator(close_prices, window=20, fillna=True)
    full_historical_data["RSI14"] = ta.momentum.rsi(close_prices, window=14, fillna=True)
    macd = ta.trend.MACD(close_prices, window_fast=12, window_slow=26, window_sign=9, fillna=True)
    full_historical_data["MACD"] = macd.macd()
    full_historical_data["MACD_signal"] = macd.macd_signal()
    print("技术指标计算完成。")

    backtest_start_index = 60
    backtest_range = range(backtest_start_index, len(full_historical_data))
    latest_news_text = "没有最新的相关新闻。"

    for i in tqdm(range(len(backtest_range)), desc="回测进度"):
        loop_index = backtest_start_index + i
        current_date = full_historical_data.index[loop_index]
        current_price = full_historical_data['Close'].iloc[loop_index].item()
        
        # --- 新闻获取逻辑 ---
        if i % news_fetch_interval == 0:
            try:
                news_data = await news_agent.fetch_news(query=ticker)
                latest_news_text = news_agent.news_to_text(news_data)
            except Exception as e:
                print(f"  获取新闻时出错: {e}")

        ai_data_view = full_historical_data.iloc[:loop_index]
        current_kline_data = ai_data_view.tail(WINDOW_SIZE)
        
        current_change = calculate_price_change(current_kline_data)
        historical_search_df = ai_data_view.iloc[:len(ai_data_view) - WINDOW_SIZE]
        found_examples = find_similar_examples(historical_search_df, current_change)

        examples_text = ""
        for j, ex_df in enumerate(found_examples):
            examples_text += f"\n--- 历史示例 {j+1} ---\n"
            examples_text += data_agent_for_text.kline_to_text(ex_df)

        current_data_text = data_agent_for_text.kline_to_text(current_kline_data)

        full_prompt = PromptTemplates.trading_decision_prompt(
            examples=examples_text,
            current_data=current_data_text,
            risk_preference=risk_preference,
            news_text=latest_news_text
        )

        decision_result = await asyncio.to_thread(decision_agent.generate_decision, full_prompt)
        decision = decision_result.get('decision', '持有')

        portfolio.execute_trade(ticker, decision, current_price)

        # --- 中间报告逻辑 ---
        if (i + 1) % report_interval_days == 0 or (i + 1) == len(backtest_range):
            current_portfolio_value = portfolio.get_value({ticker: current_price})
            current_return_pct = (current_portfolio_value - initial_capital) / initial_capital * 100
            print(f"\n--- 日期: {full_historical_data['Date'].iloc[loop_index].strftime('%Y-%m-%d')} 中间报告 ---")
            print(f"  当前资产: ${current_portfolio_value:,.2f}")
            print(f"  当前收益率: {current_return_pct:.2f}%")
            print("----------------------------------")

    print("\n--- 回测结束 --- generating report ---")
    final_price = full_historical_data['Close'].iloc[-1].item()
    final_value = portfolio.get_value({ticker: final_price})
    total_return_pct = (final_value - initial_capital) / initial_capital * 100

    buy_and_hold_start_price = full_historical_data['Close'].iloc[backtest_start_index].item()
    buy_and_hold_shares = initial_capital / buy_and_hold_start_price
    buy_and_hold_final_value = buy_and_hold_shares * final_price
    buy_and_hold_return_pct = (buy_and_hold_final_value - initial_capital) / initial_capital * 100

    print("\n--- 回测结果报告 ---")
    print(f"股票代码: {ticker}")
    print(f"回测时段: {full_historical_data['Date'].iloc[backtest_start_index].strftime('%Y-%m-%d')} to {full_historical_data['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print("----------------------------------")
    print("AI 策略表现:")
    print(f"  初始资本: ${initial_capital:,.2f}")
    print(f"  最终资产: ${final_value:,.2f}")
    print(f"  总收益率: {total_return_pct:.2f}%")
    print(f"  交易次数: {len(portfolio.trades)}")
    print("----------------------------------")
    print("买入并持有策略表现:")
    print(f"  初始资本: ${initial_capital:,.2f}")
    print(f"  最终资产: ${buy_and_hold_final_value:,.2f}")
    print(f"  总收益率: {buy_and_hold_return_pct:.2f}%")
    print("----------------------------------")

if __name__ == '__main__':
    # --- 回测配置 ---
    TICKER = "AAPL"
    DAYS_TO_BACKTEST = 30
    INITIAL_CAPITAL = 10000.0
    RISK_PREFERENCE = "稳健"
    NEWS_FETCH_INTERVAL = 5 # 每隔N天获取一次新闻
    REPORT_INTERVAL_DAYS = 3 # 每隔N天打印一次中间报告

    asyncio.run(run_backtest(TICKER, DAYS_TO_BACKTEST, INITIAL_CAPITAL, RISK_PREFERENCE, NEWS_FETCH_INTERVAL, REPORT_INTERVAL_DAYS))