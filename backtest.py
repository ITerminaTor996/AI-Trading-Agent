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
from agents.risk_agent import RiskAgent
from agents.financial_agent import FinancialAgent # 导入FinancialAgent
from prompts.prompt_templates import PromptTemplates
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# --- 辅助函数 ---
def calculate_price_change(df_slice: pd.DataFrame) -> float:
    if df_slice.empty or len(df_slice) < 2:
        return 0.0
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
    一个更完善的投资组合模拟器，能追踪持仓的平均成本。
    """
    def __init__(self, initial_capital=10000.0, commission_rate=0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.holdings = {}
        self.trades = []

    def get_value(self, current_prices: dict) -> float:
        stock_value = 0.0
        for ticker, data in self.holdings.items():
            stock_value += data["shares"] * current_prices.get(ticker, 0)
        return self.cash + stock_value

    def execute_trade(self, ticker: str, price: float, decision_result: dict):
        decision = decision_result.get('decision', '持有')
        trade_percent = decision_result.get('trade_percent', 0.0)
        
        trade_percent = max(0.0, min(1.0, trade_percent))
        
        position = self.holdings.get(ticker)

        if decision == "买入" and trade_percent > 0:
            investment_amount = self.cash * trade_percent
            shares_to_buy = int(investment_amount / price)
            
            if shares_to_buy > 0:
                trade_cost = shares_to_buy * price
                commission = trade_cost * self.commission_rate
                if self.cash >= trade_cost + commission:
                    self.cash -= (trade_cost + commission)
                    
                    if position:
                        old_shares = position["shares"]
                        old_avg_price = position["average_buy_price"]
                        new_total_shares = old_shares + shares_to_buy
                        new_avg_price = ((old_avg_price * old_shares) + (price * shares_to_buy)) / new_total_shares
                        position["shares"] = new_total_shares
                        position["average_buy_price"] = new_avg_price
                    else:
                        self.holdings[ticker] = {"shares": shares_to_buy, "average_buy_price": price}
                    
                    self.trades.append({"date": pd.Timestamp.now(), "type": "BUY", "price": price, "shares": shares_to_buy, "commission": commission})

        elif decision == "卖出" and position and trade_percent > 0:
            shares_to_sell = int(position["shares"] * trade_percent)
            
            if shares_to_sell > 0:
                trade_cost = shares_to_sell * price
                commission = trade_cost * self.commission_rate
                
                self.cash += (trade_cost - commission)
                position["shares"] -= shares_to_sell

                if position["shares"] <= 0:
                    del self.holdings[ticker]
                
                self.trades.append({"date": pd.Timestamp.now(), "type": "SELL", "price": price, "shares": shares_to_sell, "commission": commission})

async def run_backtest(ticker: str, days_to_backtest: int, initial_capital: float, risk_preference: str, news_fetch_interval: int, commission_rate: float, stop_loss_pct: float, take_profit_pct: float):
    print(f"--- 开始对 {ticker} 进行回测 ---")
    print(f"回测周期: 最近 {days_to_backtest} 个交易日")
    print(f"初始资金: ${initial_capital:,.2f}")
    print(f"风险偏好: {risk_preference}")
    print(f"交易手续费率: {commission_rate*100:.3f}%")
    print(f"新闻获取频率: 每 {news_fetch_interval} 天")
    print(f"止损线: {stop_loss_pct:.2%}")
    print(f"止盈线: {take_profit_pct:.2%}")
    print("----------------------------------")

    try:
        decision_agent = DecisionAgent()
        news_agent = NewsAgent()
        data_agent_for_text = DataAgent(ticker)
        risk_agent = RiskAgent(stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
        financial_agent = FinancialAgent() # 初始化FinancialAgent
    except ValueError as e:
        print(e)
        return

    portfolio = Portfolio(initial_capital, commission_rate)
    WINDOW_SIZE = 5
    data_load_days = days_to_backtest + 60
    print(f"正在加载最近 {data_load_days} 天的历史数据...")
    full_historical_data = yf.download(ticker, period=f"{data_load_days}d", auto_adjust=True, progress=False)
    if full_historical_data.empty or len(full_historical_data) < data_load_days - 5:
        print("错误：无法加载足够的回测数据。")
        return
    print("历史数据加载完成。")
    full_historical_data.reset_index(inplace=True)

    print("正在为历史数据计算技术指标...")
    close_prices = full_historical_data["Close"].squeeze()
    high_prices = full_historical_data["High"].squeeze()
    low_prices = full_historical_data["Low"].squeeze()
    full_historical_data["MA20"] = ta.trend.sma_indicator(close_prices, window=20, fillna=True)
    full_historical_data["RSI14"] = ta.momentum.rsi(close_prices, window=14, fillna=True)
    macd = ta.trend.MACD(close_prices, window_fast=12, window_slow=26, window_sign=9, fillna=True)
    full_historical_data["MACD"] = macd.macd()
    full_historical_data["MACD_signal"] = macd.macd_signal()
    bollinger = ta.volatility.BollingerBands(close_prices, window=20, window_dev=2, fillna=True)
    full_historical_data["Bollinger_High"] = bollinger.bollinger_hband()
    full_historical_data["Bollinger_Low"] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high_prices, low_prices, close_prices, window=14, smooth_window=3, fillna=True)
    full_historical_data["Stoch_K"] = stoch.stoch()
    full_historical_data["Stoch_D"] = stoch.stoch_signal()
    full_historical_data["ATR"] = ta.volatility.average_true_range(high_prices, low_prices, close_prices, window=14, fillna=True)
    print("技术指标计算完成。")

    # 在回测开始前，获取一次财务摘要
    financial_summary_data = await financial_agent.get_financial_summary(ticker)
    financial_summary_text = financial_agent.summary_to_text(financial_summary_data)

    backtest_start_index = 60
    backtest_range = range(backtest_start_index, len(full_historical_data))
    
    print("\n--- 开始每日回测循环 ---")
    for i in tqdm(range(len(backtest_range)), desc="回测进度"):
        loop_index = backtest_start_index + i
        current_price = full_historical_data['Close'].iloc[loop_index].item()
        date_str = full_historical_data['Date'].iloc[loop_index].strftime('%Y-%m-%d')
        
        position = portfolio.holdings.get(ticker)
        if position:
            risk_decision = risk_agent.check_risk(current_price=current_price, average_buy_price=position["average_buy_price"], current_shares=position["shares"])
            if risk_decision:
                shares_to_sell = risk_decision["shares_to_sell"]
                sell_percent = shares_to_sell / position["shares"] if position["shares"] > 0 else 0
                portfolio.execute_trade(ticker, current_price, {"decision": "卖出", "trade_percent": sell_percent})
                tqdm.write(f"日期: {date_str} | 决策: 卖出   | 原因: {risk_decision['reason']:<12} | 价格: ${current_price:7.2f} | 总资产: ${portfolio.get_value({ticker: current_price}):9,.2f}")
                continue

        if i % news_fetch_interval == 0:
            try:
                news_data = await news_agent.fetch_news(query=ticker)
                latest_news_text = news_agent.news_to_text(news_data)
            except Exception as e:
                tqdm.write(f"获取新闻时出错: {e}")

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

        current_shares_before_trade = position['shares'] if position else 0
        if current_shares_before_trade > 0:
            portfolio_status = f"当前持有 {current_shares_before_trade} 股 {ticker}。"
        else:
            portfolio_status = "当前无持仓。"

        full_prompt = PromptTemplates.trading_decision_prompt(
            examples=examples_text,
            current_data=current_data_text,
            portfolio_status=portfolio_status,
            risk_preference=risk_preference,
            news_text=latest_news_text,
            financial_summary_text=financial_summary_text # 传递财务摘要
        )

        decision_result = await asyncio.to_thread(decision_agent.generate_decision, full_prompt)
        portfolio.execute_trade(ticker, current_price, decision_result)

        current_portfolio_value = portfolio.get_value({ticker: current_price})
        decision = decision_result.get('decision', '持有')
        trade_percent = decision_result.get('trade_percent', 0.0) * 100
        
        tqdm.write(f"日期: {date_str} | 决策: {decision:<4} | 仓位%: {trade_percent:<3.0f}% | 持仓: {current_shares_before_trade:<4} | 价格: ${current_price:7.2f} | 总资产: ${current_portfolio_value:9,.2f}")

    print("\n--- 回测结束 --- 正在生成报告 ---", flush=True)
    final_price = full_historical_data['Close'].iloc[-1].item()
    final_value = portfolio.get_value({ticker: final_price})
    total_return_pct = (final_value - initial_capital) / initial_capital * 100
    buy_and_hold_start_price = full_historical_data['Close'].iloc[backtest_start_index].item()
    buy_and_hold_shares = initial_capital / buy_and_hold_start_price
    buy_and_hold_final_value = buy_and_hold_shares * final_price
    buy_and_hold_return_pct = (buy_and_hold_final_value - initial_capital) / initial_capital * 100
    print("\n--- 回测结果报告 ---", flush=True)
    print(f"股票代码: {ticker}", flush=True)
    print(f"回测时段: {full_historical_data['Date'].iloc[backtest_start_index].strftime('%Y-%m-%d')} to {full_historical_data['Date'].iloc[-1].strftime('%Y-%m-%d')}", flush=True)
    print("----------------------------------", flush=True)
    print("AI 策略表现:", flush=True)
    print(f"  初始资本: ${initial_capital:,.2f}", flush=True)
    print(f"  最终资产: ${final_value:,.2f}", flush=True)
    print(f"  总收益率: {total_return_pct:.2f}%", flush=True)
    print(f"  交易次数: {len(portfolio.trades)}", flush=True)
    print("----------------------------------", flush=True)
    print("买入并持有策略表现:", flush=True)
    print(f"  初始资本: ${initial_capital:,.2f}", flush=True)
    print(f"  最终资产: ${buy_and_hold_final_value:,.2f}", flush=True)
    print(f"  总收益率: {buy_and_hold_return_pct:.2f}%", flush=True)
    print("----------------------------------", flush=True)

if __name__ == '__main__':
    # --- 回测配置 ---
    TICKER = "AAPL"
    DAYS_TO_BACKTEST = 1
    INITIAL_CAPITAL = 10000.0
    RISK_PREFERENCE = "稳健"
    COMMISSION_RATE = 0.001
    NEWS_FETCH_INTERVAL = 5
    STOP_LOSS_PCT = 0.05
    TAKE_PROFIT_PCT = 0.10

    asyncio.run(
        run_backtest(
            TICKER, 
            DAYS_TO_BACKTEST, 
            INITIAL_CAPITAL, 
            RISK_PREFERENCE, 
            NEWS_FETCH_INTERVAL, 
            COMMISSION_RATE, 
            STOP_LOSS_PCT, 
            TAKE_PROFIT_PCT
        )
    )

    input("\n按任意键退出...")