import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import asyncio
from agents.data_agent import DataAgent
from agents.decision_agent import DecisionAgent
from agents.news_agent import NewsAgent
from agents.financial_agent import FinancialAgent # 导入FinancialAgent
from prompts.prompt_templates import PromptTemplates
import os

def calculate_price_change(df_slice: pd.DataFrame) -> float:
    """
    计算DataFrame切片的价格变化百分比。
    """
    if df_slice.empty or len(df_slice) < 2:
        return 0.0
    first_open = df_slice['Open'].iloc[0]
    if isinstance(first_open, pd.Series): first_open = first_open.item()
    last_close = df_slice['Close'].iloc[-1]
    if isinstance(last_close, pd.Series): last_close = last_close.item()
    if first_open == 0: return 0.0
    return (last_close - first_open) / first_open

def find_similar_examples(full_df: pd.DataFrame, current_change: float, window_size: int = 5, num_examples: int = 3, similarity_threshold: float = 0.1) -> list[pd.DataFrame]:
    """
    在历史数据中查找与当前价格变化相似的示例。
    """
    similar_examples = []
    search_end_idx = len(full_df) - (2 * window_size)
    if search_end_idx < window_size: return []
    for i in range(search_end_idx, window_size - 1, -1):
        window_slice = full_df.iloc[i - window_size : i]
        if len(window_slice) != window_size: continue
        window_change = calculate_price_change(window_slice)
        diff = abs(current_change - window_change) # 修复：确保diff在此处定义
        if diff <= similarity_threshold:
            similar_examples.append((diff, window_slice))
    return [ex[1] for ex in similar_examples[:num_examples]]

async def main():
    # --- 配置参数 ---
    WINDOW_SIZE = 5
    NUM_EXAMPLES = 3
    SIMILARITY_THRESHOLD = 0.1

    print("欢迎使用AI交易决策助手！")
    print("----------------------------------")

    # 1. 获取用户输入
    valid_risk_preferences = ['保守', '稳健', '激进', '中性']
    risk_preference = ""
    while risk_preference not in valid_risk_preferences:
        risk_preference = input(f"请输入您的风险偏好（{', '.join(valid_risk_preferences)}）：").strip()
        if not risk_preference:
            risk_preference = "中性"
            print(f"未输入风险偏好，默认为：{risk_preference}")
            break
        elif risk_preference not in valid_risk_preferences:
            print("输入无效，请从列表中选择一个有效的风险偏好。")
    print(f"您的风险偏好设置为：{risk_preference}")
    stock_ticker = input("请输入您想查询的股票代码（例如：AAPL, MSFT）：").strip().upper()
    if not stock_ticker:
        stock_ticker = "AAPL"
        print(f"未输入股票代码，将使用默认代码：{stock_ticker}")
    else:
        print(f"您选择的股票代码为：{stock_ticker}")

    # 2. 初始化Agent
    try:
        data_agent = DataAgent(ticker=stock_ticker)
        decision_agent = DecisionAgent()
        news_agent = NewsAgent()
        financial_agent = FinancialAgent() # 初始化FinancialAgent
    except ValueError as e:
        print(e)
        print("请确保您已经在环境中正确设置了 DEEPSEEK_API_KEY 和 NEWS_API_KEY。")
        input("\n按任意键退出...")
        return

    # 3. 并行加载所有数据
    print("\n正在并行加载市场数据、新闻和财务摘要...")
    data_coroutine = data_agent.load_data()
    news_coroutine = news_agent.fetch_news(query=stock_ticker)
    financial_coroutine = financial_agent.get_financial_summary(stock_ticker)
    results = await asyncio.gather(
        data_coroutine,
        news_coroutine,
        financial_coroutine
    )
    df, news_data_list, financial_summary_data = results
    print("所有数据加载完成。")

    if df.empty or len(df) < (2 * WINDOW_SIZE):
        print("错误：未能加载足够的数据进行分析，无法进行决策。")
        input("\n按任意键退出...")
        return

    # 4. 准备K线历史示例和当前数据
    current_kline_data = df.tail(WINDOW_SIZE)
    current_change = calculate_price_change(current_kline_data)
    print(f"当前市场 ({stock_ticker}) 最近{WINDOW_SIZE}天价格变化: {current_change:.2%}")
    historical_search_df = df.iloc[:len(df) - WINDOW_SIZE]
    found_examples = find_similar_examples(historical_search_df, current_change, window_size=WINDOW_SIZE, num_examples=NUM_EXAMPLES, similarity_threshold=SIMILARITY_THRESHOLD)
    examples_kline_data_list = []
    if found_examples:
        print(f"找到 {len(found_examples)} 个与当前市场趋势相似的历史示例。")
        examples_kline_data_list = found_examples
    else:
        print("未找到与当前市场趋势相似的历史示例，将使用固定的历史数据作为备用。")
        fallback_example_start = len(df) - (2 * WINDOW_SIZE)
        fallback_example_end = len(df) - WINDOW_SIZE
        if fallback_example_start >= 0:
            examples_kline_data_list.append(df.iloc[fallback_example_start:fallback_example_end])

    # 5. 将所有数据转换为文本
    examples_text = ""
    for i, ex_df in enumerate(examples_kline_data_list):
        examples_text += f"\n--- 历史示例 {i+1} ---"
        examples_text += data_agent.kline_to_text(ex_df)
    current_data_text = data_agent.kline_to_text(current_kline_data)
    news_text = news_agent.news_to_text(news_data_list)
    financial_summary_text = financial_agent.summary_to_text(financial_summary_data)
    print("数据转换完成。")

    # 6. 构建并显示Prompt
    print("\n数据处理完成，正在构建Prompt...")
    portfolio_status = "当前无持仓。"
    full_prompt = PromptTemplates.trading_decision_prompt(
        examples=examples_text,
        current_data=current_data_text,
        portfolio_status=portfolio_status,
        risk_preference=risk_preference,
        news_text=news_text,
        financial_summary_text=financial_summary_text
    )

    print("\n--- 生成的完整Prompt ---")
    print(full_prompt)
    print("------------------------\n")
    
    # 7. 请求AI生成决策
    print("正在请求AI生成决策...")
    decision = decision_agent.generate_decision(full_prompt)

    # 8. 显示结果
    print("\n----------------------------------")
    print("AI分析的市场数据：")
    print(current_data_text)
    print("----------------------------------")
    print("AI交易决策：")
    trade_percent = decision.get('trade_percent', 0.0) * 100
    print(f"决策: {decision.get('decision', '未知')}")
    print(f"原因: {decision.get('reason', '无')}")
    print(f"信心指数: {decision.get('confidence', 0.0):.2f}")
    print(f"建议仓位: {trade_percent:.0f}%")
    print("----------------------------------")

    input("\n按任意键退出...")

if __name__ == '__main__':
    asyncio.run(main())
