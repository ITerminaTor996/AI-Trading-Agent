import sys
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import pandas as pd

# 导入重构后的Agent和新的app_config模块
from agents.data_agent import DataAgent
from agents.decision_agent import DecisionAgent
from agents.news_agent import NewsAgent
from agents.financial_agent import FinancialAgent
from prompts.prompt_templates import PromptTemplates
import app_config

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

def find_similar_examples(full_df: pd.DataFrame, current_change: float, window_size: int, num_examples: int, similarity_threshold: float) -> list[pd.DataFrame]:
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
        diff = abs(current_change - window_change)
        if diff <= similarity_threshold:
            similar_examples.append((diff, window_slice))
    similar_examples.sort(key=lambda x: x[0])
    return [ex[1] for ex in similar_examples[:num_examples]]

async def main():
    # --- 1. 从app_config模块加载配置 ---
    conf = app_config.MAIN_CONFIG
    stock_ticker = conf["default_ticker"]
    risk_preference = conf["default_risk_preference"]
    window_size = conf["window_size"]
    num_examples = conf["num_examples"]
    similarity_threshold = conf["similarity_threshold"]

    print("欢迎使用AI交易决策助手！")
    print("----------------------------------")
    print(f"配置加载成功：")
    print(f"  - 股票代码: {stock_ticker}")
    print(f"  - 风险偏好: {risk_preference}")
    print("----------------------------------")

    # --- 2. 初始化Agent ---
    try:
        decision_agent = DecisionAgent(
            api_key=app_config.DEEPSEEK_API_KEY,
            model_name=app_config.DEEPSEEK_MODEL_NAME,
            base_url=app_config.DEEPSEEK_BASE_URL
        )
        news_agent = NewsAgent(api_key=app_config.NEWS_API_KEY)
        data_agent = DataAgent(ticker=stock_ticker)
        financial_agent = FinancialAgent()
    except ValueError as e:
        print(f"Agent初始化失败: {e}")
        print("请确保您已经在 .env 文件或环境中正确设置了 API 密钥。")
        input("\n按任意键退出...")
        return

    # --- 3. 并行加载所有数据 ---
    print("\n正在并行加载市场数据、新闻和财务摘要...")
    try:
        results = await asyncio.gather(
            data_agent.load_data(),
            news_agent.fetch_news(query=stock_ticker),
            financial_agent.get_financial_summary(stock_ticker)
        )
        df, news_data_list, financial_summary_data = results
        print("所有数据加载完成。")
    except Exception as e:
        print(f"数据加载过程中发生错误: {e}")
        input("\n按任意键退出...")
        return

    if df.empty or len(df) < (2 * window_size):
        print("错误：未能加载足够的数据进行分析，无法进行决策。")
        input("\n按任意键退出...")
        return

    # --- 4. 准备K线历史示例和当前数据 ---
    current_kline_data = df.tail(window_size)
    current_change = calculate_price_change(current_kline_data)
    print(f"当前市场 ({stock_ticker}) 最近{window_size}天价格变化: {current_change:.2%}")
    
    historical_search_df = df.iloc[:len(df) - window_size]
    found_examples = find_similar_examples(historical_search_df, current_change, window_size, num_examples, similarity_threshold)
    
    if found_examples:
        print(f"找到 {len(found_examples)} 个与当前市场趋势相似的历史示例。")
    else:
        print("未找到与当前市场趋势相似的历史示例。")

    # --- 5. 将所有数据转换为文本 ---
    examples_text = ""
    for i, ex_df in enumerate(found_examples):
        examples_text += f"\n--- 历史示例 {i+1} ---\n"
        examples_text += data_agent.kline_to_text(ex_df)
    
    current_data_text = data_agent.kline_to_text(current_kline_data)
    news_text = news_agent.news_to_text(news_data_list)
    financial_summary_text = financial_agent.summary_to_text(financial_summary_data)
    print("数据转换完成。")

    # --- 6. 构建并显示Prompt ---
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
    
    # --- 7. 请求AI生成决策 ---
    print("正在请求AI生成决策...")
    decision = decision_agent.generate_decision(full_prompt)

    # --- 8. 显示结果 ---
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

