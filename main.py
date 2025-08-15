import sys
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import asyncio
from agents.data_agent import DataAgent
from agents.decision_agent import DecisionAgent
from agents.news_agent import NewsAgent # 导入NewsAgent
from prompts.prompt_templates import PromptTemplates
import os

def calculate_price_change(df_slice: pd.DataFrame) -> float:
    """
    计算DataFrame切片的价格变化百分比。
    (最后收盘价 - 第一个开盘价) / 第一个开盘价
    """
    if df_slice.empty or len(df_slice) < 2:
        return 0.0
    
    # 确保获取的是标量值，而不是Series
    first_open = df_slice['Open'].iloc[0]
    if isinstance(first_open, pd.Series): # 防御性检查
        first_open = first_open.item()

    last_close = df_slice['Close'].iloc[-1]
    if isinstance(last_close, pd.Series): # 防御性检查
        last_close = last_close.item()

    if first_open == 0: # 避免除以零
        return 0.0
    return (last_close - first_open) / first_open

def find_similar_examples(
    full_df: pd.DataFrame,
    current_change: float,
    window_size: int = 5,
    num_examples: int = 3,
    similarity_threshold: float = 0.1 # 相似度阈值，例如0.1表示10%的绝对差异
) -> list[pd.DataFrame]:
    """
    在历史数据中查找与当前价格变化相似的示例。
    """
    similar_examples = []
    # 确保有足够的数据进行滑动窗口，并排除最近用于current_kline_data的部分
    # 假设current_kline_data是最后window_size天，examples_kline_data是再往前window_size天
    # 所以我们从倒数第 (2 * window_size) 天开始往前找
    search_end_idx = len(full_df) - (2 * window_size) 

    if search_end_idx < window_size: # 如果历史数据不足以进行搜索
        return []

    for i in range(search_end_idx, window_size - 1, -1): # 从后往前遍历
        window_slice = full_df.iloc[i - window_size : i]
        if len(window_slice) != window_size:
            continue # 确保窗口大小正确

        window_change = calculate_price_change(window_slice)
        
        # 计算相似度（绝对差值）
        diff = abs(current_change - window_change)
        
        if diff <= similarity_threshold:
            similar_examples.append((diff, window_slice))
    
    # 按相似度排序，取最相似的num_examples个
    similar_examples.sort(key=lambda x: x[0])
    
    return [ex[1] for ex in similar_examples[:num_examples]]

async def main():
    # --- 配置参数 ---
    WINDOW_SIZE = 5  # 用于分析当前市场和历史示例的窗口大小（天）
    NUM_EXAMPLES = 3  # 在历史数据中查找的相似示例数量
    SIMILARITY_THRESHOLD = 0.1 # 判断历史示例与当前市场趋势的相似度阈值

    print("欢迎使用AI交易决策助手！")
    print("----------------------------------")

    # 1. 获取用户输入
    valid_risk_preferences = ['保守', '稳健', '激进', '中性']
    risk_preference = ""
    while risk_preference not in valid_risk_preferences:
        risk_preference = input(f"请输入您的风险偏好（{', '.join(valid_risk_preferences)}）：").strip()
        if not risk_preference:
            risk_preference = "中性" # 默认值
            print(f"未输入风险偏好，默认为：{risk_preference}")
            break
        elif risk_preference not in valid_risk_preferences:
            print("输入无效，请从列表中选择一个有效的风险偏好。")
    
    print(f"您的风险偏好设置为：{risk_preference}")

    stock_ticker = input("请输入您想查询的股票代码（例如：AAPL, MSFT）：").strip().upper()
    if not stock_ticker:
        stock_ticker = "AAPL" # 默认值
        print(f"未输入股票代码，将使用默认代码：{stock_ticker}")
    else:
        print(f"您选择的股票代码为：{stock_ticker}")

    # 2. 初始化Agent
    try:
        data_agent = DataAgent(ticker=stock_ticker)
        decision_agent = DecisionAgent()
        news_agent = NewsAgent()
    except ValueError as e:
        print(e)
        print("请确保您已经在环境中正确设置了 DEEPSEEK_API_KEY 和 NEWS_API_KEY。")
        input("\n按任意键退出...")
        return

    # 3. 并行加载数据和新闻
    print("\n正在并行加载市场数据和新闻...")
    results = await asyncio.gather(
        data_agent.load_data(),
        news_agent.fetch_news(query=stock_ticker)
    )
    df, news_data_list = results
    print("市场数据和新闻加载完成。")

    if df.empty or len(df) < (2 * WINDOW_SIZE):
        print("错误：未能加载足够的数据进行分析，无法进行决策。")
        input("\n按任意键退出...")
        return

    # 4. 准备历史示例和当前数据
    current_kline_data = df.tail(WINDOW_SIZE)
    current_change = calculate_price_change(current_kline_data)
    print(f"当前市场 ({stock_ticker}) 最近{WINDOW_SIZE}天价格变化: {current_change:.2%}")

    # 查找相似的历史示例
    historical_search_df = df.iloc[:len(df) - WINDOW_SIZE]
    found_examples = find_similar_examples(
        historical_search_df, 
        current_change, 
        window_size=WINDOW_SIZE, 
        num_examples=NUM_EXAMPLES, 
        similarity_threshold=SIMILARITY_THRESHOLD
    )

    examples_kline_data_list = []
    if found_examples:
        print(f"找到 {len(found_examples)} 个与当前市场趋势相似的历史示例。")
        examples_kline_data_list = found_examples
    else:
        print("未找到与当前市场趋势相似的历史示例，将使用固定的历史数据作为备用。")
        # 回退逻辑：使用倒数第二个窗口作为示例
        fallback_example_start = len(df) - (2 * WINDOW_SIZE)
        fallback_example_end = len(df) - WINDOW_SIZE
        if fallback_example_start >= 0:
            examples_kline_data_list.append(df.iloc[fallback_example_start:fallback_example_end])

    # 5. 将数据转换为文本
    examples_text = ""
    for i, ex_df in enumerate(examples_kline_data_list):
        examples_text += f"\n--- 历史示例 {i+1} ---\n"
        examples_text += data_agent.kline_to_text(ex_df)

    current_data_text = data_agent.kline_to_text(current_kline_data)

    # 6. 获取新闻数据
    news_text = news_agent.news_to_text(news_data_list)
    print("新闻数据处理完成。")

    # 7. 构建并显示Prompt
    print("\n数据处理完成，正在构建Prompt...")
    full_prompt = PromptTemplates.trading_decision_prompt(
        examples=examples_text,
        current_data=current_data_text,
        risk_preference=risk_preference,
        news_text=news_text
    )
    
    # 8. 请求AI生成决策
    print("\n正在请求AI生成决策...")
    decision = decision_agent.generate_decision(full_prompt)

    # 9. 显示结果
    print("\n----------------------------------")
    print("AI分析的市场数据：")
    print(current_data_text)
    print("----------------------------------")
    print("AI交易决策：")
    print(f"决策: {decision.get('decision', '未知')}")
    print(f"原因: {decision.get('reason', '无')}")
    print(f"信心指数: {decision.get('confidence', 0.0):.2f}")
    print("----------------------------------")

    input("\n按任意键退出...")

if __name__ == '__main__':
    asyncio.run(main())
