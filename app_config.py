"""
集中化配置文件
"""
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量 (如果存在)
load_dotenv()

# --- API密钥配置 ---
# 优先从环境变量获取，如果未设置，则使用默认值或留空
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# --- 大语言模型配置 ---
LLM_PROVIDER = "deepseek" # LLM提供商, 例如 "deepseek", "openai"
# DEEPSEEK 模型配置
DEEPSEEK_MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# --- 市场数据源配置 ---
# yfinance 是默认的，未来可以扩展其他数据源
DATA_SOURCE = "yfinance"

# --- 主程序 (main.py) 配置 ---
MAIN_CONFIG = {
    "window_size": 5,
    "num_examples": 3,
    "similarity_threshold": 0.1,
    "default_ticker": "AAPL", # 默认股票代码
    "default_risk_preference": "中性" # 默认风险偏好
}

# --- 回测 (backtest.py) 配置 ---
BACKTEST_CONFIG = {
    "ticker": "AAPL",
    "days_to_backtest": 5,
    "initial_capital": 10000.0,
    "risk_preference": "稳健",
    "commission_rate": 0.001,
    "news_fetch_interval": 5, # 每隔几天获取一次新闻
    "stop_loss_pct": 0.05, # 5% 止损
    "take_profit_pct": 0.10 # 10% 止盈
}

# --- 检查API密钥是否存在 ---
if not DEEPSEEK_API_KEY:
    print("警告: 环境变量 DEEPSEEK_API_KEY 未设置。")

if not NEWS_API_KEY:
    print("警告: 环境变量 NEWS_API_KEY 未设置。")
