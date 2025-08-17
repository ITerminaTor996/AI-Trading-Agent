import sys
sys.stdout.reconfigure(encoding='utf-8')

import datetime
import requests
import re # 导入re模块用于正则表达式
import asyncio
from .base_agent import BaseAgent

class NewsAgent(BaseAgent):
    """
    负责获取和处理新闻数据，为LLM提供文本格式的输入。
    """
    def __init__(self, api_key: str):
        """
        初始化NewsAgent。
        :param api_key: NewsAPI.org的API密钥。
        """
        super().__init__(name="新闻舆情Agent")
        if not api_key:
            raise ValueError("错误：NEWS_API_KEY 未提供。")
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def _fetch_news_sync(self, query: str, page_size: int = 5) -> list[dict]:
        """
        从NewsAPI.org获取最新的财经新闻。

        :param query: 搜索关键词（例如：股票代码或公司名称）。
        :param page_size: 获取的新闻数量。
        :return: 新闻文章列表，每篇文章是一个字典。
        """
        print(f"[{self.name}] 正在从NewsAPI.org获取关于 \"{query}\" 的新闻...")
        params = {
            'q': query, # 搜索关键词
            'language': 'en', # 语言：英文
            'sortBy': 'relevancy', # 按相关性排序
            'pageSize': page_size, # 获取的新闻数量
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # 如果请求失败，抛出HTTPError
            data = response.json()
            
            articles = []
            if data['status'] == 'ok':
                for article in data['articles']:
                    # 过滤掉没有标题或内容的文章
                    if not article.get('title') or not article.get('content'):
                        continue
                    articles.append({
                        "title": article['title'],
                        "content": article['content'],
                        "publish_time": datetime.datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')) # 转换ISO格式日期
                    })
            return articles
        except requests.exceptions.RequestException as e:
            print(f"[{self.name}] 错误：从NewsAPI.org获取新闻失败: {e}")
            return []
        except Exception as e:
            print(f"[{self.name}] 错误：处理NewsAPI响应时发生异常: {e}")
            return []

    async def fetch_news(self, query: str, page_size: int = 5) -> list[dict]:
        return await asyncio.to_thread(self._fetch_news_sync, query, page_size)

    def news_to_text(self, news_list: list[dict]) -> str:
        """
        将新闻列表转换为自然语言描述，适合LLM阅读。
        """
        if not news_list:
            return "没有最新的相关新闻。"

        formatted_news = []
        for news in news_list:
            title = news.get("title", "无标题")
            content = news.get("content", "无内容")
            
            # 清理内容：移除HTML标签和[+XXXX chars]标记
            try:
                content = re.sub(r'<[^>]+>', '', content) # 移除HTML标签
                content = re.sub(r'\[\+\d+ chars\]', '', content) # 移除[+XXXX chars]标记
                content = content.strip()
            except Exception as e:
                print(f"[{self.name}] 清理新闻内容时发生错误: {e}")
                content = title # 如果内容清理失败，至少使用标题作为内容

            # 确保publish_time是datetime对象
            publish_time = news.get("publish_time", "未知时间")
            if isinstance(publish_time, datetime.datetime):
                publish_time = publish_time.strftime('%Y-%m-%d %H:%M')
            
            formatted_news.append(
                f"【新闻】发布时间: {publish_time}\n标题: {title}\n内容: {content}\n"
            )
        return "\n---\n".join(formatted_news)

# 示例用法 (用于测试)
async def main_test():
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import app_config

    try:
        news_agent = NewsAgent(api_key=app_config.NEWS_API_KEY)
        
        print(f"\n--- {news_agent.name} 测试 ---")
        if not app_config.NEWS_API_KEY:
            print("警告: NEWS_API_KEY 未在 app_config.py 或 .env 文件中设置，测试将失败。")

        news_data = await news_agent.fetch_news(query="Apple Inc.")
        news_text = news_agent.news_to_text(news_data)
        print(news_text)
        
    except ValueError as ve:
        print(ve)
    except ImportError:
        print("错误：无法导入app_config模块。请确保 `app_config.py` 文件在项目根目录，")
        print("并且你正在从项目根目录运行此脚本，或者已经将项目根目录添加到了PYTHONPATH。")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    asyncio.run(main_test())
