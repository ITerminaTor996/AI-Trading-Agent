import sys
sys.stdout.reconfigure(encoding='utf-8')

import datetime
import requests
import re # 导入re模块用于正则表达式
import os
import asyncio

class NewsAgent:
    """
    负责获取和处理新闻数据，为LLM提供文本格式的输入。
    """
    def __init__(self):
        """
        初始化NewsAgent。
        API密钥将从环境变量 NEWS_API_KEY 中读取。
        """
        self.api_key = os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("错误：NEWS_API_KEY 环境变量未设置。")
        self.base_url = "https://newsapi.org/v2/everything"
        print("NewsAgent 初始化成功。")

    def _fetch_news_sync(self, query: str, page_size: int = 5) -> list[dict]:
        """
        从NewsAPI.org获取最新的财经新闻。

        :param query: 搜索关键词（例如：股票代码或公司名称）。
        :param page_size: 获取的新闻数量。
        :return: 新闻文章列表，每篇文章是一个字典。
        """
        print(f"正在从NewsAPI.org获取关于 \"{query}\" 的新闻...")
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
            print(f"错误：从NewsAPI.org获取新闻失败: {e}")
            return []
        except Exception as e:
            print(f"错误：处理NewsAPI响应时发生异常: {e}")
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
            content = re.sub(r'<[^>]+>', '', content) # 移除HTML标签
            content = re.sub(r'\[\+\d+ chars\]', '', content) # 移除[+XXXX chars]标记，修正正则表达式
            content = content.strip()

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
    from dotenv import load_dotenv
    # For standalone testing, load environment variables from .env file
    load_dotenv()

    try:
        # 初始化时会自动从环境变量加载API Key
        news_agent = NewsAgent()
        
        print("\n--- 新闻数据转文本示例 ---")
        print("注意：此测试需要您已在环境中设置 NEWS_API_KEY。")
        # 测试获取Apple Inc.的新闻
        news_data = await news_agent.fetch_news(query="Apple Inc.")
        news_text = news_agent.news_to_text(news_data)
        print(news_text)
        
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main_test())
