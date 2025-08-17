import sys
sys.stdout.reconfigure(encoding='utf-8')

class BaseAgent:
    """
    所有Agent的基类，用于提供通用功能和标准化结构。
    """
    def __init__(self, name: str):
        """
        初始化一个Agent实例。
        :param name: Agent的名称，用于日志和识别。
        """
        self._name = name
        print(f"{self.name} 初始化成功。")

    @property
    def name(self) -> str:
        """
        返回Agent的名称。
        """
        return self._name
