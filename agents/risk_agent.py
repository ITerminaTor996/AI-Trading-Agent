import sys
sys.stdout.reconfigure(encoding='utf-8')

class RiskAgent:
    """
    负责根据预设规则（如止盈止损）对投资组合进行风险管理。
    """
    def __init__(self, stop_loss_pct: float, take_profit_pct: float):
        """
        初始化风险管理Agent。

        :param stop_loss_pct: 止损百分比 (例如, 0.05 表示 5%的亏损).
        :param take_profit_pct: 止盈百分比 (例如, 0.10 表示 10%的盈利).
        """
        if not 0 < stop_loss_pct < 1:
            raise ValueError("止损百分比必须在0和1之间。")
        if not 0 < take_profit_pct < 1:
            raise ValueError("止盈百分比必须在0和1之间。")
            
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        print(f"RiskAgent 初始化成功，止损线: {self.stop_loss_pct:.2%}, 止盈线: {self.take_profit_pct:.2%}")

    def check_risk(self, current_price: float, average_buy_price: float, current_shares: int) -> dict | None:
        """
        检查当前持仓是否触发了止盈或止损条件。

        :param current_price: 当前市场价格。
        :param average_buy_price: 持仓的平均买入成本。
        :param current_shares: 当前持有的股数。
        :return: 如果触发风控，则返回一个决策字典；否则返回None。
        """
        if current_shares == 0 or average_buy_price == 0:
            return None # 没有持仓，无需风控

        # 计算当前盈亏百分比
        pnl_pct = (current_price - average_buy_price) / average_buy_price

        # 检查是否触发止损
        if pnl_pct <= -self.stop_loss_pct:
            return {
                "action": "SELL",
                "reason": f"触发 {self.stop_loss_pct:.2%} 止损线",
                "shares_to_sell": current_shares # 默认卖出全部
            }

        # 检查是否触发止盈
        if pnl_pct >= self.take_profit_pct:
            return {
                "action": "SELL",
                "reason": f"触发 {self.take_profit_pct:.2%} 止盈线",
                "shares_to_sell": current_shares # 默认卖出全部
            }

        return None # 未触发任何风控条件

if __name__ == '__main__':
    # --- 测试配置 ---
    STOP_LOSS = 0.05  # 5%
    TAKE_PROFIT = 0.10 # 10%

    # 1. 初始化Agent
    risk_agent = RiskAgent(stop_loss_pct=STOP_LOSS, take_profit_pct=TAKE_PROFIT)
    print("\n--- RiskAgent 测试开始 ---\n")

    # 2. 模拟场景
    scenarios = {
        "场景1: 触发止损": {
            "current_price": 94.0,
            "average_buy_price": 100.0,
            "current_shares": 10
        },
        "场景2: 触发止盈": {
            "current_price": 111.0,
            "average_buy_price": 100.0,
            "current_shares": 10
        },
        "场景3: 未触发任何条件": {
            "current_price": 102.0,
            "average_buy_price": 100.0,
            "current_shares": 10
        },
        "场景4: 没有持仓": {
            "current_price": 105.0,
            "average_buy_price": 100.0,
            "current_shares": 0
        }
    }

    # 3. 执行测试并打印结果
    for name, data in scenarios.items():
        print(f"--- {name} ---")
        print(f"持仓成本: {data['average_buy_price']}, 当前价格: {data['current_price']}, 持仓数量: {data['current_shares']}")
        decision = risk_agent.check_risk(
            current_price=data["current_price"],
            average_buy_price=data["average_buy_price"],
            current_shares=data["current_shares"]
        )
        if decision:
            print(f"风控决策: {decision['action']}, 原因: {decision['reason']}\n")
        else:
            print("风控决策: 无\n")