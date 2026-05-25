"""
项目对外入口文件。

当前项目的 test_runner.py 会执行：
    from agent import Agent

因此这个文件必须存在，并且至少提供：
- class Agent(BaseAgent)
- def act(self, input_data: AgentInput) -> AgentOutput
- def reset(self) -> None

实现策略建议：
1. 本文件尽量薄，只做“接线”和生命周期管理
2. 真正逻辑放到 gui_agent/ 下的 A/B/C/D/E 模块
3. 保持与 agent_base.BaseAgent 的接口完全兼容
"""

from __future__ import annotations

from agent_base import AgentInput, AgentOutput, BaseAgent
from gui_agent.e_integration_safety.main import IntegratedGUIAgentController


class Agent(BaseAgent):
    """给 TestRunner 调用的总 Agent。"""

    def _initialize(self) -> None:
        self.controller = IntegratedGUIAgentController(call_llm=self._call_api)

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        处理当前轮输入并返回标准动作。

        Args:
            input_data: 当前截图、任务指令、历史动作等

        Returns:
            AgentOutput: 必须符合 agent_base.py 的标准
        """
        return self.controller.run_step(input_data)

    def reset(self) -> None:
        """每个新任务开始前重置内部状态。"""
        self.controller.reset()
