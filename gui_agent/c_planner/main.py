"""
C：Planner 工作目录入口。

E 模块主循环会按顺序调用 C 的这些函数：
1. build_planner_prompt(input_data, perception, memory, reflection)
2. plan_next_action(input_data, perception, memory, reflection, call_llm)

其中：
- build_planner_prompt 负责组织规划阶段输入
- plan_next_action 负责给出当前轮的最终 PlannerDecision
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from agent_base import ACTION_COMPLETE, ACTION_OPEN, AgentInput
from gui_agent.shared.schemas import MemoryState, PlannerDecision, ReflectionSignal, ScreenPerception


class PlannerModule:
    """C 模块：负责决定下一步动作。"""

    def build_planner_prompt(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        reflection: ReflectionSignal,
    ) -> List[Dict[str, Any]]:
        """
        构建规划阶段的提示词或消息列表。

        上层调用位置：
- E.run_step 的第 5 步

        Returns:
- list[dict]: 推荐直接返回规划模型的消息数组
- 消息中至少应体现任务目标、页面摘要、历史动作、反思建议
        """
        return []

    def plan_next_action(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        reflection: ReflectionSignal,
        call_llm: Callable[..., Any] | None = None,
    ) -> PlannerDecision:
        """
        根据任务、感知结果和反思信号，给出当前轮最终动作。

        上层调用位置：
- E.run_step 的第 6 步

        Returns:
- action: 标准动作名
- parameters: 标准动作参数
- thought: 文字说明，解释为什么要这么做
- target_element_id: 若动作针对某个元素，则填写其 element_id
- confidence: 当前动作置信度
- is_terminal: 是否打算结束任务
        """
        _ = self.build_planner_prompt(input_data, perception, memory, reflection)

        if reflection.need_backoff:
            advice = "；".join(reflection.recovery_advice) if reflection.recovery_advice else "检测到风险，先保守结束。"
            return PlannerDecision(
                action=ACTION_COMPLETE,
                parameters={},
                thought=advice,
                confidence=0.1,
                is_terminal=True,
            )

        if perception.app_name and not self._has_opened_app(memory):
            return PlannerDecision(
                action=ACTION_OPEN,
                parameters={"app_name": perception.app_name},
                thought=f"任务中提到了 {perception.app_name}，当前先尝试打开目标应用。",
                confidence=0.3,
                is_terminal=False,
            )

        return PlannerDecision(
            action=ACTION_COMPLETE,
            parameters={},
            thought="当前仅完成了主循环骨架，具体规划策略尚未实现，因此保守结束。",
            confidence=0.1,
            is_terminal=True,
        )

    def parse_planner_response(self, raw_text: str) -> PlannerDecision:
        """
        将规划模型原始输出解析成 PlannerDecision。

        Returns:
- PlannerDecision: 字段与 plan_next_action 的返回要求一致
        """
        return PlannerDecision(
            action=ACTION_COMPLETE,
            parameters={},
            thought=raw_text.strip() or "planner parser fallback",
            confidence=0.0,
            is_terminal=True,
        )

    @staticmethod
    def _has_opened_app(memory: MemoryState) -> bool:
        return any(step.action == ACTION_OPEN for step in memory.history)

