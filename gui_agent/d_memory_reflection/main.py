"""
D：Memory / Reflection 工作目录入口。

E 模块主循环会按顺序调用 D 的这些函数：
1. bootstrap_memory(input_data)
2. reflect_before_planning(memory, perception)
3. update_after_decision(memory, perception, decision)

其中：
- bootstrap_memory 负责把历史动作恢复成当前 MemoryState
- reflect_before_planning 负责在规划前给出风险信号和恢复建议
- update_after_decision 负责在本轮决策后更新记忆
"""

from __future__ import annotations

from typing import Dict, List

from agent_base import AgentInput
from gui_agent.shared.schemas import (
    ErrorRecord,
    MemoryState,
    PlannerDecision,
    ReflectionSignal,
    ScreenPerception,
    StepRecord,
    action_signature,
)


class MemoryReflectionModule:
    """D 模块：负责状态跟踪、反思和错误恢复。"""

    def __init__(self) -> None:
        self._memory = MemoryState(task_goal="")

    def reset(self, task_goal: str | None = None) -> None:
        """重置跨任务状态。"""
        self._memory = MemoryState(task_goal=task_goal or "")

    def bootstrap_memory(self, input_data: AgentInput) -> MemoryState:
        """
        根据 AgentInput 中的历史动作，恢复当前任务记忆。

        上层调用位置：
- E.run_step 的第 1 步

        Returns:
- task_goal: 当前任务原始指令
- history: 历史动作列表
- forbidden_actions: 历史上已经标记为不该重复尝试的动作
- error_records: 已有的错误记录
- notes: 自由备注
        """
        history = self._rebuild_history(input_data.history_actions)
        self._memory.task_goal = input_data.instruction
        self._memory.history = history
        return self._memory

    def reflect_before_planning(
        self,
        memory: MemoryState,
        perception: ScreenPerception,
    ) -> ReflectionSignal:
        """
        在规划前做一次轻量反思，判断是否卡住或有重复风险。

        上层调用位置：
- E.run_step 的第 4 步

        Returns:
- need_backoff: 是否建议本轮不要继续激进尝试
- risk_flags: 风险标签列表
- recovery_advice: 给 C/E 的恢复建议
- blocked_action_signatures: 当前不建议继续执行的动作签名
        """
        signal = ReflectionSignal(
            need_backoff=False,
            risk_flags=[],
            recovery_advice=[],
            blocked_action_signatures=list(memory.forbidden_actions),
        )

        if len(memory.history) >= 3:
            last_signatures = [action_signature(step.action, step.parameters) for step in memory.history[-3:]]
            if len(set(last_signatures)) == 1:
                signal.need_backoff = True
                signal.risk_flags.append("repeated_loop")
                signal.recovery_advice.append("最近三步动作完全相同，下一步不要继续重复。")

        if "perception_not_implemented" in perception.warnings:
            signal.risk_flags.append("weak_perception")
            signal.recovery_advice.append("当前屏幕理解较弱，规划时需要保守。")

        return signal

    def update_after_decision(
        self,
        memory: MemoryState,
        perception: ScreenPerception,
        decision: PlannerDecision,
    ) -> MemoryState:
        """
        在本轮动作决策完成后，把这一步写回记忆。

        上层调用位置：
- E.run_step 的第 7 步之后

        Returns:
- 更新后的 MemoryState
- history 中追加本轮动作记录
- 如发现重复风险，可同步更新 error_records 和 forbidden_actions
        """
        step = StepRecord(
            step_index=len(memory.history) + 1,
            action=decision.action,
            parameters=decision.parameters,
            thought=decision.thought,
            screen_summary=perception.screen_summary,
        )
        memory.history.append(step)

        signature = action_signature(decision.action, decision.parameters)
        recent_signatures = [action_signature(item.action, item.parameters) for item in memory.history[-3:]]
        if recent_signatures.count(signature) >= 2:
            memory.error_records.append(
                ErrorRecord(
                    error_type="repeated_loop",
                    trigger_step=len(memory.history),
                    message="最近几步出现重复动作。",
                    recovery_hint="下一步改用不同策略，不要继续重复当前动作。",
                )
            )
            if signature not in memory.forbidden_actions:
                memory.forbidden_actions.append(signature)

        self._memory = memory
        return self._memory

    def should_block_repeat(self, action: str, parameters: Dict[str, object]) -> bool:
        """
        查询某个动作是否已经被标记为不应重复执行。

        Returns:
- bool: True 表示应该拦截，False 表示允许继续
        """
        return action_signature(action, parameters) in self._memory.forbidden_actions

    @staticmethod
    def _rebuild_history(history_actions: List[Dict[str, object]]) -> List[StepRecord]:
        """将 TestRunner 的历史动作恢复为统一 StepRecord。"""
        rebuilt: List[StepRecord] = []
        for item in history_actions:
            rebuilt.append(
                StepRecord(
                    step_index=int(item.get("step", len(rebuilt) + 1)),
                    action=str(item.get("action", "")),
                    parameters=dict(item.get("parameters", {})),
                    thought=str(item.get("raw_output", "")),
                )
            )
        return rebuilt

