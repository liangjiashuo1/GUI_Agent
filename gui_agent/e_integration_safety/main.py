"""
E：Integration / Safety 工作目录入口。

本文件负责主循环编排。当前规定单步主循环的执行顺序必须是：

1. D.bootstrap_memory(input_data)
2. B.build_perception_prompt(input_data, memory)
3. B.perceive_screen(input_data, memory, call_llm)
4. D.reflect_before_planning(memory, perception)
5. C.build_planner_prompt(input_data, perception, memory, reflection)
6. C.plan_next_action(input_data, perception, memory, reflection, call_llm)
7. E.validate_decision(decision, memory, perception)
8. D.update_after_decision(memory, perception, final_decision)
9. A.build_device_command(final_decision)
10. A.compile_decision(final_decision)

其中第 9 步是为真机控制预留的接口，当前离线评测场景只做命令构建，不实际执行。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent_base import ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE, AgentInput, AgentOutput
from gui_agent.a_executor.main import ExecutorModule
from gui_agent.b_perception.main import PerceptionModule
from gui_agent.c_planner.main import PlannerModule
from gui_agent.d_memory_reflection.main import MemoryReflectionModule
from gui_agent.shared.schemas import (
    MemoryState,
    PlannerDecision,
    ReflectionSignal,
    SafetyCheckResult,
    ScreenPerception,
)

logger = logging.getLogger(__name__)


class SafetyGuard:
    """E 模块中的安全检查器。"""

    def validate_decision(
        self,
        decision: PlannerDecision,
        memory: MemoryState,
        perception: ScreenPerception,
        memory_module: MemoryReflectionModule,
    ) -> SafetyCheckResult:
        """
        对 C 模块输出的 PlannerDecision 做安全校验。

        Returns:
- ok: 是否允许执行
- reason: 若不允许执行，给出失败原因
- sanitized_decision: 若做了修正，返回修正后的 PlannerDecision
        """
        if memory_module.should_block_repeat(decision.action, decision.parameters):
            return SafetyCheckResult(ok=False, reason="decision_blocked_by_reflection")

        if decision.action == "CLICK":
            point = decision.parameters.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return SafetyCheckResult(ok=False, reason="invalid_click_point")

        if decision.action == ACTION_SCROLL:
            start_point = decision.parameters.get("start_point")
            end_point = decision.parameters.get("end_point")
            if not self._is_point(start_point) or not self._is_point(end_point):
                return SafetyCheckResult(ok=False, reason="invalid_scroll_points")

        if decision.action == ACTION_TYPE and "text" not in decision.parameters:
            return SafetyCheckResult(ok=False, reason="missing_type_text")

        if decision.action == ACTION_OPEN and "app_name" not in decision.parameters:
            return SafetyCheckResult(ok=False, reason="missing_open_app_name")

        return SafetyCheckResult(ok=True, sanitized_decision=decision)

    @staticmethod
    def _is_point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2


class IntegratedGUIAgentController:
    """E 模块主控制器，负责把 A/B/C/D 串成可运行主循环。"""

    def __init__(self, call_llm: Callable[..., Any] | None = None) -> None:
        self.call_llm = call_llm
        self.perception = PerceptionModule()
        self.planner = PlannerModule()
        self.memory = MemoryReflectionModule()
        self.executor = ExecutorModule()
        self.safety = SafetyGuard()

    def reset(self) -> None:
        """重置跨任务状态。"""
        self.memory.reset()

    def run_step(self, input_data: AgentInput) -> AgentOutput:
        """
        单步主循环。

        执行顺序：
1. 恢复记忆
2. 构建感知输入
3. 获取屏幕理解
4. 生成反思信号
5. 构建规划输入
6. 生成动作决策
7. 做安全检查
8. 更新记忆
9. 生成设备命令
10. 编译成 AgentOutput
        """
        memory = self.memory.bootstrap_memory(input_data)
        _ = self.perception.build_perception_prompt(input_data, memory)
        perception = self.perception.perceive_screen(input_data, memory, self.call_llm)
        reflection = self.memory.reflect_before_planning(memory, perception)
        _ = self.planner.build_planner_prompt(input_data, perception, memory, reflection)
        decision = self.planner.plan_next_action(input_data, perception, memory, reflection, self.call_llm)

        safety_result = self.safety.validate_decision(decision, memory, perception, self.memory)
        final_decision = safety_result.sanitized_decision if safety_result.ok else self.make_fallback_decision(
            reflection,
            f"安全检查未通过: {safety_result.reason}",
        )

        updated_memory = self.memory.update_after_decision(memory, perception, final_decision)
        device_command = self.executor.build_device_command(final_decision)
        self._log_step(input_data, updated_memory, perception, reflection, final_decision, safety_result, device_command)
        return self.executor.compile_decision(final_decision)

    def make_fallback_decision(self, reflection: ReflectionSignal, reason: str) -> PlannerDecision:
        """
        当安全检查失败或整体不确定性较高时，生成保守兜底动作。

        Returns:
- action: 通常是 COMPLETE
- parameters: 空字典
- thought: 说明为什么进入兜底
- confidence: 低置信度
- is_terminal: True
        """
        advice = "；".join(reflection.recovery_advice) if reflection.recovery_advice else ""
        thought = reason if not advice else f"{reason}；恢复建议：{advice}"
        return PlannerDecision(
            action=ACTION_COMPLETE,
            parameters={},
            thought=thought,
            confidence=0.0,
            is_terminal=True,
        )

    def _log_step(
        self,
        input_data: AgentInput,
        memory: MemoryState,
        perception: ScreenPerception,
        reflection: ReflectionSignal,
        decision: PlannerDecision,
        safety_result: SafetyCheckResult,
        device_command: dict,
    ) -> None:
        """记录本轮主循环的关键信息，便于调试和演示。"""
        logger.info(
            (
                "step=%s app=%s page=%s action=%s params=%s safety_ok=%s "
                "risks=%s thought=%s device=%s history_size=%s"
            ),
            input_data.step_count,
            perception.app_name,
            perception.page_type,
            decision.action,
            decision.parameters,
            safety_result.ok,
            reflection.risk_flags,
            decision.thought,
            device_command,
            len(memory.history),
        )

