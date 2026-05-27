"""
E：Integration / Safety 工作目录入口

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

增强点：
- 熔断器：连续安全失败 3 次自动终止任务，避免无效重试。
- 动作参数扩展校验：坐标去重、滑动距离过短、TYPE 文本合法性等，仅拦截明显错误。
- Trace 日志：每步生成结构化 trace，可导出用于离线分析。
- 钩子机制：支持注册 before_step / after_step 回调。
- 记忆快照导出：提供 get_memory_snapshot() 方法。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

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
    """E 模块中的安全检查器（增强版）。

    新增校验：
    - 连续相同 CLICK 坐标拦截
    - SCROLL 滑动距离过短拦截
    - TYPE 文本包含控制字符拦截
    """

    MAX_SAME_CLICK_COUNT = 3          # 连续相同坐标点击最大次数
    MIN_SCROLL_DISTANCE = 50          # 滑动最小距离（归一化坐标）

    def validate_decision(
        self,
        decision: PlannerDecision,
        memory: MemoryState,
        perception: ScreenPerception,
        memory_module: MemoryReflectionModule,
    ) -> SafetyCheckResult:
        # 1. 反射模块拦截重复动作
        if memory_module.should_block_repeat(decision.action, decision.parameters):
            return SafetyCheckResult(ok=False, reason="decision_blocked_by_reflection")

        # 2. 动作参数基本校验 + 扩展校验
        if decision.action == "CLICK":
            point = decision.parameters.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return SafetyCheckResult(ok=False, reason="invalid_click_point")
            if not (0 <= point[0] <= 1000 and 0 <= point[1] <= 1000):
                return SafetyCheckResult(ok=False, reason="click_point_out_of_bounds")
            # 连续相同坐标拦截
            if self._is_repeated_click(memory, point):
                return SafetyCheckResult(ok=False, reason="repeated_click_on_same_point")

        elif decision.action == ACTION_SCROLL:
            start = decision.parameters.get("start_point")
            end = decision.parameters.get("end_point")
            if not self._is_point(start) or not self._is_point(end):
                return SafetyCheckResult(ok=False, reason="invalid_scroll_points")
            if not (0 <= start[0] <= 1000 and 0 <= start[1] <= 1000 and
                    0 <= end[0] <= 1000 and 0 <= end[1] <= 1000):
                return SafetyCheckResult(ok=False, reason="scroll_point_out_of_bounds")
            # 滑动距离过短拦截
            dist = math.hypot(end[0] - start[0], end[1] - start[1])
            if dist < self.MIN_SCROLL_DISTANCE:
                return SafetyCheckResult(ok=False, reason="scroll_distance_too_short")

        elif decision.action == ACTION_TYPE:
            if "text" not in decision.parameters:
                return SafetyCheckResult(ok=False, reason="missing_type_text")
            text = decision.parameters["text"]
            if not text or len(text) > 200:
                return SafetyCheckResult(ok=False, reason="invalid_type_text_length")
            # 禁止控制字符（保留常用空白符）
            if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', text):
                return SafetyCheckResult(ok=False, reason="type_text_contains_control_chars")

        elif decision.action == ACTION_OPEN:
            if "app_name" not in decision.parameters or not decision.parameters["app_name"]:
                return SafetyCheckResult(ok=False, reason="missing_open_app_name")

        return SafetyCheckResult(ok=True, sanitized_decision=decision)

    def _is_repeated_click(self, memory: MemoryState, point: tuple) -> bool:
        """检查最近 MAX_SAME_CLICK_COUNT 个 CLICK 动作的坐标是否相同"""
        click_count = 0
        for step in reversed(memory.history):
            if step.action == "CLICK":
                if step.parameters.get("point") == list(point):
                    click_count += 1
                    if click_count >= self.MAX_SAME_CLICK_COUNT:
                        return True
                else:
                    break
        return False

    @staticmethod
    def _is_point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2


class IntegratedGUIAgentController:
    """E 模块主控制器（增强版，非侵入式）。"""

    # 熔断阈值
    MAX_CONSECUTIVE_SAFETY_FAILURES = 3

    def __init__(self, call_llm: Callable[..., Any] | None = None) -> None:
        self.call_llm = call_llm
        self.perception = PerceptionModule()
        self.planner = PlannerModule()
        self.memory = MemoryReflectionModule()
        self.executor = ExecutorModule()
        self.safety = SafetyGuard()

        # 增强功能相关属性
        self.consecutive_safety_failures = 0      # 熔断器计数器
        self.traces: List[Dict[str, Any]] = []    # Trace 历史
        self.hooks: List[Callable] = []           # 外部钩子

    def reset(self) -> None:
        """重置跨任务状态。"""
        self.memory.reset()
        self.consecutive_safety_failures = 0
        self.traces.clear()

    def run_step(self, input_data: AgentInput) -> AgentOutput:
        """单步主循环。"""
        # 执行 before_step 钩子
        self._run_hooks("before_step", input_data=input_data, memory=None)

        memory = self.memory.bootstrap_memory(input_data)
        _ = self.perception.build_perception_prompt(input_data, memory)
        perception = self.perception.perceive_screen(input_data, memory, self.call_llm)
        reflection = self.memory.reflect_before_planning(memory, perception)
        _ = self.planner.build_planner_prompt(input_data, perception, memory, reflection)
        decision = self.planner.plan_next_action(input_data, perception, memory, reflection, self.call_llm)

        safety_result = self.safety.validate_decision(decision, memory, perception, self.memory)

        # 熔断器逻辑
        if not safety_result.ok:
            self.consecutive_safety_failures += 1
            if self.consecutive_safety_failures >= self.MAX_CONSECUTIVE_SAFETY_FAILURES:
                logger.warning("连续安全失败达到阈值，强制终止任务")
                final_decision = PlannerDecision(
                    action=ACTION_COMPLETE,
                    parameters={},
                    thought=f"连续 {self.consecutive_safety_failures} 次安全检查失败，强制终止",
                    confidence=0.0,
                    is_terminal=True,
                )
                self.consecutive_safety_failures = 0   # 避免下一次重复触发
            else:
                final_decision = self.make_fallback_decision(
                    reflection,
                    f"安全检查未通过: {safety_result.reason}",
                )
        else:
            self.consecutive_safety_failures = 0
            final_decision = safety_result.sanitized_decision

        updated_memory = self.memory.update_after_decision(memory, perception, final_decision)
        device_command = self.executor.build_device_command(final_decision)
        self._log_step(input_data, updated_memory, perception, reflection, final_decision, safety_result, device_command)

        # 构建 Trace 并存储
        trace = self._build_trace(input_data, decision, safety_result, final_decision, perception, reflection, device_command)
        self.traces.append(trace)

        # 执行 after_step 钩子
        self._run_hooks("after_step", input_data=input_data, memory=updated_memory,
                        perception=perception, decision=final_decision)

        return self.executor.compile_decision(final_decision)

    def make_fallback_decision(self, reflection: ReflectionSignal, reason: str) -> PlannerDecision:
        """安全检查失败时的兜底决策。"""
        advice = "；".join(reflection.recovery_advice) if reflection.recovery_advice else ""
        thought = reason if not advice else f"{reason}；恢复建议：{advice}"
        return PlannerDecision(
            action=ACTION_COMPLETE,
            parameters={},
            thought=thought,
            confidence=0.0,
            is_terminal=True,
        )

    # ---------- 增强功能辅助方法 ----------

    def register_hook(self, hook: Callable) -> None:
        """注册一个钩子函数，签名需为 hook(**kwargs)。"""
        self.hooks.append(hook)

    def get_memory_snapshot(self) -> Dict[str, Any]:
        """导出当前记忆状态的序列化快照。"""
        from dataclasses import asdict
        return asdict(self.memory.get_current_memory()) if hasattr(self.memory, 'get_current_memory') else {}

    def get_traces(self) -> List[Dict[str, Any]]:
        """返回所有步骤的 Trace 列表。"""
        return self.traces

    def _build_trace(
        self,
        input_data: AgentInput,
        raw_decision: PlannerDecision,
        safety_result: SafetyCheckResult,
        final_decision: PlannerDecision,
        perception: ScreenPerception,
        reflection: ReflectionSignal,
        device_command: dict,
    ) -> Dict[str, Any]:
        """构建单步结构化 Trace 字典。"""
        return {
            "step": input_data.step_count,
            "perception_app": perception.app_name,
            "perception_page": perception.page_type,
            "perception_elements_count": len(perception.elements),
            "keyboard_visible": perception.keyboard_visible,
            "planner_raw_decision": {
                "action": raw_decision.action,
                "params": raw_decision.parameters,
                "thought": raw_decision.thought,
                "confidence": raw_decision.confidence,
            },
            "safety": {
                "ok": safety_result.ok,
                "reason": safety_result.reason,
            },
            "final_decision": {
                "action": final_decision.action,
                "params": final_decision.parameters,
                "thought": final_decision.thought,
                "confidence": final_decision.confidence,
                "is_terminal": final_decision.is_terminal,
            },
            "reflection_risks": reflection.risk_flags,
            "device_command": device_command,
        }

    def _run_hooks(self, hook_type: str, **kwargs) -> None:
        """执行所有已注册的钩子，异常不影响主流程。"""
        for hook in self.hooks:
            try:
                hook(hook_type=hook_type, **kwargs)
            except Exception:
                logger.debug("Hook execution failed", exc_info=True)

    def _log_step(self, input_data, memory, perception, reflection, decision, safety_result, device_command):
        logger.info(
            "step=%s app=%s page=%s action=%s params=%s safety_ok=%s risks=%s thought=%s device=%s history_size=%s",
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