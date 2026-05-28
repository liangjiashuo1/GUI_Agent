"""
D: Memory / Reflection 工作目录入口。

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

from typing import Any, Dict, List, Optional

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

    _SEVERE_RISK_FLAGS = {"repeated_loop", "oscillation", "planner_stalling"}
    _RECENT_REPEAT_WINDOW = 3

    def __init__(self) -> None:
        self._memory = MemoryState(task_goal="")

    def reset(self, task_goal: str | None = None) -> None:
        """重置跨任务状态。"""
        self._memory = MemoryState(task_goal=task_goal or "")

    def get_current_memory(self) -> MemoryState:
        """返回当前持有的记忆对象，供 E 模块做快照导出。"""
        return self._memory

    #===================================================================================================
    # 详细函数介绍（输入和输出是什么），功能是什么
    #
    # 函数名：
    # bootstrap_memory
    #
    # 输入：
    # - input_data: AgentInput
    #   来自 E 模块主流程的第 1 步输入，包含：
    #   1. instruction: 用户任务目标
    #   2. history_actions: 历史动作列表
    #   3. step_count / current_image / extra 等上下文信息
    #
    # 输出：
    # - MemoryState
    #   D 模块恢复出的当前任务记忆，主要包含：
    #   1. task_goal: 当前任务目标
    #   2. history: 统一格式的历史步骤列表
    #   3. current_subgoal: 当前推断出的子目标
    #   4. completed_subgoals: 已完成子目标
    #   5. error_records: 历史错误记录
    #   6. forbidden_actions: 不建议重复执行的动作签名
    #   7. notes: 备注信息
    #
    # 功能：
    # - 把 AgentInput 中的 history_actions 恢复为统一的 StepRecord 列表
    # - 基于历史记录恢复 D 自己维护的记忆状态
    # - 为后续 B/C/E 模块提供可消费的 MemoryState
    #
    # 在主流程中的位置：
    # 1. D.bootstrap_memory(input_data)
    #===================================================================================================
    def bootstrap_memory(self, input_data: AgentInput) -> MemoryState:
        """
        根据 AgentInput 中的历史动作，恢复当前任务记忆。

        上层调用位置：
        - E.run_step 的第 1 步
        """
        history = self._rebuild_history(input_data.history_actions)
        memory = MemoryState(task_goal=input_data.instruction, history=history)
        self._restore_memory_metadata(memory)
        self._memory = memory
        return self._memory

    #===================================================================================================
    # 详细函数介绍（输入和输出是什么），功能是什么
    #
    # 函数名：
    # reflect_before_planning
    #
    # 输入：
    # - memory: MemoryState
    #   由 D.bootstrap_memory 输出，表示当前已恢复的任务记忆
    # - perception: ScreenPerception
    #   由 B.perceive_screen 输出，表示当前屏幕理解结果
    #
    # 输出：
    # - ReflectionSignal
    #   给 C 和 E 使用的反思结果，主要包含：
    #   1. need_backoff: 是否建议回退/保守处理
    #   2. risk_flags: 风险标签列表
    #   3. recovery_advice: 恢复建议
    #   4. blocked_action_signatures: 当前不建议继续执行的动作签名
    #
    # 功能：
    # - 在规划前分析是否存在重复、震荡、无进展、感知过弱等风险
    # - 把历史和当前屏幕状态压缩成结构化风险信号
    # - 为 C 模块提供 prompt 约束，为 E 模块提供保守决策依据
    #
    # 在主流程中的位置：
    # 4. D.reflect_before_planning(memory, perception)
    #===================================================================================================
    def reflect_before_planning(
        self,
        memory: MemoryState,
        perception: ScreenPerception,
    ) -> ReflectionSignal:
        """
        在规划前做一次轻量反思，判断是否卡住或有重复风险。

        上层调用位置：
        - E.run_step 的第 4 步
        """
        risk_flags: List[str] = []
        recovery_advice: List[str] = []
        blocked_action_signatures = list(memory.forbidden_actions)

        self._add_risk_if(
            self._detect_exact_repeated_loop(memory),
            "repeated_loop",
            "最近三步动作完全相同，下一步不要继续重复当前策略。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_repeated_click_same_point(memory),
            "repeated_click",
            "最近连续点击同一坐标，优先改为滚动、返回或选择其他元素。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_oscillation(memory),
            "oscillation",
            "最近动作在来回震荡，避免重复上一轮路径，优先尝试不同类型的动作。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_scroll_no_progress(memory, perception),
            "scroll_no_progress",
            "最近多次滚动但页面变化很小，优先检查弹窗、输入框或顶部提交按钮。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_no_progress(memory, perception),
            "no_progress",
            "最近几步页面几乎没有变化，规划时不要机械重复，可优先尝试其他入口。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_too_many_completes(memory),
            "planner_stalling",
            "近期多次保守结束，说明当前策略可能停滞，下一步需要避免继续直接结束任务。",
            risk_flags,
            recovery_advice,
        )
        self._add_risk_if(
            self._detect_weak_perception(perception),
            "weak_perception",
            "当前屏幕理解较弱，规划时优先使用明确元素，避免编造目标。",
            risk_flags,
            recovery_advice,
        )

        blocked_action_signatures = self._extend_blocked_signatures(
            blocked_action_signatures,
            memory,
            risk_flags,
        )

        return ReflectionSignal(
            need_backoff=any(flag in self._SEVERE_RISK_FLAGS for flag in risk_flags),
            risk_flags=risk_flags,
            recovery_advice=recovery_advice,
            blocked_action_signatures=blocked_action_signatures,
        )

    #===================================================================================================
    # 详细函数介绍（输入和输出是什么），功能是什么
    #
    # 函数名：
    # update_after_decision
    #
    # 输入：
    # - memory: MemoryState
    #   当前轮开始时的记忆状态
    # - perception: ScreenPerception
    #   B 模块输出的当前屏幕理解结果
    # - decision: PlannerDecision
    #   经过 C 规划、并可能被 E 安全层修正后的最终动作
    #
    # 输出：
    # - MemoryState
    #   更新后的任务记忆，包含新增的历史步骤以及同步更新后的：
    #   1. current_subgoal
    #   2. completed_subgoals
    #   3. notes
    #   4. error_records
    #   5. forbidden_actions
    #
    # 功能：
    # - 把当前轮 final_decision 写入 history
    # - 根据动作结果推进任务记忆和子目标
    # - 记录重复、震荡、无进展等错误状态
    # - 为下一轮 D/B/C/E 提供更新后的上下文
    #
    # 在主流程中的位置：
    # 8. D.update_after_decision(memory, perception, final_decision)
    #===================================================================================================
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
        """
        step = self._build_step_record(perception, decision, len(memory.history) + 1)
        memory.history.append(step)

        self._update_progress_fields(memory, perception, decision)
        self._update_forbidden_actions(memory, decision)
        self._update_error_state(memory, perception, decision)

        self._memory = memory
        return self._memory

    #===================================================================================================
    # 详细函数介绍（输入和输出是什么），功能是什么
    #
    # 函数名：
    # should_block_repeat
    #
    # 输入：
    # - action: str
    #   当前待执行动作名，例如 CLICK / TYPE / SCROLL / OPEN / COMPLETE
    # - parameters: Dict[str, object]
    #   当前待执行动作参数
    #
    # 输出：
    # - bool
    #   True: 建议 E 的安全层拦截该动作
    #   False: 允许继续执行
    #
    # 功能：
    # - 根据 D 模块内部维护的 forbidden_actions 和最近动作签名
    #   判断该动作是否属于明显重复执行
    # - 为 E.validate_decision(...) 提供快速布尔判断
    #
    # 在主流程中的位置：
    # 7. E.validate_decision(decision, memory, perception)
    #    -> 间接调用 D.should_block_repeat(action, parameters)
    #===================================================================================================
    def should_block_repeat(self, action: str, parameters: Dict[str, object]) -> bool:
        """
        查询某个动作是否已经被标记为不应重复执行。

        Returns:
        - bool: True 表示应该拦截，False 表示允许继续
        """
        signature = self._make_signature(action, parameters)
        if signature in self._memory.forbidden_actions:
            return True

        recent_signatures = self.get_recent_signatures(self._RECENT_REPEAT_WINDOW)
        return len(recent_signatures) >= 2 and recent_signatures[-1] == signature and recent_signatures[-2] == signature

    def get_recent_actions(self, n: int = 5) -> List[StepRecord]:
        """返回最近 n 条动作记录。"""
        if n <= 0:
            return []
        return list(self._memory.history[-n:])

    def get_recent_signatures(self, n: int = 5) -> List[str]:
        """返回最近 n 条动作签名。"""
        return [self._make_signature(step.action, step.parameters) for step in self.get_recent_actions(n)]

    def has_recent_action(self, action: str, within: int = 3) -> bool:
        """判断最近若干步内是否出现过某类动作。"""
        normalized = self._normalize_action(action)
        return any(step.action == normalized for step in self.get_recent_actions(within))

    def has_recent_typed_text(self, text: str, within: int = 10) -> bool:
        """判断最近若干步是否输入过某段文本。"""
        target = self._normalize_text(text)
        if not target:
            return False
        for step in self.get_recent_actions(within):
            if step.action != "TYPE":
                continue
            typed = self._normalize_text(step.parameters.get("text", ""))
            if target in typed or typed in target:
                return True
        return False

    @classmethod
    def _restore_memory_metadata(cls, memory: MemoryState) -> None:
        """根据历史记录恢复 D 模块维护的衍生状态。"""
        for index, step in enumerate(memory.history, start=1):
            cls._restore_step_metadata(memory, step, index)

        cls._restore_forbidden_from_history(memory)
        memory.current_subgoal = cls._infer_current_subgoal(memory)

    @classmethod
    def _restore_step_metadata(cls, memory: MemoryState, step: StepRecord, step_index: int) -> None:
        """从单步历史恢复 error_records、notes、completed_subgoals 和 forbidden_actions。"""
        if step.thought:
            cls._append_note(memory, f"第{step_index}步 thought: {step.thought[:120]}")
        if step.note:
            cls._append_note(memory, step.note)
        if step.error_type:
            cls._append_error_record(
                memory,
                error_type=step.error_type,
                trigger_step=step_index,
                message=f"历史中记录到错误: {step.error_type}",
                recovery_hint="规划时需要避免重复该失败路径。",
            )

        cls._restore_progress_from_step(memory, step)

    @classmethod
    def _restore_progress_from_step(cls, memory: MemoryState, step: StepRecord) -> None:
        """从历史动作恢复任务进度。"""
        if step.action == "OPEN":
            app_name = str(step.parameters.get("app_name", "")).strip()
            if app_name:
                cls._append_completed_subgoal(memory, f"opened_app:{app_name}")
        elif step.action == "TYPE":
            text = str(step.parameters.get("text", "")).strip()
            if text:
                cls._append_completed_subgoal(memory, f"typed_text:{text[:40]}")
        elif step.action == "COMPLETE":
            cls._append_completed_subgoal(memory, "task_marked_complete")

    @classmethod
    def _restore_forbidden_from_history(cls, memory: MemoryState) -> None:
        """根据最近历史恢复需要拦截的动作签名。"""
        if cls._detect_exact_repeated_loop(memory):
            last_step = memory.history[-1]
            cls._add_forbidden_action(memory, cls._make_signature(last_step.action, last_step.parameters))

        if cls._detect_repeated_click_same_point(memory):
            last_step = memory.history[-1]
            cls._add_forbidden_action(memory, cls._make_signature(last_step.action, last_step.parameters))

    def _update_progress_fields(
        self,
        memory: MemoryState,
        perception: ScreenPerception,
        decision: PlannerDecision,
    ) -> None:
        """维护 current_subgoal、completed_subgoals 和 notes。"""
        if decision.action == "OPEN":
            app_name = str(decision.parameters.get("app_name", "")).strip()
            if app_name:
                self._append_completed_subgoal(memory, f"opened_app:{app_name}")
                memory.current_subgoal = f"确认应用「{app_name}」是否已打开"
                self._append_note(memory, f"已请求打开应用: {app_name}")
                return

        if decision.action == "TYPE":
            text = str(decision.parameters.get("text", "")).strip()
            if text:
                self._append_completed_subgoal(memory, f"typed_text:{text[:40]}")
                memory.current_subgoal = "确认输入结果并寻找提交入口"
                self._append_note(memory, f"最近输入文本: {text[:80]}")
                return

        if decision.action == "CLICK":
            memory.current_subgoal = self._infer_click_subgoal(decision, perception)
            self._append_note(memory, f"最近点击动作: {decision.parameters}")
            return

        if decision.action == "SCROLL":
            memory.current_subgoal = "继续探索页面并寻找新的可交互元素"
            self._append_note(memory, f"最近滚动页面: {decision.parameters}")
            return

        if decision.action == "COMPLETE":
            self._append_completed_subgoal(memory, "task_marked_complete")
            memory.current_subgoal = "任务已结束"
            self._append_note(memory, f"任务结束原因: {decision.thought[:120]}")
            return

        memory.current_subgoal = self._infer_current_subgoal(memory)

    def _update_forbidden_actions(self, memory: MemoryState, decision: PlannerDecision) -> None:
        """基于最新动作更新 forbidden_actions。"""
        signature = self._make_signature(decision.action, decision.parameters)
        recent_signatures = [
            self._make_signature(item.action, item.parameters)
            for item in memory.history[-self._RECENT_REPEAT_WINDOW:]
        ]
        if recent_signatures.count(signature) >= 2:
            self._add_forbidden_action(memory, signature)

        if decision.action == "CLICK" and self._detect_repeated_click_same_point(memory):
            self._add_forbidden_action(memory, signature)

        if decision.action == "COMPLETE" and self._detect_too_many_completes(memory):
            self._add_forbidden_action(memory, signature)

    def _update_error_state(
        self,
        memory: MemoryState,
        perception: ScreenPerception,
        decision: PlannerDecision,
    ) -> None:
        """把最新一步中暴露出的风险沉淀为错误记录。"""
        trigger_step = len(memory.history)

        if self._detect_exact_repeated_loop(memory):
            self._append_error_record(
                memory,
                error_type="repeated_loop",
                trigger_step=trigger_step,
                message="最近三步动作完全相同。",
                recovery_hint="下一步改用不同策略，不要继续重复当前动作。",
            )

        if self._detect_repeated_click_same_point(memory):
            self._append_error_record(
                memory,
                error_type="repeated_click",
                trigger_step=trigger_step,
                message="最近连续点击了同一坐标。",
                recovery_hint="优先改为滚动、返回或选择其他元素。",
            )

        if self._detect_oscillation(memory):
            self._append_error_record(
                memory,
                error_type="oscillation",
                trigger_step=trigger_step,
                message="最近动作出现了来回震荡。",
                recovery_hint="避免走回头路，下一步尝试不同类型的动作。",
            )

        if self._detect_scroll_no_progress(memory, perception):
            self._append_error_record(
                memory,
                error_type="scroll_no_progress",
                trigger_step=trigger_step,
                message="连续滚动后页面变化很小。",
                recovery_hint="优先检查弹窗、输入框或固定按钮，而不是继续盲滚。",
            )

        if decision.action == "COMPLETE" and self._detect_too_many_completes(memory):
            self._append_error_record(
                memory,
                error_type="planner_stalling",
                trigger_step=trigger_step,
                message="近期多次输出 COMPLETE，可能处于保守停滞。",
                recovery_hint="复盘最近失败原因，避免下一轮过早结束任务。",
            )

    @staticmethod
    def _build_step_record(
        perception: ScreenPerception,
        decision: PlannerDecision,
        step_index: int,
    ) -> StepRecord:
        """构建标准化的单步记录。"""
        return StepRecord(
            step_index=step_index,
            action=MemoryReflectionModule._normalize_action(decision.action),
            parameters=dict(decision.parameters),
            thought=decision.thought,
            screen_summary=perception.screen_summary,
            success=None,
            error_type=MemoryReflectionModule._infer_step_error_type(decision),
            note=MemoryReflectionModule._build_step_note(perception, decision),
        )

    @staticmethod
    def _infer_step_error_type(decision: PlannerDecision) -> str:
        """尽量从最终决策中推断显性错误标签。"""
        thought = decision.thought or ""
        if "安全检查未通过" in thought:
            return "safety_fallback"
        if "强制终止" in thought:
            return "forced_stop"
        return ""

    @staticmethod
    def _build_step_note(perception: ScreenPerception, decision: PlannerDecision) -> str:
        """为历史步骤生成简洁备注。"""
        parts: List[str] = []
        if perception.app_name:
            parts.append(f"app={perception.app_name}")
        if perception.page_type:
            parts.append(f"page={perception.page_type}")
        if decision.target_element_id:
            parts.append(f"target={decision.target_element_id}")
        if perception.screen_summary:
            parts.append(f"summary={perception.screen_summary[:80]}")
        return " | ".join(parts)

    @staticmethod
    def _detect_exact_repeated_loop(memory: MemoryState) -> bool:
        """检测最近三步是否完全重复。"""
        if len(memory.history) < 3:
            return False
        last_signatures = [
            MemoryReflectionModule._make_signature(step.action, step.parameters)
            for step in memory.history[-3:]
        ]
        return len(set(last_signatures)) == 1

    @staticmethod
    def _detect_repeated_click_same_point(memory: MemoryState) -> bool:
        """检测最近是否连续点击同一坐标。"""
        if len(memory.history) < 2:
            return False

        repeated = 0
        last_point: Optional[List[int]] = None
        for step in reversed(memory.history):
            if step.action != "CLICK":
                break
            point = MemoryReflectionModule._extract_click_point(step.parameters)
            if point is None:
                break
            if last_point is None:
                last_point = point
                repeated = 1
                continue
            if point != last_point:
                break
            repeated += 1
            if repeated >= 2:
                return True
        return False

    @staticmethod
    def _detect_scroll_no_progress(memory: MemoryState, perception: ScreenPerception) -> bool:
        """检测连续滚动后是否没有明显页面进展。"""
        if len(memory.history) < 2:
            return False

        recent_steps = memory.history[-2:]
        if not all(step.action == "SCROLL" for step in recent_steps):
            return False

        previous_summaries = [MemoryReflectionModule._normalize_text(step.screen_summary) for step in recent_steps]
        current_summary = MemoryReflectionModule._normalize_text(perception.screen_summary)
        summaries = [summary for summary in previous_summaries + [current_summary] if summary]
        return len(summaries) >= 2 and len(set(summaries)) == 1

    @staticmethod
    def _detect_oscillation(memory: MemoryState) -> bool:
        """检测最近四步是否出现 A-B-A-B 式往返。"""
        if len(memory.history) < 4:
            return False

        signatures = [
            MemoryReflectionModule._make_signature(step.action, step.parameters)
            for step in memory.history[-4:]
        ]
        return signatures[0] == signatures[2] and signatures[1] == signatures[3] and signatures[0] != signatures[1]

    @staticmethod
    def _detect_no_progress(memory: MemoryState, perception: ScreenPerception) -> bool:
        """检测最近多步是否停留在几乎相同的页面状态。"""
        if len(memory.history) < 3:
            return False

        recent_actions = [step.action for step in memory.history[-3:]]
        if all(action == "OPEN" for action in recent_actions):
            return False

        history_summaries = [MemoryReflectionModule._normalize_text(step.screen_summary) for step in memory.history[-3:]]
        current_summary = MemoryReflectionModule._normalize_text(perception.screen_summary)
        summaries = [summary for summary in history_summaries + [current_summary] if summary]
        return len(summaries) >= 3 and len(set(summaries)) == 1

    @staticmethod
    def _detect_too_many_completes(memory: MemoryState) -> bool:
        """检测近期是否频繁保守结束。"""
        recent = memory.history[-3:]
        return len(recent) >= 2 and sum(step.action == "COMPLETE" for step in recent) >= 2

    @staticmethod
    def _detect_weak_perception(perception: ScreenPerception) -> bool:
        """检测感知是否过弱，供 C/E 保守处理。"""
        warning_text = " ".join(perception.warnings).lower()
        if "vlm_unavailable" in warning_text or "json_extract_failed" in warning_text or "json_parse_failed" in warning_text:
            return True
        if perception.page_type == "unknown" and not perception.elements:
            return True
        if not perception.screen_summary and len(perception.elements) <= 1:
            return True
        return False

    @staticmethod
    def _extend_blocked_signatures(
        blocked_action_signatures: List[str],
        memory: MemoryState,
        risk_flags: List[str],
    ) -> List[str]:
        """在严重风险下，把最近危险动作显式透传给 C。"""
        blocked = list(blocked_action_signatures)
        if not memory.history:
            return blocked

        last_signature = MemoryReflectionModule._make_signature(
            memory.history[-1].action,
            memory.history[-1].parameters,
        )
        if any(flag in {"repeated_loop", "repeated_click", "oscillation"} for flag in risk_flags):
            MemoryReflectionModule._append_unique(blocked, last_signature)
        return blocked

    @staticmethod
    def _append_error_record(
        memory: MemoryState,
        error_type: str,
        trigger_step: int,
        message: str,
        recovery_hint: str = "",
    ) -> None:
        """避免重复写入同一步的相同错误。"""
        for item in memory.error_records:
            if item.error_type == error_type and item.trigger_step == trigger_step:
                return
        memory.error_records.append(
            ErrorRecord(
                error_type=error_type,
                trigger_step=trigger_step,
                message=message,
                recovery_hint=recovery_hint,
            )
        )

    @staticmethod
    def _append_completed_subgoal(memory: MemoryState, subgoal: str) -> None:
        """按唯一值追加已完成子目标。"""
        if subgoal and subgoal not in memory.completed_subgoals:
            memory.completed_subgoals.append(subgoal)

    @staticmethod
    def _append_note(memory: MemoryState, note: str) -> None:
        """追加简短备注，避免完全重复。"""
        note = note.strip()
        if not note:
            return
        if note not in memory.notes:
            memory.notes.append(note)

    @staticmethod
    def _add_forbidden_action(memory: MemoryState, signature: str) -> None:
        """安全地写入 forbidden_actions。"""
        signature = signature.strip()
        if signature and signature not in memory.forbidden_actions:
            memory.forbidden_actions.append(signature)

    @staticmethod
    def _append_unique(items: List[str], value: str) -> None:
        """向列表中按顺序追加唯一值。"""
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _add_risk_if(
        condition: bool,
        risk_flag: str,
        advice: str,
        risk_flags: List[str],
        recovery_advice: List[str],
    ) -> None:
        """按条件追加风险标签与恢复建议。"""
        if not condition:
            return
        if risk_flag not in risk_flags:
            risk_flags.append(risk_flag)
        if advice not in recovery_advice:
            recovery_advice.append(advice)

    @staticmethod
    def _infer_current_subgoal(memory: MemoryState) -> str:
        """根据历史粗略推断当前子目标。"""
        if not memory.history:
            return "开始任务并识别当前页面"

        last_step = memory.history[-1]
        if last_step.action == "OPEN":
            return "确认目标应用是否已打开"
        if last_step.action == "TYPE":
            return "确认输入结果并尝试提交"
        if last_step.action == "SCROLL":
            return "继续探索页面中的目标元素"
        if last_step.action == "CLICK":
            return "确认点击结果并判断是否进入下一阶段"
        if last_step.action == "COMPLETE":
            return "任务已结束"
        return "继续根据当前页面推进任务"

    @staticmethod
    def _infer_click_subgoal(
        decision: PlannerDecision,
        perception: ScreenPerception,
    ) -> str:
        """从点击动作和页面语义中猜测当前子目标。"""
        text = f"{decision.thought} {perception.screen_summary}".lower()
        if "搜索" in text:
            return "确认搜索入口是否被激活"
        if "评论" in text or "回复" in text:
            return "确认输入框或评论入口是否出现"
        if "提交" in text or "发送" in text or "发布" in text:
            return "确认操作是否已提交"
        return "确认点击是否带来了有效页面变化"

    @staticmethod
    def _extract_click_point(parameters: Dict[str, Any]) -> Optional[List[int]]:
        """从参数中提取 CLICK 坐标。"""
        point = parameters.get("point")
        if isinstance(point, list) and len(point) == 2:
            return [int(point[0]), int(point[1])]
        if isinstance(point, tuple) and len(point) == 2:
            return [int(point[0]), int(point[1])]
        return None

    @staticmethod
    def _normalize_action(action: str) -> str:
        """统一动作名格式。"""
        return str(action or "").strip().upper()

    @staticmethod
    def _normalize_text(value: Any) -> str:
        """把文本压缩成便于比较的形式。"""
        if value is None:
            return ""
        return "".join(str(value).strip().lower().split())

    @staticmethod
    def _make_signature(action: str, parameters: Dict[str, Any]) -> str:
        """用统一动作名生成可比较签名。"""
        return action_signature(MemoryReflectionModule._normalize_action(action), dict(parameters or {}))

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        """尽量把输入转成 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_bool_or_none(value: Any) -> Optional[bool]:
        """把松散输入恢复成 bool/None。"""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "ok", "success"}:
                return True
            if normalized in {"false", "0", "no", "fail", "failed"}:
                return False
        return bool(value)

    @staticmethod
    def _rebuild_history(history_actions: List[Dict[str, object]]) -> List[StepRecord]:
        """把 TestRunner 的历史动作恢复为统一 StepRecord。"""
        rebuilt: List[StepRecord] = []
        for item in history_actions:
            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}

            rebuilt.append(
                StepRecord(
                    step_index=MemoryReflectionModule._to_int(item.get("step"), len(rebuilt) + 1),
                    action=MemoryReflectionModule._normalize_action(item.get("action", "")),
                    parameters=dict(params),
                    thought=str(item.get("raw_output") or item.get("thought") or ""),
                    screen_summary=str(item.get("screen_summary") or item.get("page_summary") or ""),
                    success=MemoryReflectionModule._to_bool_or_none(item.get("success")),
                    error_type=str(item.get("error_type") or ""),
                    note=str(item.get("note") or ""),
                )
            )
        return rebuilt
