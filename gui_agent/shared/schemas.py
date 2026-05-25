"""
共享接口契约文件。

建议由 E 负责人优先维护本文件，A/B/C/D 的实现都以这里的数据结构为准。

主循环中的标准数据流如下：
1. AgentInput
2. MemoryState
3. ScreenPerception
4. ReflectionSignal
5. PlannerDecision
6. SafetyCheckResult
7. AgentOutput

统一约定：
- 所有屏幕坐标使用归一化坐标，范围是 [0, 1000]
- 元素框 bbox 统一使用 [x1, y1, x2, y2]
- action 必须是 CLICK / SCROLL / TYPE / OPEN / COMPLETE 之一
- parameters 必须严格符合 agent_base.py 的标准格式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


NormalizedPoint = Tuple[int, int]
ActionName = Literal["CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"]
ElementRole = Literal["button", "input", "icon", "tab", "list_item", "text", "image", "other"]


@dataclass
class UIElement:
    """单个 UI 元素的结构化表示。"""

    element_id: str
    role: ElementRole
    text: str = ""
    description: str = ""
    bbox: List[int] = field(default_factory=list)
    clickable: bool = False
    enabled: bool = True
    confidence: float = 0.0


@dataclass
class ScreenPerception:
    """
    B 模块输出。

    字段说明：
- app_name: 当前推测的应用名
- page_type: 当前页面类型，如 home / search / detail / popup / unknown
- screen_summary: 对当前屏幕的简洁总结
- elements: 识别出的 UI 元素列表
- keyboard_visible: 是否有输入法
- scrollable: 当前页面是否可滚动
- warnings: 感知阶段发现的问题
- raw_model_output: 原始模型输出，便于日志回放
    """

    app_name: str = ""
    page_type: str = "unknown"
    screen_summary: str = ""
    elements: List[UIElement] = field(default_factory=list)
    focused_element_id: Optional[str] = None
    keyboard_visible: bool = False
    scrollable: bool = True
    warnings: List[str] = field(default_factory=list)
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    raw_model_output: str = ""


@dataclass
class ReflectionSignal:
    """
    D 模块的反思输出。

    字段说明：
- need_backoff: 是否建议本轮不要继续重复尝试
- risk_flags: 风险标签列表，如 repeated_loop / no_progress
- recovery_advice: 给 C/E 的恢复建议
- blocked_action_signatures: 当前不建议再执行的动作签名
    """

    need_backoff: bool = False
    risk_flags: List[str] = field(default_factory=list)
    recovery_advice: List[str] = field(default_factory=list)
    blocked_action_signatures: List[str] = field(default_factory=list)


@dataclass
class PlannerDecision:
    """
    C 模块输出，表示当前轮最终建议动作。

    字段说明：
- action: 标准动作名
- parameters: 标准动作参数
- thought: 人类可读的理由说明
- target_element_id: 若针对某个元素，则填写对应 element_id
- confidence: 当前动作的置信度
- is_terminal: 是否打算结束任务
    """

    action: ActionName
    parameters: Dict[str, Any]
    thought: str
    target_element_id: Optional[str] = None
    confidence: float = 0.0
    is_terminal: bool = False


@dataclass
class StepRecord:
    """D/E 维护的单步记录。"""

    step_index: int
    action: str
    parameters: Dict[str, Any]
    thought: str = ""
    screen_summary: str = ""
    success: Optional[bool] = None
    error_type: str = ""
    note: str = ""


@dataclass
class ErrorRecord:
    """D 模块维护的错误记录。"""

    error_type: str
    trigger_step: int
    message: str
    recovery_hint: str = ""


@dataclass
class MemoryState:
    """
    D 模块维护的任务记忆。

    字段说明：
- task_goal: 原始任务目标
- current_subgoal: 当前子目标
- history: 历史步骤
- completed_subgoals: 已完成的子目标
- error_records: 失败或异常记录
- forbidden_actions: 不建议继续尝试的动作签名
- notes: 自由备注
    """

    task_goal: str
    current_subgoal: str = ""
    history: List[StepRecord] = field(default_factory=list)
    completed_subgoals: List[str] = field(default_factory=list)
    error_records: List[ErrorRecord] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class SafetyCheckResult:
    """
    E 模块安全检查输出。

    字段说明：
- ok: 是否允许执行
- reason: 不允许执行时的原因
- sanitized_decision: 如进行了修正，这里返回修正后的动作
    """

    ok: bool
    reason: str = ""
    sanitized_decision: Optional[PlannerDecision] = None


def clamp_point(point: NormalizedPoint) -> NormalizedPoint:
    """将坐标限制在 [0, 1000]。"""
    return max(0, min(1000, point[0])), max(0, min(1000, point[1]))


def action_signature(action: str, parameters: Dict[str, Any]) -> str:
    """将动作和参数转成可比较签名。"""
    return f"{action}:{parameters}"

