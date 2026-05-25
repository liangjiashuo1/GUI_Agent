"""
A：Executor 工作目录入口。

E 模块主循环会调用 A 的两个核心函数：
1. build_device_command(decision)
2. compile_decision(decision)

其中：
- build_device_command 用于真机执行场景的命令描述生成
- compile_decision 用于当前离线评测场景，最终返回 AgentOutput
"""

from __future__ import annotations

from typing import Any, Dict

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE, AgentOutput
from gui_agent.shared.schemas import PlannerDecision, clamp_point


class ExecutorModule:
    """A 模块：负责将标准决策转换成可执行结果。"""

    def build_device_command(self, decision: PlannerDecision) -> Dict[str, Any]:
        """
        根据 C 模块给出的 PlannerDecision，构建真机执行层可消费的命令描述。

        上层调用位置：
- E.run_step 中，在最终返回前可选调用

        Returns:
- executor: 执行器名称，如 mock / adb / appium
- action: 设备层动作名，如 click / scroll / type / open / complete
- payload: 设备层需要的动作参数
        """
        return {
            "executor": "mock",
            "action": decision.action.lower(),
            "payload": dict(decision.parameters),
        }

    def compile_decision(self, decision: PlannerDecision) -> AgentOutput:
        """
        将 PlannerDecision 转成当前评测框架要求的 AgentOutput。

        上层调用位置：
- E.run_step 的最后一步

        Returns:
- action: 标准动作名，必须是 CLICK / SCROLL / TYPE / OPEN / COMPLETE
- parameters: 标准动作参数，必须与 agent_base.py 完全一致
- raw_output: 当前轮的 thought，便于日志和回放
        """
        action = decision.action
        parameters = self._sanitize_parameters(action, decision.parameters)
        return AgentOutput(action=action, parameters=parameters, raw_output=decision.thought)

    def _sanitize_parameters(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """对标准动作参数做最后一层校验和兜底。"""
        if action == ACTION_CLICK:
            point = parameters.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("CLICK 参数必须是 {'point': [x, y]}")
            x, y = clamp_point((int(point[0]), int(point[1])))
            return {"point": [x, y]}

        if action == ACTION_SCROLL:
            start_point = parameters.get("start_point")
            end_point = parameters.get("end_point")
            if not self._is_point(start_point) or not self._is_point(end_point):
                raise ValueError("SCROLL 参数必须是 {'start_point': [x1, y1], 'end_point': [x2, y2]}")
            sx, sy = clamp_point((int(start_point[0]), int(start_point[1])))
            ex, ey = clamp_point((int(end_point[0]), int(end_point[1])))
            return {"start_point": [sx, sy], "end_point": [ex, ey]}

        if action == ACTION_TYPE:
            text = parameters.get("text")
            if not isinstance(text, str):
                raise ValueError("TYPE 参数必须是 {'text': '...'}")
            return {"text": text}

        if action == ACTION_OPEN:
            app_name = parameters.get("app_name")
            if not isinstance(app_name, str) or not app_name.strip():
                raise ValueError("OPEN 参数必须是 {'app_name': '...'}")
            return {"app_name": app_name.strip()}

        if action == ACTION_COMPLETE:
            return {}

        raise ValueError(f"未知动作类型: {action}")

    @staticmethod
    def _is_point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2

