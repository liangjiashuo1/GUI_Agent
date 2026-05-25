"""
B：Perception 工作目录入口。

E 模块主循环会按顺序调用 B 的这些函数：
1. build_perception_prompt(input_data, memory)
2. perceive_screen(input_data, memory, call_llm)

其中：
- build_perception_prompt 负责组织发给多模态模型的输入
- perceive_screen 负责输出结构化的 ScreenPerception
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from agent_base import AgentInput
from gui_agent.shared.schemas import MemoryState, ScreenPerception


class PerceptionModule:
    """B 模块：负责看懂当前屏幕。"""

    def build_perception_prompt(self, input_data: AgentInput, memory: MemoryState) -> List[Dict[str, Any]]:
        """
        构建多模态感知阶段的提示词或消息列表。

        上层调用位置：
- E.run_step 的第 2 步

        Returns:
- list[dict]: 推荐直接返回多模态消息数组
- 每个 dict 可以包含 role / content 等字段
- 当前骨架允许返回空列表，表示暂未接入真实提示词
        """
        return []

    def perceive_screen(
        self,
        input_data: AgentInput,
        memory: MemoryState,
        call_llm: Callable[..., Any] | None = None,
    ) -> ScreenPerception:
        """
        对当前截图做结构化理解，并返回 ScreenPerception。

        上层调用位置：
- E.run_step 的第 3 步

        Returns:
- app_name: 当前推测的应用名称
- page_type: 页面类型
- screen_summary: 对当前屏幕的简洁说明
- elements: 识别出的可交互元素列表
- keyboard_visible: 是否有输入法
- scrollable: 页面是否可滚动
- warnings: 感知阶段发现的风险或不确定项
- raw_model_output: 原始模型输出字符串
        """
        _ = self.build_perception_prompt(input_data, memory)
        return ScreenPerception(
            app_name=self._guess_app_name(input_data.instruction),
            page_type="unknown",
            screen_summary="当前已拿到截图，但屏幕理解模块尚未接入具体视觉模型。",
            elements=[],
            keyboard_visible=False,
            scrollable=True,
            warnings=["perception_not_implemented"],
            raw_model_output="",
        )

    def parse_perception_response(self, raw_text: str) -> ScreenPerception:
        """
        将多模态模型原始输出解析成 ScreenPerception。

        Returns:
- ScreenPerception: 字段与 perceive_screen 的返回要求一致
        """
        return ScreenPerception(
            page_type="unknown",
            screen_summary=raw_text.strip(),
            elements=[],
            raw_model_output=raw_text,
        )

    @staticmethod
    def _guess_app_name(instruction: str) -> str:
        known_apps = [
            "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手", "芒果TV",
            "美团", "腾讯视频", "喜马拉雅", "QQ", "淘宝", "微信",
        ]
        for app in known_apps:
            if app in instruction:
                return app
        return ""

