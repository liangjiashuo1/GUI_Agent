"""
C：Planner 工作目录入口。

E 模块主循环会按顺序调用 C 的这些函数：
1. build_planner_prompt(input_data, perception, memory, reflection)
2. plan_next_action(input_data, perception, memory, reflection, call_llm)

核心机制：VLM 选择 target_id，C 负责解析为精确坐标。
B 模块提供 OCR 精确 bbox 的元素列表 → C 让 VLM 选 target_id → C 算中心坐标。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from agent_base import (
    ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE,
    AgentInput,
)
from gui_agent.shared.schemas import (
    MemoryState, PlannerDecision, ReflectionSignal, ScreenPerception, UIElement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner prompt 模板
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """你是一个手机GUI自动化规划器。根据任务目标、当前屏幕内容和历史操作，决定下一步执行什么动作。

## 可用动作
- CLICK(target_id): 点击某个元素，target_id 从下方元素列表中选择
- SCROLL(direction): 滑动屏幕，direction 为 up/down/left/right
- TYPE(text): 在当前输入框输入文本
- OPEN(app_name): 打开应用
- COMPLETE(): 任务已经完成，不需要再操作

## 输出格式（严格 JSON）
```json
{
    "thought": "分析当前状态和选择此动作的原因",
    "action": "CLICK",
    "target_id": "ocr_3",
    "text": "",
    "direction": "",
    "confidence": 0.85
}
```

## 决策规则
1. 如果当前不在目标 App 内 → OPEN
2. 如果屏幕上有广告弹窗/更新提示 → CLICK 关闭或跳过按钮
3. 如果需要搜索：点击搜索框 → TYPE 关键词 → CLICK 搜索结果中匹配的项
4. 如果要发评论：先找"评论""讨论""写评论"入口 CLICK → 点击输入框 → TYPE 评论内容 → CLICK 发送
5. 如果目标元素不在当前屏幕且页面可滚动 → SCROLL
6. 如果输入框已激活（键盘可见）且需要输入文字 → TYPE
7. 历史中已经失败过的操作不要重复
8. 确认所有子任务（打开App、搜索、输入文字、提交）都完成后才能 COMPLETE
9. target_id 必须从下方元素列表中选取，严禁编造"""

_PLANNER_USER_TEMPLATE = """## 任务目标
{instruction}

## 当前屏幕状态
- 应用: {app_name}
- 页面类型: {page_type}
- 描述: {screen_summary}
- 键盘可见: {keyboard_visible}
- 可滚动: {scrollable}
- 当前第 {step_count} 步

## 可交互元素（共 {element_count} 个）
{elements_list}

## 历史操作
{history_text}

## 注意事项
{reflection_text}

请输出下一步动作的 JSON。"""


class PlannerModule:
    """C 模块：负责决定下一步动作（target_id → 坐标解析）。"""

    # ------------------------------------------------------------------
    # 公开接口（E 模块调用）
    # ------------------------------------------------------------------

    def build_planner_prompt(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        reflection: ReflectionSignal,
    ) -> List[Dict[str, Any]]:
        """构建规划阶段的提示词消息列表。

        上层调用位置：E.run_step 的第 5 步
        """
        elements_text = self._format_elements_for_prompt(perception.elements)
        history_text = self._format_history(memory)
        reflection_text = self._format_reflection(reflection)

        user_text = _PLANNER_USER_TEMPLATE.format(
            instruction=input_data.instruction,
            app_name=perception.app_name or "未知",
            page_type=perception.page_type,
            screen_summary=perception.screen_summary or "无",
            keyboard_visible=perception.keyboard_visible,
            scrollable=perception.scrollable,
            step_count=input_data.step_count,
            element_count=len(perception.elements),
            elements_list=elements_text,
            history_text=history_text,
            reflection_text=reflection_text,
        )

        return [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def plan_next_action(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        reflection: ReflectionSignal,
        call_llm: Callable[..., Any] | None = None,
    ) -> PlannerDecision:
        """根据任务、感知结果和反思信号，给出当前轮最终动作。

        上层调用位置：E.run_step 的第 6 步
        """
        # ---- 硬约束检查（不消耗 VLM 调用）----

        # 反思要求回退
        if reflection.need_backoff:
            advice = "；".join(reflection.recovery_advice) if reflection.recovery_advice else "检测到风险，保守结束。"
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={}, thought=advice,
                confidence=0.1, is_terminal=True,
            )

        # 还没打开目标 App
        if perception.app_name and not self._has_opened_app(memory):
            return PlannerDecision(
                action=ACTION_OPEN,
                parameters={"app_name": perception.app_name},
                thought=f"当前不在目标应用中，先打开「{perception.app_name}」。",
                confidence=0.5, is_terminal=False,
            )

        # ---- 调用 VLM 做真正的规划 ----
        if call_llm is not None:
            decision = self._plan_with_llm(input_data, perception, memory, reflection, call_llm)
        else:
            # 无 VLM：保守结束
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought="规划模型不可用，保守结束。", confidence=0.1, is_terminal=True,
            )

        # ---- 后处理：硬约束修正 ----
        decision = self._enforce_hard_constraints(decision, input_data, perception, memory)
        return decision

    def parse_planner_response(self, raw_text: str) -> PlannerDecision:
        """将 VLM 原始输出解析成 PlannerDecision。"""
        json_str = self._extract_json(raw_text)
        if json_str is None:
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought=raw_text.strip()[:300] or "VLM 返回格式异常，保守结束",
                confidence=0.0, is_terminal=True,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought=json_str[:300], confidence=0.0, is_terminal=True,
            )

        if not isinstance(data, dict):
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought=str(data)[:300], confidence=0.0, is_terminal=True,
            )

        action = str(data.get("action", "COMPLETE")).upper().strip()
        if action not in ("CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"):
            action = ACTION_COMPLETE

        # 从 VLM JSON 中提取参数（text/direction 通常在顶层，不在 parameters 子对象中）
        params: Dict[str, Any] = {}
        raw_params = data.get("parameters")
        if isinstance(raw_params, dict):
            params.update(raw_params)
        for key in ("text", "direction", "app_name"):
            val = data.get(key)
            if val and key not in params:
                params[key] = str(val)

        return PlannerDecision(
            action=action,
            parameters=params,
            thought=str(data.get("thought", "")),
            target_element_id=str(data.get("target_id")) if data.get("target_id") else None,
            confidence=float(data.get("confidence", 0.5)),
            is_terminal=(action == ACTION_COMPLETE),
        )

    # ------------------------------------------------------------------
    # VLM 规划 + target_id 坐标解析
    # ------------------------------------------------------------------

    def _plan_with_llm(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        reflection: ReflectionSignal,
        call_llm: Callable[..., Any],
    ) -> PlannerDecision:
        """调用 VLM 做动作决策，并解析 target_id 为精确坐标。"""
        messages = self.build_planner_prompt(input_data, perception, memory, reflection)

        try:
            response = call_llm(messages)
            raw_text = response.choices[0].message.content
        except Exception as exc:
            logger.warning("Planner VLM 调用失败: %s", exc)
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought=f"VLM 调用异常: {exc}", confidence=0.0, is_terminal=True,
            )

        if not raw_text:
            return PlannerDecision(
                action=ACTION_COMPLETE, parameters={},
                thought="VLM 返回空内容", confidence=0.0, is_terminal=True,
            )

        decision = self.parse_planner_response(raw_text)

        # ---- 核心：target_id → 坐标解析 ----
        decision = self._resolve_action_params(decision, perception.elements)

        return decision

    def _resolve_action_params(
        self, decision: PlannerDecision, elements: List[UIElement]
    ) -> PlannerDecision:
        """将 VLM 返回的 target_id / direction / text 解析为标准动作参数。

        - CLICK: target_id → bbox → 中心坐标 → {"point": [x, y]}
        - SCROLL: direction → {"start_point": [...], "end_point": [...]}
        - TYPE: text → {"text": "..."}
        - OPEN: 保持原样
        - COMPLETE: {}
        """
        params: Dict[str, Any] = {}

        if decision.action == ACTION_CLICK:
            target_id = decision.target_element_id
            if target_id:
                point = self._element_to_center(target_id, elements)
                if point is not None:
                    params["point"] = point
                else:
                    # target_id 无效，降级为 COMPLETE
                    logger.warning("target_id '%s' 未在元素列表中找到，降级为 COMPLETE", target_id)
                    return PlannerDecision(
                        action=ACTION_COMPLETE, parameters={},
                        thought=f"VLM 选择的 target_id '{target_id}' 不存在于元素列表中，保守结束",
                        confidence=0.0, is_terminal=True,
                    )
            else:
                # CLICK 但无 target_id，尝试从 parameters 中取坐标（兼容旧格式）
                if "point" in decision.parameters:
                    params["point"] = decision.parameters["point"]
                else:
                    logger.warning("CLICK 动作缺少 target_id 和 point，降级为 COMPLETE")
                    return PlannerDecision(
                        action=ACTION_COMPLETE, parameters={},
                        thought="CLICK 动作缺少有效的目标标识", confidence=0.0, is_terminal=True,
                    )

        elif decision.action == ACTION_SCROLL:
            direction = (decision.parameters.get("direction") or "").lower()
            if not direction:
                direction = "down"  # 默认向下滑动
            params = self._direction_to_scroll_params(direction)

        elif decision.action == ACTION_TYPE:
            text = decision.parameters.get("text") or ""
            if not text:
                logger.warning("TYPE 动作缺少 text")
            params["text"] = str(text)

        elif decision.action == ACTION_OPEN:
            app_name = decision.parameters.get("app_name") or ""
            if not app_name:
                app_name = decision.thought or ""
            params["app_name"] = str(app_name)

        # 更新 decision 的 parameters
        decision.parameters = params
        return decision

    def _enforce_hard_constraints(
        self,
        decision: PlannerDecision,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
    ) -> PlannerDecision:
        """对 VLM 决策做硬约束修正，防止关键步骤被跳过。"""
        instruction = input_data.instruction

        # 1. 提取任务中的关键词
        required_text = self._extract_required_text(instruction)
        search_keyword = self._extract_search_keyword(instruction)

        # 2. 检查历史中已完成的 TYPE
        has_typed_required = False
        has_typed_search = False
        for step in memory.history:
            if step.action == ACTION_TYPE:
                typed = str(step.parameters.get("text", ""))
                if required_text and (required_text in typed or typed in required_text):
                    has_typed_required = True
                if search_keyword and (search_keyword in typed or typed in search_keyword):
                    has_typed_search = True

        # 3. 搜索页 + 键盘可见 + 搜索词未输入 → 强制 TYPE 搜索关键词
        if (perception.page_type == "search" and search_keyword
                and not has_typed_search and perception.keyboard_visible):
            logger.info("搜索页面且关键词未输入，强制 TYPE: %s", search_keyword)
            return PlannerDecision(
                action=ACTION_TYPE,
                parameters={"text": search_keyword},
                thought=f"搜索页面已激活，输入搜索关键词「{search_keyword}」",
                confidence=0.95, is_terminal=False,
            )

        # 搜索词已经输入后，优先点击页面顶部的搜索提交按钮。
        # 一些模型会点键盘右下角搜索键，但离线 checker 通常只认应用内顶部按钮。
        if (perception.page_type == "search" and search_keyword and has_typed_search
                and decision.action == ACTION_CLICK):
            point = decision.parameters.get("point") if isinstance(decision.parameters, dict) else None
            if isinstance(point, list) and len(point) == 2 and point[1] >= 800:
                submit_point = self._find_top_search_submit_point(perception.elements)
                logger.info("搜索词已输入，但点击落在键盘区域，改点顶部搜索提交按钮: %s", submit_point)
                return PlannerDecision(
                    action=ACTION_CLICK,
                    parameters={"point": submit_point},
                    thought="搜索词已输入，点击页面顶部搜索按钮提交查询",
                    confidence=0.95, is_terminal=False,
                )

        # 4. VLM 想 COMPLETE 但要求的文字还没输入 → 拦截，改为 TYPE
        if decision.action == ACTION_COMPLETE and required_text and not has_typed_required:
            logger.info("VLM 想 COMPLETE 但文字未输入，拦截改为 TYPE: %s", required_text)
            return PlannerDecision(
                action=ACTION_TYPE,
                parameters={"text": required_text},
                thought=f"任务要求输入「{required_text}」，尚未完成，先输入文字",
                confidence=0.9, is_terminal=False,
            )

        return decision

    @staticmethod
    def _extract_required_text(instruction: str) -> str:
        """从任务指令中提取必须输入的文字。"""
        # 匹配 "发布评论：XXX" / "输入XXX" / "评论：XXX"
        m = re.search(r'(?:发布评论|评论|输入|回复)[:：]\s*[""](.+?)[""]?\s*$', instruction)
        if m:
            return m.group(1).strip()
        # 匹配 "发布评论XXX"
        m = re.search(r'(?:发布评论|评论|输入|回复)\s*[""](.+?)[""]?\s*$', instruction)
        if m:
            return m.group(1).strip()
        # 匹配结尾的 "：XXX" 或 ":XXX"
        m = re.search(r'[：:]\s*([^\s，。,\.]{2,30})$', instruction)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_search_keyword(instruction: str) -> str:
        """从任务指令中提取搜索关键词。"""
        # 匹配 "搜索XXX" / "打开XXX的" / "搜XXX"
        m = re.search(r'搜索[""](.+?)[""]', instruction)
        if m:
            return m.group(1).strip()
        m = re.search(r'打开(.+?)的', instruction)
        if m:
            return m.group(1).strip()
        m = re.search(r'搜\s*[""](.+?)[""]', instruction)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _element_to_center(target_id: str, elements: List[UIElement]) -> Optional[List[int]]:
        """根据 target_id 查找元素，返回 bbox 中心坐标 [cx, cy]。"""
        for elem in elements:
            if elem.element_id == target_id and elem.bbox and len(elem.bbox) == 4:
                cx = (elem.bbox[0] + elem.bbox[2]) // 2
                cy = (elem.bbox[1] + elem.bbox[3]) // 2
                return [cx, cy]
        return None

    @staticmethod
    def _find_top_search_submit_point(elements: List[UIElement]) -> List[int]:
        """查找顶部右侧搜索提交按钮，找不到时给出常见右上角兜底坐标。"""
        keywords = ("搜索", "确定", "完成")
        candidates: List[UIElement] = []

        for elem in elements:
            if not elem.bbox or len(elem.bbox) != 4:
                continue
            x1, y1, x2, y2 = elem.bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = f"{elem.text or ''} {elem.description or ''} {elem.role or ''}"
            if cy <= 180 and cx >= 650 and any(word in label for word in keywords):
                candidates.append(elem)

        if candidates:
            # 右侧、偏上的按钮通常是搜索提交入口。
            best = max(candidates, key=lambda elem: ((elem.bbox[0] + elem.bbox[2]) // 2, -elem.bbox[1]))
            return [(best.bbox[0] + best.bbox[2]) // 2, (best.bbox[1] + best.bbox[3]) // 2]

        return [910, 75]

    @staticmethod
    def _direction_to_scroll_params(direction: str) -> Dict[str, Any]:
        """将方向文字转为归一化坐标的滑动参数。"""
        direction = direction.lower()
        if direction == "up":
            return {"start_point": [500, 700], "end_point": [500, 300]}
        elif direction == "down":
            return {"start_point": [500, 300], "end_point": [500, 700]}
        elif direction == "left":
            return {"start_point": [700, 500], "end_point": [300, 500]}
        else:  # right
            return {"start_point": [300, 500], "end_point": [700, 500]}

    # ------------------------------------------------------------------
    # prompt 格式化工具
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_task_progress(instruction: str, memory: MemoryState) -> str:
        """分析任务进度：检查历史操作，总结已完成和待完成的步骤。"""
        if not memory.history:
            return "任务尚未开始，这是第一步。"

        parts: List[str] = []
        has_open = False
        has_type = False
        typed_texts: List[str] = []
        has_complete = False

        for step in memory.history:
            if step.action == "OPEN":
                has_open = True
            if step.action == "TYPE":
                has_type = True
                text = str(step.parameters.get("text", ""))
                if text:
                    typed_texts.append(text)

        # 提取任务中要求输入的文字
        text_to_type = ""
        m = re.search(r'(?:输入|发布评论[:：]|评论[:：]|回复[:：])\s*["“]?(.+?)["”]?\s*$', instruction)
        if not m:
            m = re.search(r'[：:]\s*([^\s]{2,20})$', instruction)
        if m:
            text_to_type = m.group(1).strip()

        if has_open:
            parts.append("已打开应用")
        else:
            parts.append("尚未打开目标应用")

        if has_type:
            parts.append(f"已输入文字: {', '.join(typed_texts)}")
            if text_to_type and any(text_to_type in t or t in text_to_type for t in typed_texts):
                parts.append("任务要求的评论文本已输入")
            elif text_to_type:
                parts.append(f"⚠ 但尚未输入任务要求的完整文字: \"{text_to_type}\"")
        elif text_to_type:
            parts.append(f"⚠ 尚未输入任务要求的文字: \"{text_to_type}\"")

        return "\n".join(f"- {p}" for p in parts)

    @staticmethod
    def _format_elements_for_prompt(elements: List[UIElement]) -> str:
        """将元素列表格式化为 prompt 中的可读文本。"""
        if not elements:
            return "（无可用元素，可能需要先滑动屏幕或打开应用）"

        lines = []
        for elem in elements:
            bbox_str = f"[{elem.bbox[0]},{elem.bbox[1]},{elem.bbox[2]},{elem.bbox[3]}]" if elem.bbox else "未知"
            clickable = "可点击" if elem.clickable else "不可点击"
            desc = f" — {elem.description}" if elem.description else ""
            lines.append(
                f"  {elem.element_id}: \"{elem.text}\" "
                f"角色={elem.role} {clickable} bbox={bbox_str}{desc}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_history(memory: MemoryState) -> str:
        """格式化历史操作为文本。"""
        if not memory.history:
            return "（这是第一步，暂无历史操作）"

        recent = memory.history[-8:]  # 最近 8 步
        lines = []
        for step in recent:
            entry = f"  第{step.step_index}步: {step.action}"
            if step.parameters:
                entry += f" {step.parameters}"
            if step.thought:
                entry += f" // {step.thought}"
            if step.error_type:
                entry += f" [失败: {step.error_type}]"
            lines.append(entry)
        return "\n".join(lines)

    @staticmethod
    def _format_reflection(reflection: ReflectionSignal) -> str:
        """格式化反思信号为提示文本。"""
        parts = []
        if reflection.risk_flags:
            parts.append(f"- 风险提示: {', '.join(reflection.risk_flags)}")
        if reflection.recovery_advice:
            parts.append(f"- 恢复建议: {'; '.join(reflection.recovery_advice)}")
        if reflection.blocked_action_signatures:
            parts.append(f"- 禁止执行以下动作: {', '.join(reflection.blocked_action_signatures)}")
        return "\n".join(parts) if parts else "- 无特殊限制"

    # ------------------------------------------------------------------
    # JSON 提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[str]:
        """从模型原始输出中提取 JSON 字符串。"""
        # 策略 1: ```json ... ```
        m = re.search(r'```json\s*([\s\S]*?)```', raw_text)
        if m:
            return m.group(1).strip()
        # 策略 2: ``` ... ```
        m = re.search(r'```\s*([\s\S]*?)```', raw_text)
        if m:
            return m.group(1).strip()
        # 策略 3: 裸 JSON
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return raw_text[start:end + 1]
        return None

    # ------------------------------------------------------------------
    # 辅助判断
    # ------------------------------------------------------------------

    @staticmethod
    def _has_opened_app(memory: MemoryState) -> bool:
        """检查历史中是否已经执行过 OPEN 动作。"""
        return any(step.action == ACTION_OPEN for step in memory.history)
