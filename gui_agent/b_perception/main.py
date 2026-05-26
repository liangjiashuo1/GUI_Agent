"""
B：Perception 工作目录入口。

E 模块主循环会按顺序调用 B 的这些函数：
1. build_perception_prompt(input_data, memory)
2. perceive_screen(input_data, memory, call_llm)

混合方案：
- OCR 引擎提取文字 + 精确 bbox（定位准）
- VLM 做屏幕语义理解 + 元素角色标注（理解对）
- 后处理阶段融合：文字元素用 OCR bbox，非文字元素用 VLM bbox
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agent_base import AgentInput
from gui_agent.shared.schemas import MemoryState, ScreenPerception, UIElement
from utils.image_utils import encode_image_to_base64

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 感知 prompt 模板
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是一个专业的手机GUI屏幕分析助手。你的任务是仔细观察手机截图，并结合 OCR 引擎提供的文字识别结果，输出结构化的屏幕理解。

## 输出要求
必须严格输出以下JSON格式：

```json
{
    "app_name": "当前应用名称",
    "page_type": "home/search/detail/list/popup/settings/login/unknown",
    "screen_summary": "用一句话简洁描述当前屏幕显示了什么内容",
    "elements": [
        {
            "element_id": "元素的唯一标识",
            "role": "button/input/icon/tab/list_item/text/image/other",
            "text": "元素上显示的文字",
            "description": "该元素的功能说明，如'搜索按钮''底部导航首页Tab'",
            "bbox": [x1, y1, x2, y2],
            "clickable": true,
            "enabled": true,
            "confidence": 0.85
        }
    ],
    "keyboard_visible": false,
    "scrollable": false,
    "warnings": []
}
```

## 坐标系统
- 屏幕宽高均视为 1000 单位
- bbox 格式为 [左上x, 左上y, 右下x, 右下y]，四个值在 [0, 1000] 内
- 参考值：顶部状态栏 [0,0,1000,40] 标题栏 [0,40,1000,90] 底部导航 [0,930,1000,1000]

## OCR 辅助信息使用说明
- 系统已经用 OCR 引擎识别了截图中的文字和它们在屏幕上的精确位置
- OCR 结果（含精确 bbox）会在 user message 中列出
- 你需要做的：
  1. 基于截图和 OCR 结果，判断当前 App 名称、页面类型、屏幕摘要
  2. 为每个 OCR 识别到的文字标注其语义角色（button/input/tab/label等）和是否可点击
  3. 补充 OCR 未能识别的非文字元素（如图标 icon、图片 image），这些元素的 bbox 由你估算
  4. element_id 统一格式：ocr_0, ocr_1... 对应 OCR 元素，vlm_0, vlm_1... 对应你补充的元素

## 元素识别规范
- 文字元素的 bbox 直接沿用 OCR 提供的精确值，不要修改
- 只列出有交互价值或信息价值的元素，忽略纯装饰性内容
- 重点识别：搜索框、搜索按钮、底部导航Tab、确认/提交/发送按钮、返回/关闭按钮、弹窗/广告
- 识别不确定时，降低 confidence 并添加 warnings
- 宁缺毋滥：完全不确定的元素不要编造

## 常见 App 识别特征
- 爱奇艺：绿色主题，视频播放，底部 首页/随刻/会员/我的
- 抖音：黑色背景，短视频，底部 首页/朋友/消息/我
- 快手：橙色主题，短视频，底部 首页/发现/消息/我
- 百度地图：地图界面，底部 首页/出行/周边/我的
- 微信：绿色主题，聊天列表，底部 微信/通讯录/发现/我
- QQ：浅蓝主题，聊天，底部 消息/联系人/动态
- 美团：黄色主题，底部 首页/我的
- 淘宝：橙色主题，商品列表，底部 首页/购物车/消息/我的
- 腾讯视频：蓝黑主题，视频播放
- 喜马拉雅：橙红主题，音频播放
- 芒果TV：橙色主题，视频播放"""

_USER_TEMPLATE = """## 任务目标
{instruction}

## 执行进度
当前是第 {step_count} 步
{history_context}

## OCR 文字识别结果（共 {ocr_count} 个）
以下文字及其精确坐标由 OCR 引擎提供，请在 elements 中为它们标注语义角色：
{ocr_list}

请分析截图并结合 OCR 结果，输出 JSON 格式的屏幕理解结果。"""


class PerceptionModule:
    """B 模块：负责看懂当前屏幕（混合方案：OCR 精确定位 + VLM 语义理解）。"""

    def __init__(self, use_ocr: bool = True) -> None:
        self._use_ocr = use_ocr
        self._ocr_reader: Any = None
        self._ocr_init_failed = False
        # OCR 缓存：同一张图只跑一次 OCR
        self._ocr_cache: Optional[List[Dict[str, Any]]] = None
        self._ocr_cache_image_id: Optional[int] = None

    # ------------------------------------------------------------------
    # 公开接口（E 模块调用）
    # ------------------------------------------------------------------

    def build_perception_prompt(
        self, input_data: AgentInput, memory: MemoryState
    ) -> List[Dict[str, Any]]:
        """构建多模态感知提示词（含 OCR 结果）。

        上层调用位置：E.run_step 的第 2 步
        """
        ocr_elements = self._run_ocr(input_data.current_image)
        history_context = self._build_history_context(memory)
        ocr_list_str, ocr_count = self._format_ocr_list(ocr_elements)

        user_text = _USER_TEMPLATE.format(
            instruction=input_data.instruction,
            step_count=input_data.step_count,
            history_context=history_context,
            ocr_count=ocr_count,
            ocr_list=ocr_list_str,
        )
        image_url = encode_image_to_base64(input_data.current_image)

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]

    def perceive_screen(
        self,
        input_data: AgentInput,
        memory: MemoryState,
        call_llm: Callable[..., Any] | None = None,
    ) -> ScreenPerception:
        """对当前截图做结构化理解，并返回 ScreenPerception。

        上层调用位置：E.run_step 的第 3 步
        """
        # 先跑 OCR（与 VLM 调用无关）
        ocr_elements = self._run_ocr(input_data.current_image)

        if call_llm is None:
            return self._build_ocr_only_result(input_data, ocr_elements, "无法调用视觉模型")

        messages = self.build_perception_prompt(input_data, memory)

        try:
            response = call_llm(messages)
            raw_text = response.choices[0].message.content
        except Exception as exc:
            logger.warning("VLM 调用失败: %s", exc)
            return self._build_ocr_only_result(
                input_data, ocr_elements, f"VLM 调用异常: {exc}"
            )

        if not raw_text:
            return self._build_ocr_only_result(
                input_data, ocr_elements, "VLM 返回了空内容"
            )

        # 解析 VLM 输出
        result = self.parse_perception_response(raw_text)
        if not result.raw_model_output:
            result.raw_model_output = raw_text

        # ---- 核心：OCR + VLM 融合 ----
        result = self._merge_ocr_into_perception(result, ocr_elements)

        # 兜底：VLM 没给出 app_name 或给出了模糊/无关名称时用关键词猜测
        _vague_names = {"未知", "unknown", "Unknown", "无法判断", "不确定",
                        "系统应用", "桌面", "手机桌面", "主屏幕", "Home", "home",
                        "安卓", "Android", "系统桌面", "Launcher"}
        if (not result.app_name or result.app_name in _vague_names):
            guess = self._guess_app_name(input_data.instruction)
            if guess:
                result.app_name = guess

        return result

    def parse_perception_response(self, raw_text: str) -> ScreenPerception:
        """将 VLM 原始输出解析成 ScreenPerception。"""
        json_str = self._extract_json(raw_text)
        if json_str is None:
            return ScreenPerception(
                page_type="unknown",
                screen_summary=raw_text.strip()[:300],
                elements=[],
                warnings=["json_extract_failed"],
                raw_model_output=raw_text,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return ScreenPerception(
                page_type="unknown",
                screen_summary=json_str[:300],
                elements=[],
                warnings=["json_parse_failed"],
                raw_model_output=raw_text,
            )

        if not isinstance(data, dict):
            return ScreenPerception(
                page_type="unknown",
                screen_summary=str(data)[:300],
                elements=[],
                warnings=["unexpected_json_type"],
                raw_model_output=raw_text,
            )

        elements = self._parse_elements(data.get("elements", []))

        return ScreenPerception(
            app_name=str(data.get("app_name", "")).strip(),
            page_type=str(data.get("page_type", "unknown")).strip(),
            screen_summary=str(data.get("screen_summary", "")).strip(),
            elements=elements,
            keyboard_visible=bool(data.get("keyboard_visible", False)),
            scrollable=bool(data.get("scrollable", False)),
            warnings=self._parse_warnings(data.get("warnings", [])),
            candidate_actions=data.get("candidate_actions", []),
            raw_model_output=raw_text,
        )

    # ------------------------------------------------------------------
    # OCR 引擎
    # ------------------------------------------------------------------

    def _get_ocr_reader(self) -> Any:
        """延迟初始化 OCR 读取器（EasyOCR），只加载一次。"""
        if self._ocr_reader is not None:
            return self._ocr_reader
        if not self._use_ocr or self._ocr_init_failed:
            return None

        try:
            import easyocr
            # Chinese + English, no GPU (compatible with most environments)
            self._ocr_reader = easyocr.Reader(
                ['ch_sim', 'en'], gpu=False, verbose=False
            )
            logger.info("EasyOCR 初始化成功")
        except ImportError:
            logger.warning("easyocr 未安装，OCR 已禁用。安装命令: pip install easyocr")
            self._ocr_init_failed = True
        except Exception as exc:
            logger.warning("EasyOCR 初始化失败: %s", exc)
            self._ocr_init_failed = True

        return self._ocr_reader

    def _run_ocr(self, image: 'Image.Image') -> List[Dict[str, Any]]:
        """对截图运行 OCR，返回文字元素列表（坐标已归一化到 [0, 1000]）。

        同一张图片多次调用时，直接返回缓存结果，避免重复 OCR。
        每个元素：{element_id, text, bbox: [x1,y1,x2,y2], confidence, source: 'ocr'}
        """
        # 命中缓存直接返回
        img_id = id(image)
        if self._ocr_cache_image_id == img_id and self._ocr_cache is not None:
            return self._ocr_cache

        reader = self._get_ocr_reader()
        if reader is None:
            return []

        img_array = np.array(image)
        try:
            raw_results = reader.readtext(img_array)
        except Exception as exc:
            logger.warning("OCR 识别出错: %s", exc)
            return []

        w, h = image.width, image.height
        elements: List[Dict[str, Any]] = []

        for i, (bbox, text, conf) in enumerate(raw_results):
            text = str(text).strip()
            if not text:
                continue  # 忽略空文本

            # bbox 是四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # 归一化到 [0, 1000]
            norm_bbox = [
                max(0, min(1000, int(x1 / w * 1000))),
                max(0, min(1000, int(y1 / h * 1000))),
                max(0, min(1000, int(x2 / w * 1000))),
                max(0, min(1000, int(y2 / h * 1000))),
            ]

            # 过滤过小的噪点区域
            bw, bh = norm_bbox[2] - norm_bbox[0], norm_bbox[3] - norm_bbox[1]
            if bw < 3 or bh < 3:
                continue

            elements.append({
                "element_id": f"ocr_{i}",
                "text": text,
                "bbox": norm_bbox,
                "confidence": round(float(conf), 3),
                "source": "ocr",
            })

        logger.info("OCR 识别到 %d 个文字元素", len(elements))

        # 写入缓存
        self._ocr_cache = elements
        self._ocr_cache_image_id = img_id

        return elements

    # ------------------------------------------------------------------
    # OCR + VLM 融合
    # ------------------------------------------------------------------

    def _merge_ocr_into_perception(
        self,
        perception: ScreenPerception,
        ocr_elements: List[Dict[str, Any]],
    ) -> ScreenPerception:
        """将 OCR 精确 bbox 融合进 VLM 输出的 ScreenPerception。

        融合策略：
        1. 对 VLM 元素中的文字元素 → 找 OCR 匹配 → 用 OCR bbox 替换 VLM bbox
        2. OCR 中有但 VLM 没提到的文字 → 作为补充元素加入（标记 source='ocr_only'）
        """
        if not ocr_elements:
            return perception  # 没有 OCR 结果，直接返回

        # 建立 OCR 文本 → 精确 bbox 的索引（长文本优先）
        ocr_index: Dict[str, Dict[str, Any]] = {}
        for elem in sorted(ocr_elements, key=lambda e: -len(e["text"])):
            text = elem["text"]
            if text and text not in ocr_index:
                ocr_index[text] = elem

        matched_ocr_ids: set = set()

        # 第 1 步：为每个 VLM 元素匹配 OCR bbox
        for elem in perception.elements:
            vlm_text = (elem.text or "").strip()
            if not vlm_text:
                continue

            matched_ocr = self._match_ocr_text(vlm_text, ocr_elements)
            if matched_ocr:
                elem.bbox = matched_ocr["bbox"]
                elem.confidence = max(elem.confidence, matched_ocr["confidence"])
                matched_ocr_ids.add(matched_ocr["element_id"])

        # 第 2 步：OCR 中有但 VLM 没提到的文字 → 补充进去
        vlm_texts = {e.text.strip() for e in perception.elements if e.text}
        for ocr_elem in ocr_elements:
            if ocr_elem["element_id"] in matched_ocr_ids:
                continue
            if ocr_elem["text"] in vlm_texts:
                continue  # 文本已被 VLM 元素覆盖
            # 判断是否可能是有意义的交互元素
            if self._is_likely_interactive(ocr_elem):
                perception.elements.append(UIElement(
                    element_id=ocr_elem["element_id"],
                    role="other",
                    text=ocr_elem["text"],
                    description=f"OCR识别: {ocr_elem['text']}",
                    bbox=ocr_elem["bbox"],
                    clickable=True,
                    enabled=True,
                    confidence=ocr_elem["confidence"],
                ))

        return perception

    @staticmethod
    def _match_ocr_text(
        vlm_text: str,
        ocr_elements: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """在 OCR 结果中查找与 VLM 文本最匹配的元素。

        匹配优先级：完全匹配 > OCR 包含 VLM 文本 > VLM 文本包含 OCR
        """
        vlm_text = vlm_text.strip()
        if not vlm_text:
            return None

        # 完全匹配（最高优先级）
        for elem in ocr_elements:
            if elem["text"] == vlm_text:
                return elem

        # OCR 文本包含 VLM 文本（如 OCR="搜索按钮" 匹配 VLM="搜索"）
        for elem in ocr_elements:
            if vlm_text in elem["text"]:
                return elem

        # VLM 文本包含 OCR 文本（如 VLM="点击搜索按钮" 匹配 OCR="搜索"）
        for elem in ocr_elements:
            if elem["text"] in vlm_text:
                return elem

        return None

    @staticmethod
    def _is_likely_interactive(ocr_elem: Dict[str, Any]) -> bool:
        """判断 OCR 元素是否可能是可交互控件（按钮、Tab、输入框等）。

        启发式规则：
        - 短文本（1~8 字）更可能是按钮/Tab
        - 过长文本一般是内容/描述，不是控件
        - 靠近底部（y>850）的文字大概率是导航Tab
        """
        text = ocr_elem["text"]
        bbox = ocr_elem["bbox"]
        text_len = len(text)

        # 太短（1字且非中文）或太长（>20字）不太像交互控件
        if text_len > 20:
            return False
        if text_len == 1 and not ('一' <= text <= '鿿'):
            return False

        # 底部区域（y1 > 850）大概率是导航Tab
        if bbox[1] > 850 and text_len <= 6:
            return True

        # 短文本（2~8 字）更可能是按钮
        if 2 <= text_len <= 8:
            return True

        return False

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _build_ocr_only_result(
        self,
        input_data: AgentInput,
        ocr_elements: List[Dict[str, Any]],
        reason: str,
    ) -> ScreenPerception:
        """VLM 不可用时，仅用 OCR 结果构造降级 ScreenPerception。"""
        elements: List[UIElement] = []
        for ocr_elem in ocr_elements:
            if self._is_likely_interactive(ocr_elem):
                elements.append(UIElement(
                    element_id=ocr_elem["element_id"],
                    role="other",
                    text=ocr_elem["text"],
                    description=f"OCR: {ocr_elem['text']}",
                    bbox=ocr_elem["bbox"],
                    clickable=True,
                    enabled=True,
                    confidence=ocr_elem["confidence"],
                ))

        return ScreenPerception(
            app_name=self._guess_app_name(input_data.instruction),
            page_type="unknown",
            screen_summary=f"{reason}，返回 OCR 降级结果。识别到 {len(ocr_elements)} 个文字元素。",
            elements=elements,
            keyboard_visible=False,
            scrollable=True,
            warnings=["vlm_unavailable", reason],
        )

    @staticmethod
    def _guess_app_name(instruction: str) -> str:
        """从任务指令中猜测目标 App 名称（降级策略）。"""
        known_apps = [
            "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手", "芒果TV",
            "美团", "腾讯视频", "喜马拉雅", "QQ", "淘宝", "微信",
            "京东", "拼多多", "铁路12306", "大众点评",
        ]
        for app in known_apps:
            if app in instruction:
                return app
        return ""

    @staticmethod
    def _build_history_context(memory: MemoryState) -> str:
        """将最近的历史步骤整理成文本，嵌入 prompt 中。"""
        if not memory.history:
            return "（这是第一步，尚无历史操作）"

        recent = memory.history[-5:]
        lines = ["## 最近的历史操作"]
        for step in recent:
            action = step.action
            params = step.parameters
            if action == "CLICK":
                detail = f"点击坐标 {params.get('point')}"
            elif action == "SCROLL":
                detail = f"从 {params.get('start_point')} 滑动到 {params.get('end_point')}"
            elif action == "TYPE":
                detail = f"输入文本 '{params.get('text')}'"
            elif action == "OPEN":
                detail = f"打开应用 {params.get('app_name')}"
            elif action == "COMPLETE":
                detail = "任务完成"
            else:
                detail = str(params)
            lines.append(f"- 第{step.step_index}步: {action} — {detail}")
        return "\n".join(lines)

    @staticmethod
    def _format_ocr_list(ocr_elements: List[Dict[str, Any]]) -> tuple:
        """将 OCR 元素格式化为 prompt 中的文本列表。"""
        if not ocr_elements:
            return "（OCR 未识别到文字或 OCR 引擎不可用）", 0

        lines = []
        for elem in ocr_elements:
            lines.append(
                f"  {elem['element_id']}: \"{elem['text']}\" "
                f"bbox={elem['bbox']} conf={elem['confidence']}"
            )
        return "\n".join(lines), len(ocr_elements)

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

        # 策略 3: 裸 JSON（第一个 { 到最后一个 }）
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return raw_text[start:end + 1]

        return None

    @staticmethod
    def _parse_elements(raw_elements: Any) -> List[UIElement]:
        """将 VLM 返回的 elements 列表解析为 UIElement 对象列表。"""
        if not isinstance(raw_elements, list):
            return []

        parsed: List[UIElement] = []
        for i, elem in enumerate(raw_elements):
            if not isinstance(elem, dict):
                continue
            bbox = elem.get("bbox", [])
            parsed.append(UIElement(
                element_id=str(elem.get("element_id", f"elem_{i}")),
                role=elem.get("role", "other"),
                text=str(elem.get("text", "")).strip(),
                description=str(elem.get("description", "")).strip(),
                bbox=PerceptionModule._clamp_bbox(bbox),
                clickable=bool(elem.get("clickable", False)),
                enabled=bool(elem.get("enabled", True)),
                confidence=float(elem.get("confidence", 0.0)),
            ))
        return parsed

    @staticmethod
    def _parse_warnings(raw_warnings: Any) -> List[str]:
        """标准化 warnings 字段。"""
        if not isinstance(raw_warnings, list):
            return []
        return [str(w)[:200] for w in raw_warnings if w]

    @staticmethod
    def _clamp_bbox(bbox: Any) -> List[int]:
        """将 bbox 四个坐标值限制在 [0, 1000] 范围内。"""
        if not isinstance(bbox, list) or len(bbox) != 4:
            return []
        try:
            return [
                max(0, min(1000, int(bbox[0]))),
                max(0, min(1000, int(bbox[1]))),
                max(0, min(1000, int(bbox[2]))),
                max(0, min(1000, int(bbox[3]))),
            ]
        except (ValueError, TypeError):
            return []
