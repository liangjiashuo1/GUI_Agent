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
- 哔哩哔哩(B站)：粉色/粉蓝色主题，底部 首页/热门/动态/我的，搜索页有"热搜"标签
- 抖音：黑色背景，短视频，底部 首页/朋友/消息/我，顶部有"推荐/附近/关注"Tab
- 快手：橙色/红色主题，短视频，底部 首页/发现/消息/我，顶部有"关注/发现/精选"Tab
- 芒果TV：橙色主题，视频播放，底部 首页/会员/我的，搜索页橙色搜索框
- 腾讯视频：蓝黑/深蓝色主题，视频播放，底部 首页/会员/我的，搜索框蓝底白字
- 百度地图：地图界面，底部 首页/出行/周边/我的
- 微信：绿色主题，聊天列表，底部 微信/通讯录/发现/我
- QQ：浅蓝主题，聊天，底部 消息/联系人/动态
- 美团：黄色主题，底部 首页/我的
- 淘宝：橙色主题，商品列表，底部 首页/购物车/消息/我的
- 喜马拉雅：橙红主题，音频播放

## element_id 选择规则
- 每个 OCR 元素有唯一 element_id（如 ocr_0, ocr_1）
- 你需要为 task 目标选择最相关的元素，返回它的 element_id
- **关键**：仔细对比 OCR 列表中每个元素的文字内容和位置，确保 element_id 对应正确的元素
- 底部Tab元素通常 y 坐标 > 900
- 搜索框通常在 y 坐标 40-120 之间
- 顶部导航Tab通常在 y 坐标 40-100 之间"""


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
        # 最低置信度阈值，低于此值的 OCR 结果将被丢弃
        self._ocr_confidence_threshold: float = 0.2

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

        # 注入目标应用名提示，帮助 VLM 正确识别 app_name
        app_hint = self._guess_app_name(input_data.instruction)
        if app_hint:
            user_text += (
                f'\n\n## 重要提示\n'
                f'根据任务指令，目标应用是 **"{app_hint}"**，'
                f'请在 app_name 字段中返回此名称。'
                f'如果屏幕显示的应用图标或界面特征与该应用匹配，请确认为当前应用。'
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

        # ---- OCR 后处理增强：键盘检测 + 页面类型纠错 ----
        # C 模块的搜索/输入硬规则依赖这两个字段，因此不要完全相信 VLM。
        if self._detect_keyboard_via_ocr(ocr_elements):
            result.keyboard_visible = True

        ocr_texts = [e["text"] for e in ocr_elements]
        inferred_page = self._infer_page_type_from_ocr(ocr_texts, result.elements)
        if inferred_page:
            # popup/search/detail 这类状态对动作决策影响更大，优先采用 OCR 纠错结果。
            if result.page_type in ("unknown", "home") or inferred_page in ("popup", "search", "detail", "settings"):
                result.page_type = inferred_page
            if result.screen_summary:
                result.screen_summary = f"{result.screen_summary} | OCR推断页面类型: {inferred_page}"
            else:
                result.screen_summary = f"OCR推断页面类型: {inferred_page}"

        # 兜底：VLM 没给出 app_name 或给出了模糊/无关名称时用关键词猜测
        _vague_names = {"未知", "unknown", "Unknown", "无法判断", "不确定",
                        "系统应用", "桌面", "手机桌面", "主屏幕", "Home", "home",
                        "安卓", "Android", "系统桌面", "Launcher",
                        "手机系统", "系统界面", "System UI", "桌面系统",
                        "安卓桌面", "Android桌面", "启动器", "默认桌面", "系统启动器"}
        # 短名称启发式：≤2字且不在已知app列表中，大概率是截断名称
        _known_apps_set = {
            "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手", "芒果TV",
            "美团", "腾讯视频", "喜马拉雅", "QQ", "淘宝", "微信",
            "京东", "拼多多", "铁路12306", "大众点评", "B站",
        }
        app_name = result.app_name.strip() if result.app_name else ""
        is_vague = (not app_name or app_name in _vague_names)
        is_short = (len(app_name) <= 2 and app_name not in _known_apps_set)
        if is_vague or is_short:
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

            # 置信度过滤：过低的结果多为噪声；但关键控件词即使低置信度也先保留。
            important_keywords = [
                "搜索", "搜", "评论", "发送", "发布", "我的", "首页", "跳过",
                "关闭", "取消", "下载", "缓存", "离线", "播放", "确定", "确认",
            ]
            is_important = any(k in text for k in important_keywords)
            if conf < self._ocr_confidence_threshold and not is_important:
                continue

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

        logger.info("OCR 识别到 %d 个文字元素（去重前）", len(elements))

        # IoU 去重：重叠度过高的元素保留置信度高者
        elements = PerceptionModule._deduplicate_by_iou(elements)

        # 按空间位置排序后重新分配 element_id（确保 ocr_0=最上方, ocr_N=最下方）
        elements.sort(key=lambda e: (
            (e["bbox"][1] + e["bbox"][3]) // 2 // 50,  # y 中心分组（每50px）
            (e["bbox"][0] + e["bbox"][2]) // 2,         # 同组按 x 中心排序
        ))
        for new_i, elem in enumerate(elements):
            elem["element_id"] = f"ocr_{new_i}"

        # 写入缓存
        self._ocr_cache = elements
        self._ocr_cache_image_id = img_id

        return elements

    @staticmethod
    def _deduplicate_by_iou(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """IoU 去重：两个元素 bbox 重叠度 > 0.7 时保留置信度更高者。"""
        if len(elements) <= 1:
            return elements

        def _iou(a: List[int], b: List[int]) -> float:
            x_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            y_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = x_overlap * y_overlap
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        kept: List[Dict[str, Any]] = []
        for elem in elements:
            replaced = False
            for i, existing in enumerate(kept):
                if _iou(elem["bbox"], existing["bbox"]) > 0.7:
                    if elem["confidence"] > existing["confidence"]:
                        kept[i] = elem  # 替换为置信度更高的
                    replaced = True
                    break
            if not replaced:
                kept.append(elem)
        return kept

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

            matched_ocr = self._match_ocr_text(vlm_text, ocr_elements, elem.bbox)
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
                hint = PerceptionModule._infer_ocr_role_hint(ocr_elem["text"], ocr_elem["bbox"])
                role = "other"
                desc = f"OCR: {ocr_elem['text']}"
                if hint:
                    desc += f"（{hint}）"
                    # 根据角色提示推断 role
                    if "按钮" in hint or "入口" in hint:
                        role = "button"
                    elif "输入框" in hint:
                        role = "input"
                    elif "Tab" in hint:
                        role = "tab"
                    elif "广告" in hint:
                        role = "other"
                        perception.warnings.append(f"检测到可能的广告: {ocr_elem['text']}")
                perception.elements.append(UIElement(
                    element_id=ocr_elem["element_id"],
                    role=role,
                    text=ocr_elem["text"],
                    description=desc,
                    bbox=ocr_elem["bbox"],
                    clickable=True,
                    enabled=True,
                    confidence=ocr_elem["confidence"],
                ))

        # 第 3 步：为爱奇艺常见纯图标控件补充虚拟元素，弥补 OCR 只能识别文字的问题。
        perception = self._add_iqiyi_virtual_elements(perception, ocr_elements)

        # 第 4 步：为所有元素描述添加屏幕位置标注，帮助 C 模块 VLM 区分相似元素。
        for elem in perception.elements:
            pos = PerceptionModule._describe_screen_position(elem.bbox)
            if pos and pos not in (elem.description or ""):
                if elem.description:
                    elem.description = f"{elem.description} | {pos}"
                else:
                    elem.description = pos

        # 第 5 步：统一去重，避免同一文字/同一位置同时出现 OCR 和 VLM 元素。
        perception.elements = PerceptionModule._deduplicate_ui_elements(perception.elements)

        return perception

    @staticmethod
    def _deduplicate_ui_elements(elements: List[UIElement]) -> List[UIElement]:
        """去除 VLM 元素和 OCR 元素之间的重复框。

        保留策略：
        - 优先保留 clickable=True 的元素；
        - 其次保留 bbox 完整、confidence 更高的元素；
        - 文本完全相同或 bbox 高度重叠时视为重复。
        """
        if len(elements) <= 1:
            return elements

        def _iou(a: List[int], b: List[int]) -> float:
            if not a or not b or len(a) != 4 or len(b) != 4:
                return 0.0
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def _score(elem: UIElement) -> float:
            return (
                int(elem.clickable) * 3
                + int(bool(elem.bbox and len(elem.bbox) == 4)) * 2
                + float(elem.confidence or 0.0)
            )

        kept: List[UIElement] = []
        for elem in elements:
            duplicate_idx: Optional[int] = None
            for idx, old in enumerate(kept):
                same_text = bool(elem.text and old.text and elem.text == old.text)
                high_overlap = _iou(elem.bbox, old.bbox) > 0.65
                if same_text or high_overlap:
                    duplicate_idx = idx
                    break

            if duplicate_idx is None:
                kept.append(elem)
            else:
                if _score(elem) > _score(kept[duplicate_idx]):
                    kept[duplicate_idx] = elem

        return kept

    @staticmethod
    def _add_iqiyi_virtual_elements(
        perception: ScreenPerception,
        ocr_elements: List[Dict[str, Any]],
    ) -> ScreenPerception:
        """为爱奇艺常见纯图标控件补充虚拟元素。

        这些控件往往没有可 OCR 的文字，但在爱奇艺任务中经常被点击。
        坐标使用 [0, 1000] 归一化坐标，作为保守兜底候选交给 C 模块选择。
        """
        app_name = perception.app_name or ""
        all_text = " ".join(e.get("text", "") for e in ocr_elements)
        if "爱奇艺" not in app_name and "爱奇艺" not in all_text:
            return perception

        existing_ids = {elem.element_id for elem in perception.elements}

        def _add(
            element_id: str,
            role: str,
            text: str,
            description: str,
            bbox: List[int],
            confidence: float,
        ) -> None:
            if element_id in existing_ids:
                return
            perception.elements.append(UIElement(
                element_id=element_id,
                role=role,
                text=text,
                description=description,
                bbox=bbox,
                clickable=True,
                enabled=True,
                confidence=confidence,
            ))
            existing_ids.add(element_id)

        # 首页/搜索页顶部搜索区域：有些界面只有放大镜图标或提示文字，OCR bbox 不一定覆盖整个输入框。
        if perception.page_type in ("home", "search", "unknown") and any(k in all_text for k in ["搜索", "搜", "热搜"]):
            _add(
                "virtual_top_search_box",
                "input",
                "搜索框",
                "爱奇艺顶部搜索输入框兜底区域",
                [70, 45, 830, 120],
                0.66,
            )

        # 弹窗/广告关闭区域：避免 C 模块误点开通会员、广告位。
        if perception.page_type == "popup" or any(k in all_text for k in ["广告", "跳过", "开通会员", "立即开通"]):
            _add(
                "virtual_popup_close",
                "button",
                "关闭",
                "弹窗/广告右上角关闭或跳过按钮兜底区域",
                [875, 55, 985, 165],
                0.64,
            )

        # 播放详情页评论入口：评论图标有时是纯图标，OCR 可能只看到评论数。
        if perception.page_type == "detail" or any(k in all_text for k in ["评论", "写评论", "发评论"]):
            _add(
                "virtual_comment_entry",
                "button",
                "评论",
                "播放页底部评论入口兜底区域",
                [35, 835, 320, 955],
                0.60,
            )

        # 底部“我的”Tab 兜底。
        if any(k in all_text for k in ["首页", "会员", "我的", "随刻"]):
            _add(
                "virtual_mine_tab",
                "tab",
                "我的",
                "底部导航我的Tab兜底区域",
                [760, 895, 1000, 1000],
                0.65,
            )

        # 我的页下载/离线缓存区域兜底。
        if perception.page_type == "settings" or any(k in all_text for k in ["离线缓存", "下载", "缓存"]):
            _add(
                "virtual_download_entry",
                "button",
                "离线缓存/下载",
                "我的页面离线缓存或下载入口兜底区域",
                [40, 230, 960, 420],
                0.56,
            )

        return perception

    @staticmethod
    def _detect_keyboard_via_ocr(ocr_elements: List[Dict[str, Any]]) -> bool:
        """通过 OCR 元素分布推断键盘是否可见。

        键盘特征：屏幕下半部分有大量短字符，并且这些短字符分布成多行。
        """
        if not ocr_elements:
            return False

        lower_elems = [
            e for e in ocr_elements
            if e["bbox"][1] > 520
            and len(str(e["text"]).strip()) <= 2
            and float(e.get("confidence", 0.0)) > 0.25
        ]
        if len(lower_elems) < 8:
            return False

        rows: Dict[int, int] = {}
        for elem in lower_elems:
            cy = (elem["bbox"][1] + elem["bbox"][3]) // 2
            row_key = cy // 35
            rows[row_key] = rows.get(row_key, 0) + 1

        rows_with_many_keys = sum(1 for count in rows.values() if count >= 3)
        return rows_with_many_keys >= 3

    @staticmethod
    def _infer_page_type_from_ocr(
        ocr_texts: List[str],
        elements: List[UIElement],
    ) -> Optional[str]:
        """根据 OCR 文本和元素位置推断页面类型。优先级：popup > detail > search > settings > home"""
        all_text = " ".join(t for t in ocr_texts if t)

        popup_words = ["跳过", "关闭", "知道了", "我知道了", "开通会员", "立即开通", "青少年模式", "广告"]
        # 强搜索特征（仅搜索页才有，避免"搜索"一词在详情页触发误判）
        strong_search_words = ["热搜", "大家都在搜", "搜索历史", "搜索你想看的", "搜一搜"]
        # 强详情页特征（只有播放详情页才有）
        detail_words = ["选集", "简介", "倍速", "缓存", "全屏", "写评论", "发表评论", "输入评论", "说说你的看法"]
        # 弱详情特征（详情页常见但搜索页也可能出现）
        weak_detail_words = ["评论", "发送", "弹幕"]
        mine_words = ["我的", "离线缓存", "下载", "观看历史", "收藏", "设置", "会员中心"]
        home_words = ["首页", "推荐", "热播", "电视剧", "电影", "综艺", "动漫", "随刻"]

        def _has_any(words: List[str]) -> bool:
            return any(word in all_text for word in words)

        # 1) 弹窗优先级最高，遮挡底层页面。
        if _has_any(popup_words):
            return "popup"

        # 2) 详情页优先于搜索页，详情页常有搜索图标易误判。
        if _has_any(detail_words):
            return "detail"
        if _has_any(weak_detail_words) and not _has_any(strong_search_words):
            return "detail"

        # 3) 搜索页：必须强特征才判，弱词"搜索""取消"不单独触发。
        if _has_any(strong_search_words):
            return "search"

        # 4) 我的页
        if "我的" in all_text and _has_any(mine_words):
            return "settings"
        if _has_any(["离线缓存", "下载", "观看历史", "收藏", "设置"]):
            return "settings"

        # 5) 首页
        if _has_any(home_words):
            return "home"

        # 位置兜底：底部 Tab 中出现"我的"
        for elem in elements:
            if elem.text == "我的" and elem.bbox and len(elem.bbox) == 4 and elem.bbox[1] > 850:
                return "home"

        return None


    @staticmethod
    def _describe_screen_position(bbox: List[int]) -> str:
        """根据 bbox 中心点返回屏幕位置描述。"""
        if not bbox or len(bbox) != 4:
            return ""
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        # 垂直分区
        if cy < 80:
            v = "顶部状态栏"
        elif cy < 200:
            v = "顶部区域"
        elif cy < 500:
            v = "中部偏上"
        elif cy < 700:
            v = "中部"
        elif cy < 850:
            v = "中部偏下"
        else:
            v = "底部"

        # 水平分区
        if cx < 250:
            h = "左侧"
        elif cx < 500:
            h = "中间偏左"
        elif cx < 750:
            h = "中间偏右"
        else:
            h = "右侧"

        return f"屏幕{v}{h}"

    @staticmethod
    def _match_ocr_text(
        vlm_text: str,
        ocr_elements: List[Dict[str, Any]],
        vlm_bbox: Optional[List[int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """在 OCR 结果中查找与 VLM 文本最匹配的元素。

        匹配优先级：完全匹配 > OCR 包含 VLM 文本 > VLM 文本包含 OCR > 高重合度匹配
        当有多个候选时，若提供了 vlm_bbox，选择 bbox 中心最接近者。
        """
        vlm_text = vlm_text.strip()
        if not vlm_text:
            return None

        def _bbox_center(b: List[int]) -> float:
            return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

        def _bbox_dist(a: List[int], b: List[int]) -> float:
            ca, cb = _bbox_center(a), _bbox_center(b)
            return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

        def _pick_best(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            if vlm_bbox and len(vlm_bbox) == 4:
                return min(candidates, key=lambda e: _bbox_dist(e["bbox"], vlm_bbox))
            return candidates[0]

        # 1) 完全匹配（最高优先级）
        exact = [e for e in ocr_elements if e["text"] == vlm_text]
        if exact:
            return _pick_best(exact)

        # 2) OCR 文本包含 VLM 文本（如 OCR="搜索按钮" 匹配 VLM="搜索"）
        contains = [e for e in ocr_elements if vlm_text in e["text"]]
        if contains:
            return _pick_best(contains)

        # 3) VLM 文本包含 OCR 文本（如 VLM="点击搜索按钮" 匹配 OCR="搜索"）
        contained = [e for e in ocr_elements if e["text"] in vlm_text]
        if contained:
            return _pick_best(contained)

        # 4) 字符重叠度匹配：两个文本有≥50%字符重叠时视为匹配
        overlaps = []
        for elem in ocr_elements:
            vlm_chars = set(vlm_text)
            ocr_chars = set(elem["text"])
            if not vlm_chars or not ocr_chars:
                continue
            overlap = len(vlm_chars & ocr_chars)
            min_len = min(len(vlm_chars), len(ocr_chars))
            if min_len >= 2 and overlap / min_len >= 0.5:
                overlaps.append(elem)
        if overlaps:
            return _pick_best(overlaps)

        return None

    @staticmethod
    def _is_likely_interactive(ocr_elem: Dict[str, Any]) -> bool:
        """判断 OCR 元素是否可能是可交互控件（按钮、Tab、输入框等）。

        启发式规则（放宽版本，尽可能保留更多元素给 C 模块选择）：
        - 短文本（1~15 字）大概率是按钮/Tab/标签
        - 底部区域（y>820）的文字大概率是导航Tab
        - 过长文本（>30字）一般是正文内容，非控件
        """
        text = ocr_elem["text"]
        bbox = ocr_elem["bbox"]
        text_len = len(text)

        # 太长（>30字）一般是文章内容/描述，非控件
        if text_len > 30:
            return False

        # 纯数字且长度 > 5（可能是计数器/ID，不太可能交互）
        if text.isdigit() and text_len > 5:
            return False

        # 底部区域（y1 > 820）大概率是导航Tab
        if bbox[1] > 820 and text_len <= 8:
            return True

        # 短文本（1~15 字）很可能是按钮/Tab/标签
        if text_len <= 15:
            return True

        # 中长文本（15-30字），如果在顶部可能是标题，在中间可能是描述
        if text_len <= 30:
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
    def _infer_ocr_role_hint(text: str, bbox: List[int]) -> str:
        """根据文字内容和位置推断角色提示，帮助VLM更准确理解元素功能。
        只在置信度高时给出提示，避免误导。"""
        t = text.strip()
        if not t:
            return ""

        # 底部区域（y>880）的短文字 → 底部导航Tab
        if bbox[1] > 880 and len(t) <= 6:
            return "底部导航Tab"

        # 完全匹配的关闭/跳过按钮
        if t in ("关闭", "跳过", "×", "✕", "取消", "Close", "Skip", "知道了", "我知道了"):
            return "关闭/跳过按钮"

        # 完全匹配的确认按钮
        if t in ("确认", "确定", "提交", "发送", "发布", "OK", "确认发布", "立即发布"):
            return "确认/提交按钮"

        # 搜索框（明确包含"搜索"+框/栏）
        if ("搜索" in t or "搜" in t) and any(kw in t for kw in ["框", "栏", "输入", "Search"]):
            return "搜索输入框"

        # 评论输入区
        if any(kw in t for kw in ["写评论", "发表评论", "输入评论", "说说你的看法"]):
            return "评论输入框"

        # 返回按钮
        if t in ("返回", "后退", "←"):
            return "返回按钮"

        # 广告标识
        if "广告" in t and len(t) <= 8:
            return "广告标识"

        return ""

    @staticmethod
    def _format_ocr_list(ocr_elements: List[Dict[str, Any]]) -> tuple:
        """将 OCR 元素格式化为 prompt 中的文本列表。元素已在上游按空间位置排序。"""
        if not ocr_elements:
            return "（OCR 未识别到文字或 OCR 引擎不可用）", 0

        lines = []
        for elem in ocr_elements:
            hint = PerceptionModule._infer_ocr_role_hint(elem["text"], elem["bbox"])
            hint_str = f" |角色: {hint}" if hint else ""
            bbox = elem["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            lines.append(
                f"  {elem['element_id']}: \"{elem['text']}\" "
                f"bbox={bbox} 中心=({cx},{cy}) conf={elem['confidence']:.2f}{hint_str}"
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
