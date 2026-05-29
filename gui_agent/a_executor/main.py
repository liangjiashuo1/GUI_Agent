"""
A: Executor module entry.

Supports CLICK / SCROLL / TYPE / OPEN / COMPLETE via ADB.
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from PIL import Image

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE, AgentOutput
from gui_agent.shared.schemas import PlannerDecision, clamp_point

logger = logging.getLogger(__name__)

# 常见 App 包名映射（用于 OPEN 动作的 adb am start）
APP_PACKAGE_MAP: Dict[str, str] = {
    "爱奇艺": "com.qiyi.video",
    "抖音": "com.ss.android.ugc.aweme",
    "快手": "com.smile.gifmaker",
    "哔哩哔哩": "tv.danmaku.bili",
    "B站": "tv.danmaku.bili",
    "腾讯视频": "com.tencent.qqlive",
    "芒果TV": "com.hunantv.imgo.activity",
    "百度地图": "com.baidu.BaiduMap",
    "美团": "com.sankuai.meituan",
    "喜马拉雅": "com.ximalaya.ting.android",
    "淘宝": "com.taobao.taobao",
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "京东": "com.jingdong.app.mall",
    "拼多多": "com.xunmeng.pinduoduo",
    "铁路12306": "com.MobileTicket",
    "大众点评": "com.dianping.v1",
}


class ExecutorModule:
    DEFAULT_SCREEN_SIZE = (1080, 2340)
    DEFAULT_MUMU_DEVICE = "127.0.0.1:16384"
    DEFAULT_ADB_PATH = r"D:\Tools\platform-tools\adb.exe"

    def __init__(self, adb_path: str | None = None, device_serial: str | None = None) -> None:
        self._adb_path = self._resolve_adb_path(adb_path)
        self._device_serial = device_serial or os.environ.get("ADB_DEVICE_SERIAL") or self._detect_device_serial()
        self._last_visual_size: Tuple[int, int] | None = None

    def build_device_command(self, decision: PlannerDecision) -> Dict[str, Any]:
        action = decision.action
        parameters = self._sanitize_parameters(action, decision.parameters)
        command = self._build_adb_command(action, parameters)
        return {
            "executor": "adb",
            "adb_path": self._adb_path,
            "device_serial": self._device_serial,
            "action": action.lower(),
            "payload": parameters,
            "command": command,
        }

    def get_screen_size(self) -> Tuple[int, int]:
        try:
            output = self._run_adb(["shell", "wm", "size"], capture_output=True, text=True)
            text = (output.stdout or "").strip()
            match = re.search(r"(\d+)\s*x\s*(\d+)", text)
            if not match:
                raise ValueError(f"无法解析屏幕尺寸输出: {text!r}")
            return int(match.group(1)), int(match.group(2))
        except Exception as exc:
            logger.warning(
                "获取屏幕分辨率失败，回退到默认值 %sx%s: %s",
                self.DEFAULT_SCREEN_SIZE[0],
                self.DEFAULT_SCREEN_SIZE[1],
                exc,
            )
            return self.DEFAULT_SCREEN_SIZE

    def get_coordinate_space_size(self) -> Tuple[int, int]:
        """
        Prefer the latest screenshot size so execution coordinates stay aligned
        with the exact image that B/C used for perception and planning.
        """
        if self._last_visual_size is not None:
            return self._last_visual_size
        return self.get_screen_size()

    def execute_adb_command(self, decision: PlannerDecision) -> bool:
        device_command = self.build_device_command(decision)
        commands = device_command["command"]

        if not commands:
            logger.info("收到 COMPLETE 动作，不执行 adb 命令。")
            return True

        # commands 是 List[List[str]]，支持多步执行（如中文输入先写剪贴板再粘贴）
        for i, cmd in enumerate(commands):
            try:
                self._run_adb(cmd, capture_output=True, text=True, check=True)
                # 多条命令间加短暂延迟，确保前一条生效（如剪贴板写入后再粘贴）
                if i < len(commands) - 1:
                    time.sleep(0.3)
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                logger.error("执行 ADB 指令失败: %s | cmd=%s | stderr=%s", exc, cmd, stderr)
                return False
            except Exception as exc:
                logger.error("执行 ADB 指令失败: %s | cmd=%s", exc, cmd)
                return False
        return True

    def compile_decision(self, decision: PlannerDecision) -> AgentOutput:
        action = decision.action
        parameters = self._sanitize_parameters(action, decision.parameters)
        return AgentOutput(action=action, parameters=parameters, raw_output=decision.thought)

    def _build_adb_command(self, action: str, parameters: Dict[str, Any]) -> List[str]:
        if action == ACTION_CLICK:
            x, y = self._normalized_to_absolute(parameters["point"])
            return [["shell", "input", "tap", str(x), str(y)]]

        if action == ACTION_SCROLL:
            sx, sy = self._normalized_to_absolute(parameters["start_point"])
            ex, ey = self._normalized_to_absolute(parameters["end_point"])
            return [["shell", "input", "swipe", str(sx), str(sy), str(ex), str(ey), "500"]]

        if action == ACTION_TYPE:
            text = parameters["text"]
            if self._has_non_ascii(text):
                # 切到 ADBKeyboard → 广播发送文本
                return [
                    ["shell", "ime", "set", "com.android.adbkeyboard/.AdbIME"],
                    ["shell", "am", "broadcast", "-a", "ADB_INPUT_TEXT", "--es", "msg", text],
                ]
            escaped = self._escape_adb_text(text)
            return [["shell", "input", "text", escaped]]

        if action == ACTION_OPEN:
            app_name = parameters["app_name"]
            pkg = self._resolve_package(app_name)
            # force-stop 清除上次残留状态（搜索历史/推荐词），然后冷启动
            return [
                ["shell", "am", "force-stop", pkg],
                ["shell", "monkey", "-p", pkg, "1"],
            ]

        if action == ACTION_COMPLETE:
            return []

        raise ValueError(f"不支持的动作类型: {action}")

    def _sanitize_parameters(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
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

        raise ValueError(f"不支持的动作类型: {action}")

    @staticmethod
    def _is_point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) == 2

    def capture_screenshot(self) -> "Image.Image | None":
        try:
            from PIL import Image

            result = self._run_adb(
                ["shell", "screencap", "-p"],
                capture_output=True,
                check=True,
                timeout=10,
            )
            png_data = result.stdout.replace(b"\r\n", b"\n")
            image = Image.open(io.BytesIO(png_data))
            image.load()
            self._last_visual_size = (image.width, image.height)
            return image
        except subprocess.TimeoutExpired:
            logger.error("截图超时，ADB 命令在 10 秒内未返回结果。")
            return None
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            logger.error("执行截图命令失败，返回码=%s stderr=%s", exc.returncode, stderr.strip())
            return None
        except Exception as exc:
            logger.error("截图或图像处理失败: %s", exc)
            return None

    def _normalized_to_absolute(self, point: List[int]) -> Tuple[int, int]:
        width, height = self.get_coordinate_space_size()
        x = int(point[0] / 1000.0 * width)
        y = int(point[1] / 1000.0 * height)
        return x, y

    @classmethod
    def _resolve_package(cls, app_name: str) -> str:
        """根据中文应用名查包名，用于 adb 启动。"""
        pkg = APP_PACKAGE_MAP.get(app_name)
        if not pkg:
            raise ValueError(
                f"未找到应用「{app_name}」的包名映射，请在 APP_PACKAGE_MAP 中添加。"
            )
        return pkg

    @staticmethod
    def _has_non_ascii(text: str) -> bool:
        return any(ord(ch) > 127 for ch in text)

    @staticmethod
    def _escape_adb_text(text: str) -> str:
        return text.replace("%", r"\%").replace(" ", "%s")

    def _run_adb(
        self,
        command: List[str],
        *,
        capture_output: bool = False,
        text: bool = False,
        check: bool = True,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        full_cmd = [self._adb_path]
        if self._device_serial:
            full_cmd.extend(["-s", self._device_serial])
        full_cmd.extend(command)
        return subprocess.run(
            full_cmd,
            capture_output=capture_output,
            text=text,
            check=check,
            timeout=timeout,
        )

    def _detect_device_serial(self) -> str:
        devices = self._list_devices()
        if not devices:
            return self.DEFAULT_MUMU_DEVICE
        if self.DEFAULT_MUMU_DEVICE in devices:
            return self.DEFAULT_MUMU_DEVICE
        if len(devices) == 1:
            return devices[0]
        for serial in devices:
            if serial.startswith("127.0.0.1:") or serial.startswith("emulator-"):
                return serial
        return devices[0]

    def _list_devices(self) -> List[str]:
        try:
            result = subprocess.run(
                [self._adb_path, "devices"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as exc:
            logger.warning("列出 adb 设备失败: %s", exc)
            return []

        devices: List[str] = []
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices attached"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    @classmethod
    def _resolve_adb_path(cls, adb_path: str | None) -> str:
        if adb_path:
            return adb_path

        env_path = os.environ.get("ADB_PATH")
        if env_path:
            return env_path

        if os.path.exists(cls.DEFAULT_ADB_PATH):
            return cls.DEFAULT_ADB_PATH

        return "adb"
