"""
A: Executor module entry.

Supports CLICK / SCROLL / TYPE / OPEN / COMPLETE.
"""

from __future__ import annotations

import io
import logging
import os
import re
import subprocess
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from PIL import Image

from agent_base import ACTION_CLICK, ACTION_COMPLETE, ACTION_OPEN, ACTION_SCROLL, ACTION_TYPE, AgentOutput
from gui_agent.shared.schemas import PlannerDecision, clamp_point

logger = logging.getLogger(__name__)


class ExecutorModule:
    DEFAULT_SCREEN_SIZE = (1080, 2340)
    DEFAULT_MUMU_DEVICE = "127.0.0.1:16384"
    DEFAULT_ADB_PATH = r"D:\Tools\platform-tools\adb.exe"
    DEFAULT_LAUNCH_COMPONENTS = {
        "微信": "com.tencent.mm/.ui.LauncherUI",
        "QQ": "com.tencent.mobileqq/.activity.SplashActivity",
        "哔哩哔哩": "tv.danmaku.bili/.ui.splash.SplashActivity",
        "B站": "tv.danmaku.bili/.ui.splash.SplashActivity",
        "抖音": "com.ss.android.ugc.aweme/.main.MainActivity",
        "淘宝": "com.taobao.taobao/com.taobao.tao.welcome.Welcome",
        "爱奇艺": "com.qiyi.video/.WelcomeActivity",
    }

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
                "获取屏幕尺寸失败，回退到默认值 %sx%s: %s",
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
        command = device_command["command"]

        if not command:
            logger.info("收到 COMPLETE 动作，无需执行 adb 命令。")
            return True

        try:
            self._run_adb(command, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logger.error("执行 ADB 命令失败: %s | stderr=%s", exc, stderr)
            return False
        except Exception as exc:
            logger.error("执行 ADB 命令失败: %s", exc)
            return False

    def compile_decision(self, decision: PlannerDecision) -> AgentOutput:
        action = decision.action
        parameters = self._sanitize_parameters(action, decision.parameters)
        return AgentOutput(action=action, parameters=parameters, raw_output=decision.thought)

    def _build_adb_command(self, action: str, parameters: Dict[str, Any]) -> List[str]:
        if action == ACTION_CLICK:
            x, y = self._normalized_to_absolute(parameters["point"])
            return ["shell", "input", "tap", str(x), str(y)]

        if action == ACTION_SCROLL:
            sx, sy = self._normalized_to_absolute(parameters["start_point"])
            ex, ey = self._normalized_to_absolute(parameters["end_point"])
            return ["shell", "input", "swipe", str(sx), str(sy), str(ex), str(ey), "500"]

        if action == ACTION_TYPE:
            text = self._escape_adb_text(parameters["text"])
            return ["shell", "input", "text", text]

        if action == ACTION_OPEN:
            return self._build_open_command(parameters["app_name"])

        if action == ACTION_COMPLETE:
            return []

        raise ValueError(f"不支持的执行动作: {action}")

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

        raise ValueError(f"不支持的执行动作: {action}")

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
            logger.error("ADB 截图超时（10 秒）。")
            return None
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
            logger.error("ADB 截图失败 returncode=%s stderr=%s", exc.returncode, stderr.strip())
            return None
        except Exception as exc:
            logger.error("截图加载失败: %s", exc)
            return None

    def _normalized_to_absolute(self, point: List[int]) -> Tuple[int, int]:
        width, height = self.get_coordinate_space_size()
        x = int(point[0] / 1000.0 * width)
        y = int(point[1] / 1000.0 * height)
        return x, y

    def _build_open_command(self, app_name: str) -> List[str]:
        launch_target = self._resolve_launch_target(app_name)
        if "/" in launch_target:
            return ["shell", "am", "start", "-n", launch_target]
        return [
            "shell",
            "monkey",
            "-p",
            launch_target,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ]

    def _resolve_launch_target(self, app_name: str) -> str:
        app_name = app_name.strip()
        if not app_name:
            raise ValueError("OPEN 动作缺少 app_name")

        if "/" in app_name:
            return app_name

        mapped = self.DEFAULT_LAUNCH_COMPONENTS.get(app_name)
        if mapped:
            return mapped

        # If the planner already returns a package name, launch it with monkey.
        if "." in app_name and " " not in app_name:
            return app_name

        raise ValueError(
            f"未找到应用“{app_name}”的启动配置。"
            "请传入完整组件名（package/.Activity）或在 DEFAULT_LAUNCH_COMPONENTS 中补充映射。"
        )

    @staticmethod
    def _escape_adb_text(text: str) -> str:
        if any(ord(ch) > 127 for ch in text):
            logger.warning("检测到非 ASCII 文本，adb input text 可能无法正确输入中文: %r", text)
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
            logger.warning("列出 adb devices 失败: %s", exc)
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
