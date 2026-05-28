"""
GUI Agent 在线测试脚本 - run_online.py

一键启动脚本，支持在线连接 ADB 模拟器进行动态测试。

主要功能：
单次任务模式：python run_online.py --instruction "打开爱奇艺并搜索狂飙"

此脚本模仿 run.py，但支持在线测试：
- 从 ADB 设备实时获取截图
- 执行实际的 ADB 命令
- 集成所有 A/B/C/D/E 模块
- 支持日志记录到文件和控制台
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any

# 设置环境变量（必须在导入 agent_base 之前）
os.environ["VLM_API_KEY"] = "sk-661cbeffbbb440e0bb1c47e6c6b79ecf"
os.environ["DEBUG_API_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["DEBUG_MODEL_ID"] = "qwen-vl-plus"

# 配置日志 - 重用 test_runner.py 的配置
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# 清除已有 handlers 后重新配置，确保 FileHandler 生效
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{output_dir}/online_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("  GUI Agent 在线测试 (ADB 连接)")
logger.info(f"  API: {os.environ['DEBUG_API_URL']}")
logger.info(f"  Model: {os.environ['DEBUG_MODEL_ID']}")
logger.info("=" * 60)

from agent_base import AgentInput, AgentOutput, UsageInfo
from agent import Agent
from gui_agent.a_executor.main import ExecutorModule
from gui_agent.shared.schemas import PlannerDecision


class OnlineTestRunner:
    """在线测试运行器 - 从 ADB 设备获取截图并执行动作"""
    
    def __init__(self, 
                 agent: Agent,
                 executor: ExecutorModule,
                 max_steps: int = 30,
                 debug_test: bool = True):
        """
        初始化在线测试运行器
        
        Args:
            agent: Agent 实例
            executor: ADB 执行器实例
            max_steps: 最大执行步数
            debug_test: 是否在错误时继续执行
        """
        self.agent = agent
        self.executor = executor
        self.max_steps = max_steps
        self.debug_test = debug_test
        
        # Token 消耗监控
        self._total_tokens = 0
        self._max_total_tokens = 1200000
        
        # 历史记录
        self.history_messages = []
        self.history_actions = []
        
    def _check_token_limit(self, usage: UsageInfo) -> None:
        """检查 token 使用量是否超过限制"""
        if usage:
            self._total_tokens += usage.total_tokens
            logger.info(f"Token usage: +{usage.total_tokens} (total: {self._total_tokens}/{self._max_total_tokens})")
            
            if self._total_tokens > self._max_total_tokens:
                logger.error(f"Token 限制超出: {self._total_tokens}/{self._max_total_tokens}")
                raise Exception(f"Token 限制超出: {self._total_tokens}/{self._max_total_tokens}")
    
    def _encode_image_for_history(self, image) -> str:
        """将图片编码为 base64 URL（用于历史消息）"""
        import io
        import base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    
    def run_single_task(self, instruction: str) -> Dict[str, Any]:
        """
        执行单个任务
        
        Args:
            instruction: 用户指令
            
        Returns:
            包含测试结果的字典
        """
        # 重置 Agent 状态
        try:
            self.agent.reset()
            logger.info("Agent 状态重置成功")
        except Exception as e:
            logger.error(f"Agent reset 失败: {e}")
        
        logger.info(f"开始执行任务: {instruction}")
        logger.info("-" * 60)
        
        step_count = 1
        steps_record = []
        
        while step_count <= self.max_steps:
            logger.info(f"--- 第 {step_count} 步 ---")
            
            # 1. 从 ADB 设备获取截图
            logger.info("正在从 ADB 设备获取截图...")
            screenshot = self.executor.capture_screenshot()
            if screenshot is None:
                logger.error("无法从 ADB 设备获取截图")
                break
            
            logger.info(f"截图尺寸: {screenshot.width} x {screenshot.height}")
            
            # 2. 准备 AgentInput
            agent_input = AgentInput(
                instruction=instruction,
                current_image=screenshot,
                step_count=step_count,
                history_messages=self.history_messages,
                history_actions=self.history_actions
            )
            
            # 3. 调用 Agent
            try:
                agent_output = self.agent.act(agent_input)
                logger.info(f"Agent 输出: 动作={agent_output.action}, 参数={agent_output.parameters}")
                
                # 检查 token 使用量
                if agent_output.usage:
                    self._check_token_limit(agent_output.usage)
                    
            except Exception as e:
                logger.error(f"Agent 执行错误: {e}")
                # 创建一个失败的输出
                agent_output = AgentOutput(
                    action="",
                    parameters={},
                    raw_output=f"错误: {str(e)}"
                )
            
            # 4. 执行 ADB 命令
            execution_success = False
            if agent_output.action and agent_output.action != "COMPLETE":
                logger.info(f"执行 ADB 命令: {agent_output.action}")
                # 将 AgentOutput 转换为 PlannerDecision 格式
                decision = PlannerDecision(
                    action=agent_output.action,
                    parameters=agent_output.parameters,
                    thought=agent_output.raw_output or ""
                )
                execution_success = self.executor.execute_adb_command(decision)
                logger.info(f"执行结果: {'成功' if execution_success else '失败'}")
            elif agent_output.action == "COMPLETE":
                logger.info("任务完成")
                execution_success = True
            
            # 5. 更新历史
            screenshot_base64 = self._encode_image_for_history(screenshot)
            self.history_messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": screenshot_base64}}
                ]
            })
            self.history_messages.append({
                "role": "assistant",
                "content": f"Action: {agent_output.action}({agent_output.parameters})"
            })
            
            self.history_actions.append({
                "step": step_count,
                "action": agent_output.action,
                "parameters": agent_output.parameters,
                "raw_output": agent_output.raw_output,
                "execution_success": execution_success
            })
            
            # 6. 记录结果
            step_record = {
                "step": step_count,
                "action": agent_output.action,
                "parameters": agent_output.parameters,
                "raw_output": agent_output.raw_output,
                "execution_success": execution_success,
                "screenshot_size": (screenshot.width, screenshot.height)
            }
            steps_record.append(step_record)
            
            # 7. 检查是否完成任务
            if agent_output.action == "COMPLETE":
                logger.info("任务标记为完成")
                break
            
            # 等待短暂时间让设备响应
            logger.info("等待设备响应...")
            time.sleep(1.0)
            
            step_count += 1
        
        logger.info("=" * 60)
        logger.info(f"任务执行完成")
        logger.info(f"总步数: {len(steps_record)}")
        
        # 统计执行成功率
        successful_steps = sum(1 for s in steps_record if s.get("execution_success", False))
        completion_rate = successful_steps / len(steps_record) if steps_record else 0
        
        logger.info(f"成功执行: {successful_steps}/{len(steps_record)} ({completion_rate:.0%})")
        logger.info("=" * 60)
        
        return {
            "instruction": instruction,
            "steps": steps_record,
            "total_steps": len(steps_record),
            "successful_steps": successful_steps,
            "completion_rate": completion_rate
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GUI Agent 在线测试 - 支持 ADB 连接',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_online.py --instruction "打开微信"                    # 执行单个任务
  python run_online.py --device 127.0.0.1:16384 --instruction "打开抖音"  # 指定设备
  python run_online.py --adb_path /path/to/adb --instruction "搜索好友"   # 指定ADB路径
  
ADB 设备配置:
  - 可以通过 --device 参数指定设备序列号
  - 可以通过 --adb_path 参数指定 adb 路径
  - 可以通过环境变量 ADB_DEVICE_SERIAL 和 ADB_PATH 设置
        """
    )
    
    parser.add_argument(
        '--instruction', '-i',
        type=str,
        required=True,
        help='要执行的指令（必需）'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='ADB 设备序列号 (默认: 自动检测)'
    )
    
    parser.add_argument(
        '--adb_path', '-a',
        type=str,
        default=None,
        help='ADB 可执行文件路径 (默认: 自动检测)'
    )
    
    parser.add_argument(
        '--max_steps', '-m',
        type=int,
        default=30,
        help='最大执行步数 (默认: 30)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    try:
        # 创建 Agent 实例
        logger.info("初始化 Agent...")
        agent = Agent()
        
        # 创建 Executor 实例
        logger.info("初始化 ADB 执行器...")
        executor = ExecutorModule(
            adb_path=args.adb_path,
            device_serial=args.device
        )
        
        # 检查 ADB 连接
        logger.info(f"连接设备: {executor._device_serial}")
        logger.info(f"ADB 路径: {executor._adb_path}")
        
        # 测试屏幕截图
        logger.info("测试 ADB 连接...")
        test_screenshot = executor.capture_screenshot()
        if test_screenshot is None:
            logger.error("错误: 无法从 ADB 设备获取截图，请检查连接")
            return 1
        
        screen_size = executor.get_screen_size()
        logger.info(f"设备屏幕尺寸: {screen_size[0]} x {screen_size[1]}")
        logger.info(f"测试截图尺寸: {test_screenshot.width} x {test_screenshot.height}")
        
        # 创建在线测试运行器
        runner = OnlineTestRunner(
            agent=agent,
            executor=executor,
            max_steps=args.max_steps,
            debug_test=True
        )
        
        # 运行测试
        result = runner.run_single_task(args.instruction)
        
        # 显示详细结果
        logger.info("\n详细执行记录:")
        logger.info("-" * 60)
        for step in result["steps"]:
            status = "✓" if step.get("execution_success", False) else "✗"
            action = step["action"] or "无动作"
            params = step["parameters"] or {}
            logger.info(f"  第 {step['step']} 步 [{status}]: {action} {params}")
        
        logger.info("\n" + "=" * 60)
        logger.info(f"任务完成!")
        logger.info(f"指令: {args.instruction}")
        logger.info(f"总步数: {result['total_steps']}")
        logger.info(f"成功执行: {result['successful_steps']}/{result['total_steps']} ({result['completion_rate']:.0%})")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\n用户中断")
        return 130
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())