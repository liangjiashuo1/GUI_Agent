"""
一键启动脚本 - 自动设置环境变量并运行测试。
直接双击或命令行: python run.py
"""
import os
import sys

# 设置环境变量（必须在导入 agent_base 之前）
os.environ["VLM_API_KEY"] = "sk-661cbeffbbb440e0bb1c47e6c6b79ecf"
os.environ["DEBUG_API_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
os.environ["DEBUG_MODEL_ID"] = "qwen-vl-plus"

print("=" * 50)
print("  GUI Agent 测试")
print(f"  API: {os.environ['DEBUG_API_URL']}")
print(f"  Model: {os.environ['DEBUG_MODEL_ID']}")
print("=" * 50)
print()

from test_runner import TestRunner
from agent import Agent

# 只跑一个测试用例
case_path = "./test_data/offline/step_aiqiyi_onekey_0011"
if len(sys.argv) > 1:
    case_path = sys.argv[1]

print(f"测试用例: {case_path}\n")

agent = Agent()
runner = TestRunner(agent, debug_test=True)
result = runner.run_task(case_path, "./output/visualization")

steps = result["steps"]
passed = sum(1 for s in steps if s["check_result"])
total = len(steps)

print(f"\n{'=' * 50}")
print(f"结果: {passed}/{total} 步通过 ({passed/total*100:.0f}%)" if total else "")
for i, s in enumerate(steps):
    status = "PASS" if s["check_result"] else "FAIL"
    params = str(s.get("action_parameter", ""))
    if len(params) > 60:
        params = params[:60] + "..."
    print(f"  Step {i+1}: {s['action']:8s} {params:60s} -> {status}")
print("=" * 50)
