"""
GUI Agent 模块包。

本目录用于承载按 A/B/C/D/E 分工拆分后的实现：
- A: executor.py，动作编译与设备执行抽象
- B: perception.py，屏幕理解
- C: planner.py，下一步动作决策
- D: memory_reflection.py，状态跟踪与错误恢复
- E: integration_safety.py，主循环、安全边界、日志与联调
- shared: schemas.py，公共数据结构与接口契约
"""

