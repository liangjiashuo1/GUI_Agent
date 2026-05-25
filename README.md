# GUI Agent 协作开发说明

当前项目已经提供了评测底座：

- `agent_base.py`
  这里定义了 `BaseAgent`、`AgentInput`、`AgentOutput`，以及评测要求的标准动作格式。
- `test_runner.py`
  这里会直接 `from agent import Agent`，然后不断调用 `Agent.act(...)` 做离线评测。

所以我们现在要做的，不是改评测器，而是在这个仓库里搭一套按 `A / B / C / D / E` 分工的 GUI Agent 工程结构，并且保证最终仍然能从 `agent.py` 对外工作。

## 当前目录结构

我已经按照“每个人一个工作目录”的方式完成了初步拆分，并且每个目录都提供了一个 `main.py` 供上层统一调用。

```text
code-for-student/
├── agent.py
├── agent_base.py
├── test_runner.py
├── README.md
└── gui_agent/
    ├── __init__.py
    ├── a_executor/
    │   ├── __init__.py
    │   └── main.py
    ├── b_perception/
    │   ├── __init__.py
    │   └── main.py
    ├── c_planner/
    │   ├── __init__.py
    │   └── main.py
    ├── d_memory_reflection/
    │   ├── __init__.py
    │   └── main.py
    ├── e_integration_safety/
    │   ├── __init__.py
    │   └── main.py
    ├── shared/
    │   ├── __init__.py
    │   └── schemas.py
    ├── executor.py
    ├── perception.py
    ├── planner.py
    ├── memory_reflection.py
    ├── integration_safety.py
    └── schemas.py
```

说明：

- `gui_agent/a_executor/main.py` 到 `gui_agent/e_integration_safety/main.py` 是新的推荐开发入口。
- `gui_agent/shared/schemas.py` 是共享数据结构和接口契约文件。
- `gui_agent/executor.py`、`gui_agent/perception.py` 等旧的平铺文件现在是兼容导入层，目的是避免以后改目录时影响上层代码。

## 上层调用方式

推荐使用新的目录入口进行导入：

```python
from gui_agent.a_executor.main import ExecutorModule
from gui_agent.b_perception.main import PerceptionModule
from gui_agent.c_planner.main import PlannerModule
from gui_agent.d_memory_reflection.main import MemoryReflectionModule
from gui_agent.e_integration_safety.main import IntegratedGUIAgentController
from gui_agent.shared.schemas import *
```

最终对评测器暴露的入口仍然是：

```python
from agent import Agent
```

## 每个人的工作目录与职责

### A：Executor，负责手机控制

工作目录：

- `gui_agent/a_executor/`

主入口文件：

- `gui_agent/a_executor/main.py`

需要完成的功能：

- 把 `PlannerDecision` 转成标准 `AgentOutput`
- 严格校验动作参数格式
- 为未来真机控制预留设备命令接口
- 对坐标做最后一层兜底修正

主接口：

```python
class ExecutorModule:
    def compile_decision(self, decision: PlannerDecision) -> AgentOutput: ...
    def build_device_command(self, decision: PlannerDecision) -> dict: ...
```

必须严格遵守的动作格式：

```python
CLICK: {"point": [x, y]}
SCROLL: {"start_point": [x1, y1], "end_point": [x2, y2]}
TYPE: {"text": "..."}
OPEN: {"app_name": "..."}
COMPLETE: {}
```

### B：Perception，负责看懂屏幕

工作目录：

- `gui_agent/b_perception/`

主入口文件：

- `gui_agent/b_perception/main.py`

需要完成的功能：

- 输入截图，输出结构化屏幕理解结果
- 识别页面类型、关键控件、是否弹窗、是否有键盘
- 将多模态模型输出转成 `ScreenPerception`
- 尽量输出稳定结构化结果，不要只返回自由文本

主接口：

```python
class PerceptionModule:
    def perceive(self, input_data: AgentInput, memory: MemoryState, call_llm=None) -> ScreenPerception: ...
    def build_perception_prompt(self, input_data: AgentInput, memory: MemoryState) -> list[dict]: ...
    def parse_perception_response(self, raw_text: str) -> ScreenPerception: ...
```

建议保证：

- 所有坐标统一使用 `[0, 1000]` 归一化坐标
- 所有框统一使用 `[x1, y1, x2, y2]`
- 识别不准时宁可返回空元素，也不要编造按钮

### C：Planner，负责下一步动作决策

工作目录：

- `gui_agent/c_planner/`

主入口文件：

- `gui_agent/c_planner/main.py`

需要完成的功能：

- 根据任务目标、屏幕理解和记忆状态决定下一步动作
- 输出标准 `PlannerDecision`
- 支持 `CLICK / SCROLL / TYPE / OPEN / COMPLETE`
- 尽量以结构化 JSON 作为模型输出中间层

主接口：

```python
class PlannerModule:
    def plan_next_action(
        self,
        input_data: AgentInput,
        perception: ScreenPerception,
        memory: MemoryState,
        call_llm=None,
    ) -> PlannerDecision: ...
```

建议保证：

- `action` 必须是全大写标准动作名
- `parameters` 必须直接符合 `agent_base.py` 的要求
- 没有把握时不要乱猜，宁可交给 E 做保守兜底

### D：Memory / Reflection，负责状态跟踪和错误恢复

工作目录：

- `gui_agent/d_memory_reflection/`

主入口文件：

- `gui_agent/d_memory_reflection/main.py`

需要完成的功能：

- 从 `history_actions` 重建当前状态
- 记录历史动作、阶段目标、失败尝试
- 检测死循环、重复点击、无效重复尝试
- 给上层提供恢复建议和禁止重复动作信息

主接口：

```python
class MemoryReflectionModule:
    def reset(self, task_goal: str | None = None) -> None: ...
    def bootstrap(self, input_data: AgentInput) -> MemoryState: ...
    def update_after_decision(self, perception: ScreenPerception, decision: PlannerDecision) -> MemoryState: ...
    def reflect(self, decision: PlannerDecision) -> MemoryState: ...
    def should_block_repeat(self, action: str, parameters: dict) -> bool: ...
```

建议保证：

- 能识别重复动作和短期循环
- 错误要有明确标签
- 恢复建议要能被 C/E 消费，而不是只写日志

### E：Integration / Safety，负责主循环、安全边界、日志和演示

工作目录：

- `gui_agent/e_integration_safety/`

主入口文件：

- `gui_agent/e_integration_safety/main.py`

需要完成的功能：

- 编排 A / B / C / D 四个模块
- 承接 `BaseAgent` 的 `act` / `reset`
- 做动作合法性检查和保守兜底
- 负责日志、回放和整体调度

主接口：

```python
class IntegratedGUIAgentController:
    def reset(self) -> None: ...
    def run_step(self, input_data: AgentInput) -> AgentOutput: ...
    def make_fallback_decision(self, reason: str) -> PlannerDecision: ...
```

建议保证：

- 主流程稳定
- 任一模块出错时，系统仍然能返回一个安全动作
- 日志里能看出“看到了什么、为什么这么做、最后输出了什么”

## 共享接口契约

共享目录：

- `gui_agent/shared/`

共享文件：

- `gui_agent/shared/schemas.py`

这个文件应该被视为团队公共协议文件，建议由 E 先定稿，A/B/C/D 全部对齐。

关键共享类：

- `UIElement`
- `ScreenPerception`
- `PlannerDecision`
- `StepRecord`
- `ErrorRecord`
- `MemoryState`
- `SafetyCheckResult`

统一约定：

- 所有坐标都使用 `[0, 1000]`
- 所有元素框都使用 `[x1, y1, x2, y2]`
- 所有动作必须是 `CLICK / SCROLL / TYPE / OPEN / COMPLETE`

## 当前总入口

项目对外入口文件：

- `agent.py`

这里的职责是：

- 继承 `BaseAgent`
- 把 `AgentInput` 交给 E 模块
- 保持与 `test_runner.py` 完全兼容

接口形式：

```python
class Agent(BaseAgent):
    def act(self, input_data: AgentInput) -> AgentOutput: ...
    def reset(self) -> None: ...
```

## 推荐开发顺序

建议按下面顺序推进，这样大家不容易互相卡住：

1. E 先确认 `gui_agent/shared/schemas.py` 的字段名
2. A 先保证标准动作输出永远合法
3. D 先完成历史重建和重复动作检测
4. B 再接入 VLM，把截图解析成稳定结构化结果
5. C 基于 B 和 D 的结果做真正的动作规划
6. E 最后统一加安全检查、日志、兜底和演示能力

## 当前状态

已经完成的部分：

- 每个人都有独立工作目录
- 每个工作目录都有自己的 `main.py`
- `agent.py` 已经接到 `gui_agent/e_integration_safety/main.py`
- 旧的平铺模块保留为兼容导入层

目前仍然只是骨架 / baseline 的部分：

- B 的屏幕理解还是占位实现
- C 的规划逻辑还是保守 baseline
- D 的反思逻辑还比较轻量
- A 还没有接真实设备执行

## 运行方式

可以先用下面命令验证导入和主流程是否正常：

```bash
python test_runner.py
```

说明：

- 当前重点是“模块拆分、目录划分、接口约定”
- 现在这版不是最终比赛策略
- 你们后续只需要在各自目录里的 `main.py` 继续实现，不需要再重新拆工程结构
