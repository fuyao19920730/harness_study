# Agent Harness

生产级 Agent Harness 框架 —— 包裹在大模型外面的"操作系统层"，让 LLM 变成能自主完成任务的智能体。

```
Agent = LLM（大脑） + Harness（操作系统）
```

## 核心能力

- **LLM 抽象层** — 统一接口适配多模型（OpenAI / Anthropic / DeepSeek / 本地模型）
- **工具系统** — `@tool` 装饰器声明工具，自动生成 JSON Schema，沙箱执行
- **规划器** — ReAct / Plan-and-Execute 策略，支持动态重规划
- **记忆系统** — 短期记忆（滑动窗口） + 工作记忆 + 长期记忆（向量检索）
- **安全护栏** — 输入输出过滤、预算控制、工具权限管理
- **可观测性** — 完整执行轨迹（Trace），每一步的 token、延迟、成本全程追踪

## 快速开始

```bash
# 安装
uv pip install -e ".[dev]"

# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 运行最简示例
python -m examples.simple_chat
```

## 使用示例

```python
import asyncio
from harness import Agent, tool

@tool(description="搜索网页")
async def search(query: str) -> str:
    return f"搜索结果: {query}"

async def main():
    async with Agent(
        name="my-agent",
        model="gpt-4o",
        tools=[search],
    ) as agent:
        result = await agent.run("帮我查一下 Python 3.12 的新特性")
        print(result.output)        # Agent 的最终回答
        print(result.trace.summary())  # 执行轨迹摘要

asyncio.run(main())
```

## 架构概览

```
┌─────────────────────────────────────────┐
│              Agent（智能体）              │
│  ┌───────────────────────────────────┐  │
│  │        Harness（操作系统层）        │  │
│  │                                   │  │
│  │  Planner    Memory    Safety      │  │
│  │  规划器      记忆      安全护栏     │  │
│  │                                   │  │
│  │  Tools      Scheduler  Tracer     │  │
│  │  工具管理    调度器     追踪器      │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │          LLM（推理核心）            │  │
│  │   OpenAI / Claude / DeepSeek ...  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## 项目结构

```
harness/
├── harness/
│   ├── agent.py          # Agent 入口类
│   ├── config.py         # 配置加载
│   ├── schema/           # 数据模型（Message, Trace, Config）
│   ├── llm/              # LLM 抽象层 + 各厂商适配器
│   ├── tools/            # 工具系统（注册、执行、内置工具、MCP）
│   ├── planner/          # 规划器（ReAct, Plan-and-Execute）
│   ├── memory/           # 记忆系统（短期 / 工作 / 长期）
│   ├── safety/           # 安全护栏（过滤、预算、权限）
│   ├── scheduler/        # 调度器（顺序 / 并行 / DAG）
│   └── observability/    # 可观测性（Trace、日志、指标）
├── examples/             # 示例代码
├── tests/                # 测试
└── pyproject.toml        # 项目配置
```

## 开发

```bash
# 运行测试
python -m pytest tests/ -v

# 代码检查
ruff check harness/ tests/
```
