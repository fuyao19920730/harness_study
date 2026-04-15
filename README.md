# Agent Harness

**生产级 Agent 框架** — 把 LLM 变成自主智能体的"操作系统层"。

Agent = LLM + **Harness**（规划、记忆、工具、安全、可观测性）

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 🧠 多模型支持 | OpenAI / Anthropic (Claude) / DeepSeek，自动路由 + 降级 |
| 🔧 工具系统 | `@tool` 装饰器一键注册，内置 Shell/HTTP/文件操作，支持 MCP 协议 |
| 📝 规划策略 | ReAct（走一步看一步）+ Plan-and-Execute（先计划后执行） |
| 💾 记忆系统 | 短期记忆（滑动窗口）+ 工作记忆 + 长期记忆（ChromaDB 向量检索） |
| 🛡️ 安全护栏 | 输入/输出过滤、Prompt Injection 检测、工具权限控制、预算限制 |
| 📊 可观测性 | 执行轨迹（Trace）+ JSON 导出 + structlog 结构化日志 |
| 🔄 生产特性 | 自动重试（指数退避）、异常分类、费用估算 |
| 👥 多 Agent 协作 | Supervisor-Worker 模式、DAG 任务调度 |

---

## 快速开始

```bash
# 安装
uv pip install -e ".[dev]"

# 设置 API Key（以 DeepSeek 为例）
export DEEPSEEK_API_KEY="sk-..."

# 运行示例
python -m examples.simple_chat
```

## 使用示例

### 基础：单工具 Agent

```python
import asyncio
from harness import Agent, tool

@tool(description="搜索网页")
async def search(query: str) -> str:
    return f"搜索结果: {query}"

async def main():
    async with Agent(
        name="my-agent",
        model="deepseek-chat",
        tools=[search],
    ) as agent:
        result = await agent.run("帮我查一下 Python 3.12 的新特性")
        print(result.output)
        print(result.trace.summary())

asyncio.run(main())
```

### 进阶：安全护栏 + 预算控制

```python
from harness import Agent, InputGuard, BudgetGuard

agent = Agent(
    name="safe-agent",
    model="gpt-4o",
    input_guard=InputGuard(blocked_patterns=[r"敏感词"]),
    budget_guard=BudgetGuard(max_tokens=50000, max_cost_usd=0.5),
    safety={"max_steps": 10, "require_confirmation": ["shell"]},
)
```

### 进阶：多模型路由

```python
from harness.llm import OpenAILLM, LLMRouter

router = LLMRouter(
    models=[
        OpenAILLM(model="gpt-4o-mini"),   # 便宜的优先
        OpenAILLM(model="gpt-4o"),         # 贵的兜底
    ],
    token_threshold=2000,  # 输入超 2000 token 自动用强力模型
)
agent = Agent(name="smart-agent", llm=router)
```

### 高级：多 Agent 协作

```python
from harness import Agent, tool
from harness.multi import AgentTeam

researcher = Agent(name="researcher", tools=[search], model="gpt-4o")
writer = Agent(name="writer", model="gpt-4o")

team = AgentTeam(
    supervisor=Agent(name="leader", model="gpt-4o"),
    workers={"researcher": researcher, "writer": writer},
)
result = await team.run("调研 AI Agent 框架并写一篇综述")
```

### 高级：DAG 任务调度

```python
from harness.scheduler import DAGScheduler

scheduler = DAGScheduler()
scheduler.add_task("fetch_a", handler=fetch_fn)
scheduler.add_task("fetch_b", handler=fetch_fn)
scheduler.add_task("merge", handler=merge_fn, depends_on=["fetch_a", "fetch_b"])
scheduler.add_task("report", handler=report_fn, depends_on=["merge"])
results = await scheduler.run()  # fetch_a 和 fetch_b 自动并行
```

---

## 架构概览

```
┌──────────────────────────────────────────────────────┐
│                    Agent (编排器)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │  Planner  │ │  Memory  │ │  Safety  │ │ Observe  ││
│  │ ·ReAct    │ │ ·短期    │ │ ·输入过滤│ │ ·Trace   ││
│  │ ·PlanExec │ │ ·工作    │ │ ·输出过滤│ │ ·导出    ││
│  │           │ │ ·长期    │ │ ·权限    │ │ ·日志    ││
│  │           │ │  (向量)  │ │ ·预算    │ │          ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
│  ┌──────────────────────────────────────────────────┐│
│  │              LLM Layer (适配器)                    ││
│  │  OpenAI │ Anthropic │ DeepSeek │ Router │ Retry  ││
│  └──────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────┐│
│  │              Tool System (工具系统)                ││
│  │  @tool装饰器 │ Registry │ 内置工具 │ MCP协议     ││
│  └──────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────┐│
│  │              Multi-Agent (协作)                    ││
│  │  AgentTeam (Supervisor) │ DAG Scheduler           ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

---

## 项目结构

```
harness/
├── agent.py                  # Agent 核心编排器
├── config.py                 # 配置加载（YAML + 环境变量）
├── schema/                   # 数据模型
│   ├── message.py            #   消息、工具调用
│   ├── config.py             #   配置 Schema
│   ├── trace.py              #   执行轨迹
│   └── action.py             #   动作类型
├── llm/                      # LLM 适配层
│   ├── base.py               #   抽象接口
│   ├── openai.py             #   OpenAI / DeepSeek 适配器
│   ├── anthropic.py          #   Claude 原生适配器
│   ├── router.py             #   多模型路由器
│   ├── retry.py              #   自动重试包装器
│   └── exceptions.py         #   统一异常体系
├── tools/                    # 工具系统
│   ├── base.py               #   BaseTool + @tool 装饰器
│   ├── registry.py           #   工具注册表
│   └── builtin/              #   内置工具（shell / http / file）
├── planner/                  # 规划策略
│   ├── base.py               #   抽象接口
│   ├── react.py              #   ReAct 策略
│   └── plan_execute.py       #   Plan-and-Execute 策略
├── memory/                   # 记忆系统
│   ├── base.py               #   抽象接口
│   ├── short_term.py         #   短期记忆（滑动窗口）
│   ├── working.py            #   工作记忆（任务草稿）
│   └── long_term.py          #   长期记忆（ChromaDB / 内存）
├── safety/                   # 安全护栏
│   └── guards.py             #   输入/输出/工具/预算护栏
├── observability/            # 可观测性
│   ├── exporter.py           #   Trace JSON/JSONL 导出
│   └── logging.py            #   structlog 配置
├── multi/                    # 多 Agent 协作
│   └── team.py               #   Supervisor-Worker 团队
├── scheduler/                # 任务调度
│   └── dag.py                #   DAG 调度器
└── mcp/                      # MCP 协议支持
    └── client.py             #   MCP 客户端 + 工具适配

examples/                     # 示例
├── simple_chat.py            # 最简对话
├── tool_usage.py             # 工具调用
├── safety_demo.py            # 安全护栏演示
├── router_demo.py            # 多模型路由演示
├── memory_demo.py            # 长期记忆演示
└── dag_demo.py               # DAG 调度演示

tests/                        # 测试（113+ 项）
```

---

## 运行示例（不需要 API Key）

```bash
# 安全护栏演示
python -m examples.safety_demo

# DAG 调度演示
python -m examples.dag_demo

# LLM Router 演示
python -m examples.router_demo

# 长期记忆演示
python -m examples.memory_demo
```

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 指定模块
python -m pytest tests/test_safety/ -v
python -m pytest tests/test_llm/ -v
```

---

## 技术栈

- **Python 3.12+**
- **pydantic** — 数据验证和序列化
- **openai** — OpenAI / DeepSeek API 客户端
- **anthropic** — Claude API 客户端（可选）
- **structlog** — 结构化日志
- **httpx** — 异步 HTTP 客户端
- **chromadb** — 向量数据库（可选）

## License

MIT
