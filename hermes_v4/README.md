# Hermes V4 — Autonomous AI Operating System

> Hermes is no longer just a chatbot. It is an autonomous AI operating system capable of planning, selecting tools, executing tasks, validating results, and reporting back.

## Overview

Hermes V4 re-architects the original Hermes (V3) from a **Telegram chatbot** into an **autonomous AI agent** with:

- **Planner** — converts every user request into an executable plan
- **Workflow Engine** — DAG-based multi-step execution with resumption
- **Tool Registry** — extensible, typed tool interface
- **Memory System** — task history, workflow state persistence, context management
- **Executor** — step runner with retry, validation, and structured reporting
- **Event System** — publish/subscribe for logging, notifications, and monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Telegram Interface                   │
│                    (V3 reuse, migrated)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Agent Core (app/agent.py)             │
│                                                          │
│  ┌───────────┐   ┌──────────┐   ┌──────────┐          │
│  │  Planner   │──▶│ Executor │──▶│ Reporter │          │
│  │ (decides)  │   │ (runs)   │   │ (reports)│          │
│  └─────┬─────┘   └────┬─────┘   └──────────┘          │
│        │              │                                  │
│        ▼              ▼                                  │
│  ┌──────────┐   ┌──────────┐                           │
│  │ Workflow │   │  Memory  │                           │
│  │  Engine  │   │  Store   │                           │
│  └──────────┘   └──────────┘                           │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    Tool Registry                         │
│                                                          │
│  ClaudeCodeTool │ GitTool │ TelegramTool │ RAGTool      │
│  TavilyTool     │ OllamaTool │ FileSystem │ Browser (TBD)│
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and setup
cd hermes-v4
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
python -m hermes_v4.app.agent
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434/v1` |
| `MODEL` | Default LLM model | `qwen2.5-coder:14b` |
| `BOT_TOKEN` | Telegram bot token | (required) |
| `TAVILY_API_KEY` | Tavily search API key | (optional) |
| `HERMES_WORKSPACE` | Working directory | `.` |
| `HERMES_MEMORY_BACKEND` | Memory backend: `sqlite`, `redis` | `sqlite` |
| `HERMES_MAX_RETRIES` | Max retry per step | `3` |
| `HERMES_PLAN_TIMEOUT` | Plan generation timeout (s) | `120` |

## Modules

- **`core/`** — Base classes, event system, execution context
- **`planner/`** — Plan generation, step parsing, planning strategies
- **`executor/`** — Step execution, retry logic, validation
- **`workflow/`** — DAG-based workflow engine with DSL
- **`memory/`** — Task history, state persistence, context management
- **`tools/`** — Tool interface, registry, and implementations
- **`app/`** — Main agent loop and optional API layer
- **`telegram/`** — Telegram bot interface (migrated from V3)
- **`rag/`** — RAG pipeline (migrated from V3)
- **`web/`** — Web search abstraction (migrated from V3)
- **`llm/`** — LLM provider abstraction (migrated from V3)

## Key Principles

1. **Planner-driven** — The planner decides which tools to use. Never hardcode routing.
2. **Modular** — Each component is independently testable and replaceable.
3. **Observable** — Every action produces events for logging and monitoring.
4. **Resumable** — Workflows can be paused and resumed from state.
5. **Backward compatible** — V3 RAG, Telegram, Tavily, and Ollama modules are reused.

## Migration from V3

See [MIGRATION.md](../MIGRATION.md) for the complete migration guide.

## Roadmap

See [ROADMAP.md](../ROADMAP.md) for the implementation roadmap.
