# LLM Multi-Agent Gym

Template for simulating multi-agent LLM systems interacting with environments, tools, and each other. The project provides the base abstractions needed to compose agentic workflows while keeping implementation details pluggable.

## Core Abstractions

- **LLM Client (`llm_client.base.LLMClient`)** – Unified interface for connecting to any LLM vendor. Supports completion, chat, and optional streaming calls.
- **Tool (`tools.base.BaseTool`)** – Declarative tool interface with schemas for request/response payloads and a unified `invoke` method.
- **Tool Registry (`tools.tool_registry.ToolRegistry`)** – Stores tools, exposes metadata to agents, and routes invocations to the correct implementation.
- **Tool Executor (`environment.tool_executor.ToolExecutor`)** – Environment-specific executor that pulls tools from the registry and handles execution policies (sync, async, sandboxing, etc.).
- **Agent (`agent.base.Agent`)** – Abstract agent with hooks for observations, tool results, internal state, and environment actions. Each agent holds its own tool registry and LLM client.
- **Agentic Network (`agent.network.AgenticNetwork`)** – Coordinates agent registration, routes messages, handles tool invocations, and advances the environment loop.
- **Environment (`environment.base.Environment`)** – Defines how agent actions transform world state and emit observations back to agents.

These components are intentionally decoupled so you can replace any layer (e.g., swap in a new LLM client or environment) without affecting the rest of the system.

## Demo Flow

`src/main.py` wires a minimal example:

1. Registers an `EchoTool` inside both the environment and agent registries.
2. Instantiates a `DummyLLMClient` that simply mirrors prompts.
3. Creates a `SimpleEnvironment` that relays agent-generated messages back as observations.
4. Runs an `EchoAgent` through a short episode orchestrated by `AgenticNetwork`.

Run the demo:

```bash
python src/main.py
```

You should see observations printed for each environment step. Replace the dummy classes with real implementations to adapt the framework to your project.

## Customisation Checklist

1. **LLM Client** – Implement `LLMClient` to wrap the provider of your choice (OpenAI, Anthropic, self-hosted, etc.).
2. **Tools** – Inherit from `BaseTool`, declare request/response schemas, and register them with each agent and/or environment.
3. **Agents** – Extend `Agent`, consume observations, produce actions/tool calls/messages, and manage internal state or memories.
4. **Environment** – Subclass `Environment` to model world dynamics, produce observations per agent, and set termination conditions.
5. **Network Loop** – Use `AgenticNetwork` as-is or subclass it to customize message routing, scheduling, or reward shaping.

This template focuses on clarity and extensibility instead of domain-specific policy logic. Adapt the interfaces to match your experimentation needs.
