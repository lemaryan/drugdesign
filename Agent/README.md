# Merck Drug Optimization Agent

This folder contains the LangGraph-based agent for the Merck drug optimization project.

The agent uses an LLM to interpret a user's request and decide when to call molecular analysis tools such as RDKit. Tool results are returned to the LLM so it can generate a final response.

## Current Workflow

User Request
    ↓
Agent (Qwen3)
    ↓
Does the request require a tool?
    ├── No  → Return response
    │
    └── Yes → ToolNode
                  ↓
              Execute tool
                  ↓
              Return result
                  ↓
                 Agent
                  ↓
            Final response

LangGraph controls the workflow and routing between the LLM agent and the available tools.

## Folder Structure

Agent/
├── agent.py
├── graph.py
├── state.py
├── README.md
└── tools/
    ├── __init__.py
    ├── druglikeness.py
    ├── docking.py
    └── potency.py

### `agent.py`

Creates the LLM agent using Qwen3 through the NRP OpenAI-compatible API.

The available tools are bound to the LLM using LangChain's `bind_tools()`. This allows the model to decide whether a tool should be called and what arguments should be passed to it.

### `graph.py`

Defines the LangGraph workflow.

The current graph contains:

- `agent` node — sends the conversation to the LLM
- `tools` node — executes tools requested by the LLM
- conditional routing — determines whether the agent should call a tool or finish

The basic flow is:

START → Agent → Tool (if needed) → Agent → END

### `state.py`

Defines the shared state passed between LangGraph nodes.

Currently the state stores the conversation messages. `add_messages` allows new messages, tool calls, and tool results to be appended to the conversation history.

### `tools/druglikeness.py`

Contains the current RDKit molecular-property tool.

Given a SMILES string, it calculates:

- Molecular weight
- LogP
- TPSA
- Hydrogen-bond donors
- Hydrogen-bond acceptors

Example input:

`CCO`

The agent can automatically recognize when molecular properties are needed and call this tool.

## API Configuration

The NRP API key should NOT be stored directly in the source code.

Set it as an environment variable:

```bash
read -s NRP_API_KEY
export NRP_API_KEY