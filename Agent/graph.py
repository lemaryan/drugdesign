# LangGraph components used to build the workflow
from langgraph.graph import StateGraph, START

# ToolNode executes tools requested by the LLM
# tools_condition decides whether the LLM wants to call a tool or finish
from langgraph.prebuilt import ToolNode, tools_condition

# Shared state that moves through the graph
from Agent.state import AgentState

# The LLM agent node
from Agent.agent import agent_node

# Tools that the agent is allowed to call
from Agent.tools.druglikeness import calculate_druglikeness


# --------------------------------------------------
# 1. Create the graph
# --------------------------------------------------

# AgentState defines the information (messages) shared
# between all nodes in the graph.
builder = StateGraph(AgentState)


# --------------------------------------------------
# 2. Add the agent node
# --------------------------------------------------

# The agent node calls the LLM.
# The LLM can either:
#   1. respond normally, or
#   2. request one of its available tools.
builder.add_node("agent", agent_node)


# --------------------------------------------------
# 3. Add the tool node
# --------------------------------------------------

# ToolNode executes tools requested by the LLM.
# For now, our only tool calculates RDKit molecular properties.
tool_node = ToolNode([calculate_druglikeness])

builder.add_node("tools", tool_node)


# --------------------------------------------------
# 4. Define where the graph starts
# --------------------------------------------------

# Every request first goes to the agent so the LLM
# can decide what action should be taken.
builder.add_edge(START, "agent")


# --------------------------------------------------
# 5. Decide whether a tool should be called
# --------------------------------------------------

# After the agent runs, tools_condition checks its response.
#
# If the LLM requested a tool:
#       agent -> tools
#
# If no tool was requested:
#       agent -> END
builder.add_conditional_edges(
    "agent",
    tools_condition,
)


# --------------------------------------------------
# 6. Return tool results to the agent
# --------------------------------------------------

# After a tool executes, its result is added to the state.
# We return to the agent so the LLM can read the tool result
# and generate a final response for the user.
builder.add_edge("tools", "agent")


# --------------------------------------------------
# 7. Compile the graph
# --------------------------------------------------

# Compile converts our graph definition into an executable graph.
graph = builder.compile()