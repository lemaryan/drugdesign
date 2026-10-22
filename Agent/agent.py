import os
from langchain_openai import ChatOpenAI

from Agent.state import AgentState
from Agent.tools.druglikeness import calculate_druglikeness

llm = ChatOpenAI(
    model = "qwen3",
    api_key = os.environ["NRP_API_KEY"],
    base_url="https://ellm.nrp-nautilus.io/v1",
)

tools = [calculate_druglikeness]

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}