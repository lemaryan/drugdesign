from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Represents the state of an agent in a conversation."""

    messages: Annotated[list, add_messages]