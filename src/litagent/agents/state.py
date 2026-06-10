"""LangGraph agent state definition."""
from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed between nodes in the ReAct agent graph.

    add_messages is a reducer: returned messages are appended, not replaced.
    This is what makes the conversation history accumulate across the loop.
    """

    messages: Annotated[list[BaseMessage], add_messages]
