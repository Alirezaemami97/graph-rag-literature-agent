"""Single LangGraph ReAct agent for literature research."""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode

from litagent.agents.state import AgentState
from litagent.config import Config


def build_agent(tools: list[BaseTool], llm: BaseChatModel) -> CompiledGraph:
    """Compile the ReAct state graph: agent → tools → agent → … → END."""
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


def create_llm(config: Config) -> AzureChatOpenAI:
    """Instantiate the Azure OpenAI chat model from environment + config."""
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],  # type: ignore[arg-type]
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature=config.agent.temperature,
    )


def load_system_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def run_agent(question: str, agent: CompiledGraph, system_prompt: str) -> str:
    """Run the agent on a single question and return the final answer string."""
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ]
        }
    )
    return str(result["messages"][-1].content)
