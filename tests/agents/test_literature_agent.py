"""Tests for the LangGraph ReAct agent — mock LLM, no real API calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from litagent.agents.literature_agent import build_agent, load_system_prompt, run_agent
from litagent.retrieval.schema import RetrievalResponse, RetrievalResult
from litagent.tools.retrieval_tools import make_retrieval_tools


def _make_retrieval_response(paper_ids: list[str]) -> RetrievalResponse:
    return RetrievalResponse(
        query="test",
        results=[
            RetrievalResult(
                chunk_id=f"{pid}__chunk0",
                paper_id=pid,
                text=f"Evidence from {pid}.",
                score=0.9,
                sources=["vector"],
            )
            for pid in paper_ids
        ],
        vector_candidates=len(paper_ids),
        bm25_candidates=0,
        graph_papers_found=0,
    )


@pytest.fixture()
def mock_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever._cfg.chroma.collection_name = "papers"
    retriever.retrieve.return_value = _make_retrieval_response(["W001"])
    return retriever


def _mock_llm_with_responses(responses: list[AIMessage]) -> MagicMock:
    """Build a mock LLM whose bind_tools() returns a mock that cycles through responses."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_bound = MagicMock()
    mock_bound.invoke.side_effect = responses
    mock_llm.bind_tools.return_value = mock_bound
    return mock_llm


class TestBuildAgent:
    def test_direct_answer_reaches_end(self, mock_retriever: MagicMock) -> None:
        """If the LLM returns a plain message (no tool calls), agent goes straight to END."""
        tools = make_retrieval_tools(mock_retriever)
        llm = _mock_llm_with_responses(
            [AIMessage(content="Direct answer with no tool calls.")]
        )
        agent = build_agent(tools, llm)

        result = agent.invoke(
            {"messages": [HumanMessage(content="What is RAG?")]}
        )

        last = result["messages"][-1]
        assert "Direct answer" in last.content

    def test_tool_call_then_answer(self, mock_retriever: MagicMock) -> None:
        """Agent calls vector_search tool, gets result, then produces final answer."""
        tools = make_retrieval_tools(mock_retriever)
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "vector_search",
                    "args": {"query": "hallucination methods"},
                    "id": "call_test_1",
                    "type": "tool_call",
                }
            ],
        )
        final_answer_msg = AIMessage(content="Based on evidence from W001, the answer is X.")
        llm = _mock_llm_with_responses([tool_call_msg, final_answer_msg])
        agent = build_agent(tools, llm)

        result = agent.invoke(
            {"messages": [HumanMessage(content="How do RAG systems handle hallucination?")]}
        )

        messages = result["messages"]
        assert any(hasattr(m, "tool_calls") and m.tool_calls for m in messages), \
            "Expected at least one tool call in message history"
        assert "Based on evidence" in messages[-1].content

    def test_retriever_was_called(self, mock_retriever: MagicMock) -> None:
        """Verify the tool actually called the retriever during the agent run."""
        tools = make_retrieval_tools(mock_retriever)
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "vector_search",
                    "args": {"query": "test query"},
                    "id": "call_test_2",
                    "type": "tool_call",
                }
            ],
        )
        llm = _mock_llm_with_responses([tool_call_msg, AIMessage(content="Done.")])
        agent = build_agent(tools, llm)

        agent.invoke({"messages": [HumanMessage(content="test")]})

        mock_retriever.retrieve.assert_called_once()
        call_arg = mock_retriever.retrieve.call_args[0][0]
        assert call_arg.text == "test query"

    def test_message_history_accumulates(self, mock_retriever: MagicMock) -> None:
        """Messages from each node step are appended, not replaced."""
        tools = make_retrieval_tools(mock_retriever)
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "vector_search",
                    "args": {"query": "evidence"},
                    "id": "call_test_3",
                    "type": "tool_call",
                }
            ],
        )
        llm = _mock_llm_with_responses([tool_call_msg, AIMessage(content="Final.")])
        agent = build_agent(tools, llm)

        result = agent.invoke({"messages": [HumanMessage(content="question")]})

        # Expect: HumanMessage, AIMessage(tool_call), ToolMessage, AIMessage(final)
        assert len(result["messages"]) >= 4


class TestRunAgent:
    def test_returns_final_answer_string(self, mock_retriever: MagicMock) -> None:
        tools = make_retrieval_tools(mock_retriever)
        llm = _mock_llm_with_responses([AIMessage(content="The answer is 42.")])
        agent = build_agent(tools, llm)

        answer = run_agent("What is the answer?", agent, "You are helpful.")

        assert answer == "The answer is 42."

    def test_system_prompt_is_first_message(self, mock_retriever: MagicMock) -> None:
        """System prompt and question are both present in the messages sent to LLM."""
        tools = make_retrieval_tools(mock_retriever)
        captured_messages: list[list[object]] = []

        mock_bound = MagicMock()
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.bind_tools.return_value = mock_bound

        def capture_and_respond(messages: object) -> AIMessage:
            captured_messages.append(messages)  # type: ignore[arg-type]
            return AIMessage(content="Answer.")

        mock_bound.invoke.side_effect = capture_and_respond
        agent = build_agent(tools, mock_llm)

        run_agent("My question.", agent, "System instructions here.")

        first_call = captured_messages[0]
        assert isinstance(first_call[0], SystemMessage)
        assert "System instructions" in first_call[0].content
        assert isinstance(first_call[1], HumanMessage)
        assert "My question." in first_call[1].content


class TestLoadSystemPrompt:
    def test_reads_file_content(self, tmp_path: pytest.TempdirFactory) -> None:
        p = tmp_path / "system.txt"  # type: ignore[operator]
        p.write_text("You are a helpful assistant.", encoding="utf-8")

        content = load_system_prompt(str(p))

        assert content == "You are a helpful assistant."

    def test_strips_trailing_whitespace(self, tmp_path: pytest.TempdirFactory) -> None:
        p = tmp_path / "system.txt"  # type: ignore[operator]
        p.write_text("Content\n\n", encoding="utf-8")

        assert load_system_prompt(str(p)) == "Content"
