from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportIncompatibleMethodOverride=false
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"WARNING! response_format is not default parameter",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"^munch$",
)

import os
import json
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing import Annotated, Literal, Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.schema import HumanMessage
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
import numpy as np
from prompts import plan_prompt, executor_prompt, agent_system_prompt
from IPython.display import HTML, display

# ──────────────────────────────────────────────────────────────
# Environment setup
# ──────────────────────────────────────────────────────────────
os.environ["TRULENS_OTEL_TRACING"] = "1"
load_dotenv()

# ──────────────────────────────────────────────────────────────
# State definition
# ──────────────────────────────────────────────────────────────
class State(MessagesState):
    enabled_agents: Optional[List[str]]
    plan: Optional[Dict[str, Dict[str, Any]]]
    user_query: Optional[str]
    current_step: int
    replan_flag: Optional[bool]
    last_reason: Optional[str]
    replan_attempts: Optional[Dict[int, int]]
    agent_query: Optional[str]

MAX_REPLANS = 2

# ──────────────────────────────────────────────────────────────
# Tools setup
# ──────────────────────────────────────────────────────────────
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Executes python code for chart generation."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


# ──────────────────────────────────────────────────────────────
# LLM setup
# ──────────────────────────────────────────────────────────────
reasoning_llm = ChatOpenAI(
    model="gpt-4o-mini",
    model_kwargs={"response_format": {"type": "json_object"}},
)

llm = ChatOpenAI(model="gpt-4o")

# ──────────────────────────────────────────────────────────────
# Planner Node
# ──────────────────────────────────────────────────────────────
def planner_node(state: State) -> Command[Literal["executor"]]:
    """Runs the planning LLM and stores the resulting plan in state."""
    llm_reply = reasoning_llm.invoke([plan_prompt(state)])
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed_plan = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Planner returned invalid JSON:\n{llm_reply.content}")

    replan = state.get("replan_flag", False)
    updated_plan: Dict[str, Any] = parsed_plan

    return Command(
        update={
            "plan": updated_plan,
            "messages": [HumanMessage(content=llm_reply.content, name="replan" if replan else "initial_plan")],
            "user_query": state.get("user_query", state["messages"][0].content),
            "current_step": 1 if not replan else state["current_step"],
            "replan_flag": state.get("replan_flag", False),
            "last_reason": "",
            "enabled_agents": state.get("enabled_agents"),
        },
        goto="executor",
    )


# ──────────────────────────────────────────────────────────────
# Executor Node
# ──────────────────────────────────────────────────────────────
def executor_node(
    state: State,
) -> Command[Literal["web_researcher", "chart_generator", "synthesizer", "planner"]]:
    """Decides next step in plan: replan or execute agent."""
    plan: Dict[str, Any] = state.get("plan", {})
    step: int = state.get("current_step", 1)

    # If just replanned, run planned agent once before reconsidering
    if state.get("replan_flag"):
        planned_agent = plan.get(str(step), {}).get("agent")
        return Command(
            update={"replan_flag": False, "current_step": step + 1},
            goto=planned_agent,
        )

    llm_reply = reasoning_llm.invoke([executor_prompt(state)])
    try:
        content_str = llm_reply.content if isinstance(llm_reply.content, str) else str(llm_reply.content)
        parsed = json.loads(content_str)
        replan: bool = parsed["replan"]
        goto: str = parsed["goto"]
        reason: str = parsed["reason"]
        query: str = parsed["query"]
    except Exception as exc:
        raise ValueError(f"Invalid executor JSON:\n{llm_reply.content}") from exc

    updates: Dict[str, Any] = {
        "messages": [HumanMessage(content=llm_reply.content, name="executor")],
        "last_reason": reason,
        "agent_query": query,
    }

    replans: Dict[int, int] = state.get("replan_attempts", {}) or {}
    step_replans = replans.get(step, 0)

    if replan:
        if step_replans < MAX_REPLANS:
            replans[step] = step_replans + 1
            updates.update({
                "replan_attempts": replans,
                "replan_flag": True,
                "current_step": step,
            })
            return Command(update=updates, goto="planner")
        else:
            next_agent = plan.get(str(step + 1), {}).get("agent", "synthesizer")
            updates["current_step"] = step + 1
            return Command(update=updates, goto=next_agent)

    planned_agent = plan.get(str(step), {}).get("agent")
    updates["current_step"] = step + 1 if goto == planned_agent else step
    updates["replan_flag"] = False
    return Command(update=updates, goto=goto)


# ──────────────────────────────────────────────────────────────
# Web Search Agent
# ──────────────────────────────────────────────────────────────
tavily_tool = TavilySearch(max_results=5)

web_search_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=agent_system_prompt(
        "You can ONLY perform research using the provided Tavily search tool. "
        "When you have found the necessary information, end your output. "
        "Do NOT attempt to take further actions."
    ),
)

def web_research_node(state: State) -> Command[Literal["executor"]]:
    agent_query = state.get("agent_query")
    result = web_search_agent.invoke({"messages": agent_query})
    goto = "executor"
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="web_researcher"
    )
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )


# ──────────────────────────────────────────────────────────────
# Chart Generator Agent
# ──────────────────────────────────────────────────────────────
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=agent_system_prompt(
        "You can only generate charts. Print the chart first, then save it to a file and provide the file path."
    ),
)

def chart_node(state: State) -> Command[Literal["chart_summarizer"]]:
    result = chart_agent.invoke(state)
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={"messages": result["messages"]},
        goto="chart_summarizer",
    )


# ──────────────────────────────────────────────────────────────
# Chart Summarizer Agent
# ──────────────────────────────────────────────────────────────
chart_summary_agent = create_react_agent(
    llm,
    tools=[],
    prompt=agent_system_prompt(
        "You can only summarize the chart generated by the chart generator. "
        "Your summary should be concise (≤3 sentences) and should not mention the chart explicitly."
    ),
)

def chart_summary_node(state: State) -> Command[Literal[END]]:
    result = chart_summary_agent.invoke(state)
    print(f"Chart summarizer answer: {result['messages'][-1].content}")
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_summarizer"
    )
    return Command(
        update={
            "messages": result["messages"],
            "final_answer": result["messages"][-1].content,
        },
        goto=END,
    )


# ──────────────────────────────────────────────────────────────
# Synthesizer Agent
# ──────────────────────────────────────────────────────────────
def synthesizer_node(state: State) -> Command[Literal[END]]:
    """Produces a concise final summary of all previous agent outputs."""
    relevant_msgs = [
        m.content for m in state.get("messages", [])
        if getattr(m, "name", None) in ("web_researcher", "chart_generator", "chart_summarizer")
    ]

    user_question = state.get("user_query", state.get("messages", [{}])[0].content if state.get("messages") else "")
    synthesis_instructions = (
        "You are the Synthesizer. Use the context below to answer the user's question directly. "
        "Perform lightweight reasoning or calculations if required. "
        "Do not invent facts. If data is missing, note it and optionally provide a reasoned estimate.\n\n"
        "Guidelines:\n"
        "- Begin with a direct answer (short paragraph or bullet list)\n"
        "- Include any key figures from tables or charts\n"
        "- If citations exist, include them in a 'Citations: [...]' line\n"
        "- Be concise and factual"
    )

    summary_prompt = [
        HumanMessage(content=f"User question: {user_question}\n\n{synthesis_instructions}\n\nContext:\n\n" + "\n\n---\n\n".join(relevant_msgs))
    ]
    llm_reply = llm.invoke(summary_prompt)
    reply_text = str(llm_reply.content).strip()
    print(f"Synthesizer answer: {reply_text}")

    return Command(
        update={
            "final_answer": reply_text,
            "messages": [HumanMessage(content=reply_text, name="synthesizer")],
        },
        goto=END,
    )


# ──────────────────────────────────────────────────────────────
# Evaluation Metrics (Trulens)
# ──────────────────────────────────────────────────────────────
provider = OpenAI(model_engine="gpt-4o")

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

f_plan_quality = (
    Feedback(provider.plan_quality_with_cot_reasons, name="Plan Quality")
    .on({"trace": Selector(trace_level=True)})
)

def display_eval_reason(text, width=800):
    raw_text = str(text).rstrip()
    cleaned_text = re.sub(r"\s*Score:\s*-?\d+(?:\.\d+)?\s*$", "", raw_text, flags=re.IGNORECASE)
    html_text = cleaned_text.replace('\n', '<br><br>')
    display(HTML(f'<div style="font-size: 15px; word-wrap: break-word; width: {width}px;">{html_text}</div>'))
