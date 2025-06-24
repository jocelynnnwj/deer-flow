# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.prompts.planner_model import StepType

from .types import State
from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    coder_node,
    human_feedback_node,
    background_investigation_node,
)
from .image_generator_node import image_generator_node
from .speech_generator_node import speech_generator_node


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not hasattr(current_plan, "steps") or not isinstance(getattr(current_plan, "steps", None), list):
        print("[continue_to_running_research_team] No current_plan or steps, return 'planner'")
        return "planner"
    if all(step.execution_res for step in current_plan.steps):
        print("[continue_to_running_research_team] All steps executed, return 'planner'")
        return "planner"
    for step in current_plan.steps:
        if not step.execution_res:
            break
    print(f"[continue_to_running_research_team] Next step_type: {getattr(step, 'step_type', None)}")
    if step.step_type and step.step_type == StepType.RESEARCH:
        print("[continue_to_running_research_team] Return 'researcher'")
        return "researcher"
    if step.step_type and step.step_type == StepType.PROCESSING:
        print("[continue_to_running_research_team] Return 'coder'")
        return "coder"
    if step.step_type and step.step_type == StepType.IMAGE_GENERATION:
        print("[continue_to_running_research_team] Return 'image_generator'")
        state["step"] = step
        return "image_generator"
    if step.step_type and step.step_type == StepType.SPEECH_GENERATION:
        print("[continue_to_running_research_team] Return 'speech_generator'")
        state["step"] = step
        return "speech_generator"
    print("[continue_to_running_research_team] Fallback return 'planner'")
    return "planner"


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("image_generator", image_generator_node)
    builder.add_node("speech_generator", speech_generator_node)
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder", "image_generator", "speech_generator"],
    )
    builder.add_edge("human_feedback", "research_team")
    builder.add_edge("human_feedback", "image_generator")
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()
