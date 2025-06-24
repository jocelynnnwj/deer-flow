# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Annotated, Literal
import re
from collections.abc import Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, Tool
from langgraph.types import Command, interrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.agents import create_agent
from src.tools.search import LoggedTavilySearch
from src.tools import (
    crawl_tool,
    get_web_search_tool,
    get_retriever_tool,
    python_repl_tool,
)

from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.llms.llm import get_llm_by_type
from src.prompts.planner_model import Plan, StepType, Step
from src.prompts.template import apply_prompt_template
from src.utils.json_utils import repair_json_output

from .types import State
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

import langgraph
print("LANGGRAPH PATH:", langgraph.__file__)
from langgraph.types import interrupt
print("INTERRUPT FUNC:", interrupt)

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    query = state.get("research_topic")
    background_investigation_results = None
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        if isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            return {
                "background_investigation_results": "\n\n".join(
                    background_investigation_results
                )
            }
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def extract_json_from_codeblock(text):
    """Extract JSON from code block if present."""
    match = re.search(r"```(?:json)?\\n([\\s\\S]+?)```", text)
    if match:
        return match.group(1)
    return text


def extract_json_object(text):
    """Extract the first JSON object from a string using regex."""
    match = re.search(r'{.*}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text  # fallback: return original text


def state_to_dict(state):
    if hasattr(state, "__dict__"):
        d = vars(state).copy()
    else:
        d = dict(state)
    if 'messages' in d and not isinstance(d['messages'], list):
        if isinstance(d['messages'], Iterable) and not isinstance(d['messages'], (str, bytes, dict)):
            d['messages'] = list(d['messages'])
        else:
            d['messages'] = []
    return d


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info(f"[planner_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Planner generating full plan")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    state_dict = state_to_dict(state)
    messages = apply_prompt_template("planner", state_dict, configurable)

    logger.info(f"Planner prompt: {messages}")

    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        if isinstance(messages, list):
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "background investigation results of user query:\n"
                        + state["background_investigation_results"]
                        + "\n"
                    ),
                }
            ]
        else:
            messages = [messages] + [
                {
                    "role": "user",
                    "content": (
                        "background investigation results of user query:\n"
                        + state["background_investigation_results"]
                        + "\n"
                    ),
                }
            ]

    if configurable.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
        # Try to use structured output if available
        if hasattr(llm, "with_structured_output"):
            # Only pass method='json_mode' for non-Gemini LLMs
            if llm.__class__.__name__ == "ChatGoogleGenerativeAI":
                llm = llm.with_structured_output(Plan)
            else:
                llm = llm.with_structured_output(Plan, method="json_mode")

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(goto="reporter")

    full_response = ""
    if AGENT_LLM_MAP["planner"] == "basic" and not configurable.enable_deep_thinking:
        response = llm.invoke(messages)
        if hasattr(response, "model_dump_json") and callable(getattr(response, "model_dump_json", None)) and not isinstance(response, dict):
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            full_response = str(response)
    else:
        response = llm.stream(messages)
        full_response = ""
        for chunk in response:
            logger.info(f"LLM stream chunk: {chunk}")
            # Pydantic v2
            if hasattr(chunk, "model_dump_json") and callable(chunk.model_dump_json):
                full_response += str(chunk.model_dump_json())
            # Pydantic v1
            elif hasattr(chunk, "dict") and callable(chunk.dict):
                full_response += json.dumps(chunk.dict())
            elif hasattr(chunk, 'content'):
                content = getattr(chunk, 'content')
                if isinstance(content, str):
                    full_response += content
                else:
                    full_response += str(content)
            elif isinstance(chunk, str):
                full_response += chunk
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    # Patch: extract JSON from code block if present
    full_response = extract_json_from_codeblock(full_response)

    # NEW: robustly extract JSON object
    json_str = extract_json_object(full_response)
    logger.info(f"Extracted JSON string for plan: {json_str}")

    try:
        curr_plan = json.loads(repair_json_output(json_str))
        # Force all image_generation steps to use the user_input as the description
        user_input = ""
        for msg in reversed(state_dict.get("messages", [])):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_input = msg.get("content", "")
                break
            elif hasattr(msg, "role") and getattr(msg, "role") == "user":
                user_input = getattr(msg, "content", "")
                break
        logger.info(f"Extracted user_input for image generation: '{user_input}'")
        if not user_input:
            # Try to use the step's title as a fallback
            for step in curr_plan.get("steps", []):
                if step.get("step_type") == "image_generation":
                    user_input = step.get("title", "") or step.get("description", "")
                    logger.info(f"Fallback to step title/description for user_input: '{user_input}'")
                    break
        if not user_input:
            logger.warning(f"User input for image generation is still empty. State: {state_dict}")
        for step in curr_plan.get("steps", []):
            if step.get("step_type") == "image_generation":
                step["description"] = user_input
        # Ensure state['step'] is set to the first image_generation step
        for step in curr_plan.get("steps", []):
            if step.get("step_type") == "image_generation":
                state["step"] = step
                break
        # PATCH: If steps include image_generation or speech_generation, force has_enough_context = False
        step_types = [step.get("step_type") for step in curr_plan.get("steps", []) if isinstance(step, dict)]
        need_patch = False
        if any(st in ["image_generation", "speech_generation"] for st in step_types):
            curr_plan["has_enough_context"] = False
        else:
            # Check if user input has intent for image or speech generation
            keywords = ["generate an image", "create an image", "image of", "picture", "speech", "read aloud", "voice"]
            if any(k in user_input.lower() for k in keywords):
                # Auto-complete step
                if "image" in user_input.lower() or any(k in user_input for k in ["image", "picture"]):
                    curr_plan.setdefault("steps", []).append({
                        "need_search": False,
                        "title": user_input,
                        "description": user_input,
                        "step_type": "image_generation",
                        "execution_res": None
                    })
                    curr_plan["has_enough_context"] = False
                elif any(k in user_input.lower() for k in ["speech", "read aloud", "voice"]):
                    curr_plan.setdefault("steps", []).append({
                        "need_search": False,
                        "title": user_input,
                        "description": f"Convert the text to speech based on the user's request: {user_input}",
                        "step_type": "speech_generation",
                        "execution_res": None
                    })
                    curr_plan["has_enough_context"] = False
        logger.info(f"PATCHED has_enough_context: {curr_plan['has_enough_context']}, steps: {curr_plan.get('steps')}")
    except json.JSONDecodeError:
        logger.warning(f"Planner response is not a valid JSON: {full_response}")
        # 返回原始内容，便于前端和调试
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "planner_error": "Planner response is not a valid JSON",
                "planner_raw_response": full_response,
            },
            goto="__end__"
        )
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        # TEMP HACK: Add the step as a dict to the messages list for propagation
        step_dict = None
        if hasattr(state["step"], "__dict__"):
            step_dict = dict(state["step"].__dict__)
        elif isinstance(state["step"], dict):
            step_dict = state["step"]
        else:
            step_dict = state["step"]
        update_dict = {
            "messages": [
                AIMessage(content=full_response, name="planner"),
                {"role": "system", "content": "__STEP__", "step": step_dict}
            ],
            "current_plan": new_plan,
            "step": state["step"],
        }
        logger.info(f"[planner_node] DEBUG: update_dict to return: {update_dict}")
        logger.info(f"[planner_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
        return Command(
            update=update_dict,
            goto="reporter",
        )
    # PATCH: If any step is image_generation or speech_generation, go directly to the corresponding generator
    plan_obj = Plan.model_validate(curr_plan)
    from src.prompts.planner_model import Step
    plan_obj.steps = [Step.model_validate(s) if not isinstance(s, Step) else s for s in plan_obj.steps]
    if any(getattr(s, "step_type", None) in ["image_generation", "speech_generation"] for s in plan_obj.steps):
        # Set the first image_generation or speech_generation step as current step
        for s in plan_obj.steps:
            if getattr(s, "step_type", None) in ["image_generation", "speech_generation"]:
                state["step"] = s
                break
        # TEMP HACK: Add the step as a dict to the messages list for propagation
        step_dict = None
        if hasattr(state["step"], "__dict__"):
            step_dict = dict(state["step"].__dict__)
        elif isinstance(state["step"], dict):
            step_dict = state["step"]
        else:
            step_dict = state["step"]
        update_dict = {
            "messages": [
                AIMessage(content=full_response, name="planner"),
                {"role": "system", "content": "__STEP__", "step": step_dict}
            ],
            "current_plan": plan_obj,
            "step": state["step"],
        }
        logger.info(f"[planner_node] DEBUG: update_dict to return: {update_dict}")
        logger.info(f"[planner_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
        # Route to the correct generator node
        if getattr(state["step"], "step_type", None) == "image_generation":
            return Command(update=update_dict, goto="image_generator")
        elif getattr(state["step"], "step_type", None) == "speech_generation":
            return Command(update=update_dict, goto="speech_generator")
    # PATCH: human_feedback 分支 current_plan 用 Plan.model_validate(curr_plan)
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": plan_obj,
            "step": state["step"],
        },
        goto="human_feedback",
    )
    logger.info(f"[planner_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


def human_feedback_node(state) -> Command[Literal["planner", "image_generator", "reporter", "__end__"]]:
    logger.info(f"[human_feedback_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("human_feedback_node called.")
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    feedback = None
    if not auto_accepted_plan:
        feedback = interrupt({
            "content": "Please Review the Plan.",
            "options": [
                {"text": "Edit plan", "value": "edit_plan"},
                {"text": "Start research", "value": "accepted"},
            ]
        })
        if feedback and str(feedback).upper().startswith("[EDIT_PLAN]"):
            return Command(goto="planner")
        # 标记 plan 已被接受，后续自动推进
        state["auto_accepted_plan"] = True
        # 强制写入 step
        current_plan = state.get("current_plan")
        if current_plan and hasattr(current_plan, "steps") and len(current_plan.steps) > 0:
            state["step"] = current_plan.steps[0]
        return Command(goto="image_generator")
    # 已经被接受，自动推进
    current_plan = state.get("current_plan")
    if current_plan and hasattr(current_plan, "steps") and len(current_plan.steps) > 0:
        state["step"] = current_plan.steps[0]
    return Command(goto="image_generator")
    logger.info(f"[human_feedback_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "background_investigator", "__end__"]]:
    """Coordinator node that communicate with customers."""
    logger.info(f"[coordinator_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Coordinator talking.")
    configurable = Configuration.from_runnable_config(config)
    state_dict = state_to_dict(state)
    messages = apply_prompt_template("coordinator", state_dict)
    response = (
        get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    logger.debug(f"Current state messages: {state['messages']}")

    goto = "__end__"
    locale = state.get("locale", "en-US")  # Default locale if not specified
    research_topic = state.get("research_topic", "")

    tool_calls = None
    if isinstance(response, dict):
        tool_calls = response.get("tool_calls")
    else:
        tool_calls = getattr(response, "tool_calls", None)

    if tool_calls and len(tool_calls) > 0:
        goto = "planner"
        if state.get("enable_background_investigation"):
            # if the search_before_planning is True, add the web search tool to the planner agent
            goto = "background_investigator"
        try:
            for tool_call in tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    continue
                if tool_call.get("args", {}).get("locale") and tool_call.get(
                    "args", {}
                ).get("research_topic"):
                    locale = tool_call.get("args", {}).get("locale")
                    research_topic = tool_call.get("args", {}).get("research_topic")
                    break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )
        logger.debug(f"Coordinator response: {response}")

    result = Command(
        update={
            "messages": [AIMessage(content=f"Handoff to planner: {research_topic}")],
            "research_topic": research_topic,
            "locale": locale,
        },
        goto=goto,
    )
    logger.info(f"[coordinator_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    return result


def reporter_node(state: State, config: RunnableConfig):
    """Reporter node to generate the final report."""
    logger.info(f"[reporter_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Reporter generating final report.")
    configurable = Configuration.from_runnable_config(config)
    state_dict = state_to_dict(state)

    # --- NEW: Extract image markdown from previous messages ---
    image_markdown = ""
    for msg in reversed(state_dict.get("messages", [])):
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "image_generator":
            image_markdown = msg.content
            break
    state_dict["generated_image_markdown"] = image_markdown
    # ---------------------------------------------------------

    messages = apply_prompt_template("reporter", state_dict, configurable)

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if configurable.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    full_response = ""
    response = llm.stream(messages)
    for chunk in response:
        # Try dict first
        if isinstance(chunk, dict):
            content = chunk.get("content")
        # Try BaseModel (Pydantic, LangChain, etc.)
        elif hasattr(chunk, "content"):
            content = getattr(chunk, "content")
        else:
            content = None
        if isinstance(content, str):
            full_response += content

    plan = state_dict.get("current_plan")
    try:
        if isinstance(plan, str):
            plan_obj = json.loads(plan)
        else:
            plan_obj = plan
        if plan_obj and "steps" in plan_obj:
            for step in plan_obj["steps"]:
                images = step.get("output", {}).get("images", [])
                if images and images[0].get("url"):
                    url = images[0]["url"]
                    image_markdown = f"![Generated Image]({url})\n"
                    break
    except Exception as e:
        logger.warning(f"Reporter image markdown parse error: {e}")

    if image_markdown:
        full_response = image_markdown + full_response

    logger.info(f"Reporter response: {full_response}")
    return Command(update={"final_report": full_response})
    logger.info(f"[reporter_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


def research_team_node(state: State):
    """This node is a team of agents that do research."""
    logger.info(f"[research_team_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Research team is researching.")
    next_node = continue_to_running_research_team(state)
    logger.info(f"[DEBUG] research_team_node next_node: {next_node}, state.current_plan: {state.get('current_plan')}")
    return Command(
        update={"messages": [AIMessage(content="Start Research", name="research_team")]},
        goto=next_node
    )
    logger.info(f"[research_team_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


async def _execute_agent_step(
    state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Execute a single agent step."""
    plan = state.get("current_plan")
    logger.debug(f"Plan: {plan}")
    if not plan or not hasattr(plan, "steps") or not isinstance(getattr(plan, "steps", None), list):
        return Command(goto="research_team")
    # find the next step to execute
    steps = []
    if hasattr(plan, 'steps'):
        steps = getattr(plan, 'steps', [])
    elif isinstance(plan, dict):
        steps = plan.get('steps', [])
    elif isinstance(plan, str):
        try:
            plan_obj = json.loads(plan)
            steps = plan_obj.get('steps', [])
        except Exception:
            steps = []
    for step in steps:
        if not (hasattr(step, 'execution_res') and getattr(step, 'execution_res') is not None):
            break
    else:
        logger.info("All steps are executed.")
        return Command(goto="research_team")
    logger.info(f"Current step to execute: {step}")
    # prepare messages for the agent
    messages = [
        *state["messages"],
        {
            "role": "user",
            "content": apply_prompt_template(
                "researcher",
                {
                    "step": step,
                    "plan": plan,
                    "messages": state["messages"],
                    "resources": state.get("resources", []),
                },
            ),
        },
    ]
    # execute the agent
    full_response = ""
    response = agent.astream(messages)
    logger.info(f"Response from {agent_name}: ")
    # TODO: Stream the response
    async for chunk in response:
        if hasattr(chunk, 'content') and isinstance(chunk.content, str):
            if isinstance(full_response, str):
                full_response += chunk.content
            elif isinstance(full_response, list):
                full_response.append(chunk.content)
            else:
                full_response = chunk.content
    # update the step with the execution result
    step.execution_res = full_response
    logger.info(f"Agent {agent_name} execution result: {full_response}")

    result = Command(
        update={
            "current_plan": plan,
            "messages": [
                AIMessage(
                    content=full_response,
                    name=agent_name,
                )
            ],
        },
        goto="research_team",
    )
    # Defensive: If result is not a dict/Command, wrap it
    if isinstance(result, list):
        return Command(update={"messages": result})
    return result


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Setup and execute an agent step with dynamic tools."""
    # Create the agent with default tools
    logger.info(f"Creating {agent_type} agent.")
    agent = create_agent(
        agent_name=agent_type,
        agent_type=agent_type,
        tools=default_tools,
        prompt_template=agent_type
    )
    # Load dynamic tools from MCP settings
    configurable = Configuration.from_runnable_config(config)
    # TODO: Implement load_mcp_tools function
    # dynamic_tools = await load_mcp_tools(configurable.mcp_settings, agent_type)
    # if dynamic_tools:
    #     agent = agent.with_tools(dynamic_tools)
    #     logger.info(f"Loaded dynamic tools for {agent_type}: {dynamic_tools}")

    # Execute the agent step
    result = await _execute_agent_step(state, agent, agent_type)
    if isinstance(result, list):
        return Command(update={"messages": result})
    return result


async def researcher_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info(f"[researcher_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Researcher node is researching.")
    configurable = Configuration.from_runnable_config(config)
    tools = [get_web_search_tool(configurable.max_search_results), crawl_tool]
    retriever_tool = get_retriever_tool(state.get("resources", []))
    if retriever_tool:
        tools.insert(0, retriever_tool)
    logger.info(f"Researcher tools: {tools}")
    result = await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        tools,
    )
    if isinstance(result, list):
        return Command(update={"messages": result})
    return result
    logger.info(f"[researcher_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code generation and execution"""
    logger.info(f"[coder_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    logger.info("Coder node is working.")
    tools = [python_repl_tool]
    return await _setup_and_execute_agent_step(state, config, "coder", tools)
    logger.info(f"[coder_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not hasattr(current_plan, "steps") or not isinstance(getattr(current_plan, "steps", None), list):
        return "planner"
    if all(step.execution_res for step in current_plan.steps):
        return "planner"
    for step in current_plan.steps:
        if not step.execution_res:
            break
    if step.step_type and step.step_type == StepType.RESEARCH:
        return "researcher"
    if step.step_type and step.step_type == StepType.PROCESSING:
        return "coder"
    if step.step_type and step.step_type == StepType.IMAGE_GENERATION:
        return "image_generator"
    if step.step_type and step.step_type == StepType.SPEECH_GENERATION:
        return "speech_generator"
    return "planner"


async def image_generator_node(
    state: State, config: RunnableConfig
) -> dict:
    logger.info("image_generator_node called!")
    file_path = generate_image_tool(prompt)
    logger.info(f"[image_generator_node] file_path generated: {file_path}")

    # Try to parse file_path as JSON, fallback to using it as a string
    try:
        file_path_obj = json.loads(file_path)
        # If the tool already returns {"type": "image", "url": ...}
        if isinstance(file_path_obj, dict) and "url" in file_path_obj:
            url = file_path_obj["url"]
        else:
            url = file_path
    except Exception:
        url = file_path

    if url and url != "No image generated.":
        # If url does not start with /static/, add it
        if not url.startswith("/static/"):
            url = f"/static/{url}"
        content = json.dumps({"type": "image", "url": url})
    else:
        content = json.dumps({"error": "Image generation failed."})

    image_message = AIMessage(content=content, name="image_generator")
    logger.info(f"[image_generator_node] image_message content: {content}")
    if "messages" in state and isinstance(state["messages"], list):
        state["messages"].append(image_message)
    else:
        state["messages"] = [image_message]
    logger.info(f"[image_generator_node] Returning update: {state['messages'][-1]}")
    return {
        "update": {
            "messages": state["messages"]
        },
        "messages": [image_message]
    }


async def speech_generator_node(
    state: State, config: RunnableConfig
) -> dict:
    # ... existing code ...
    response = speech_agent.invoke(messages)
    # 假设 response.content 是文件路径
    if response.content and response.content != "No audio generated.":
        url = f"/static/{response.content}"
        content = json.dumps({"type": "audio", "url": url})
    else:
        content = json.dumps({"error": "Audio generation failed."})
    return {
        "messages": [AIMessage(content=content, name="speech_generator")]
    }
