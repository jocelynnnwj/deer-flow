# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
import base64
from src.tools.google_genai_image import generate_image_tool
import json

from src.agents import create_agent
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template

from .types import State

logger = logging.getLogger(__name__)


async def image_generator_node(
    state: State, config: RunnableConfig
) -> dict:
    """Image generation agent node that creates images from prompts using Google GenAI API."""
    logger.info("Image generator agent is running")
    logger.info("[DEBUG] image_generator_node called. state keys: %s", list(state.keys()))
    logger.info(f"[image_generator_node] START: full state: {state}")
    logger.info(f"[image_generator_node] START: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    configurable = Configuration.from_runnable_config(config)
    # TEMP HACK: If state['step'] is None, try to extract it from messages (handle both dict and SystemMessage)
    if state.get('step') is None and 'messages' in state:
        for msg in state['messages']:
            # Handle dict
            if isinstance(msg, dict) and msg.get('role') == 'system' and msg.get('content') == '__STEP__' and 'step' in msg:
                state['step'] = msg['step']
                logger.info(f"[image_generator_node] HACK: Recovered step from dict message: {state['step']}")
                break
            # Handle SystemMessage or similar object
            if hasattr(msg, 'content') and getattr(msg, 'content', None) == '__STEP__' and hasattr(msg, 'additional_kwargs') and 'step' in getattr(msg, 'additional_kwargs', {}):
                state['step'] = msg.additional_kwargs['step']
                logger.info(f"[image_generator_node] HACK: Recovered step from SystemMessage: {state['step']}")
                break
    # Prepare prompt
    step = getattr(state, 'step', None)
    if step and hasattr(step, 'description'):
        prompt = getattr(step, 'description', None)
    elif isinstance(state, dict) and 'step' in state and 'description' in state['step']:
        prompt = state['step']['description']
    else:
        prompt = "NO_PROMPT_PROVIDED"
    prompt = prompt or ""
    logger.info(f"image_generator_node: state['step'] = {state.get('step')}")
    logger.info(f"image_generator_node: prompt before sanitization = '{prompt}'")
    # Sanitize prompt for image model
    lowered = prompt.lower()
    if lowered.startswith("generate an image of "):
        prompt = prompt[len("generate an image of "):]
    elif lowered.startswith("create an image of "):
        prompt = prompt[len("create an image of "):]
    prompt = prompt.strip()
    logger.info(f"image_generator_node: prompt after sanitization = '{prompt}'")
    logger.info(f"[image_generator_node] prompt used for image generation: {prompt}")
    # Generate image using Google GenAI
    file_path_json = generate_image_tool(prompt)
    logger.info(f"[image_generator_node] file_path_json: {file_path_json}")

    try:
        file_path_obj = json.loads(file_path_json)
    except Exception:
        file_path_obj = {"error": "Invalid image tool output"}

    logger.info(f"[image_generator_node] file_path_obj: {file_path_obj}")

    if file_path_obj.get("type") == "image" and "url" in file_path_obj:
        content = json.dumps(file_path_obj)
    else:
        content = json.dumps({"error": "Image generation failed."})

    image_message = AIMessage(content=content, name="image_generator")
    logger.info(f"[image_generator_node] image_message content: {content}")
    # Append to chat history
    if "messages" in state and isinstance(state["messages"], list):
        state["messages"].append(image_message)
    else:
        state["messages"] = [image_message]
    logger.info(f"[image_generator_node] Returning update: {state['messages'][-1]}")
    result = {
        "update": {
            "messages": state["messages"]
        },
        "messages": [image_message]
    }
    logger.info(f"[image_generator_node] END: state keys: {list(state.keys())}, state['step']: {state.get('step', None)}")
    return result 