# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agents import create_agent
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from src.prompts.template import apply_prompt_template
from src.tools.google_genai_tts import generate_speech_tool

from .types import State

logger = logging.getLogger(__name__)


async def speech_generator_node(
    state: State, config: RunnableConfig
) -> dict:
    """Speech generation agent node that creates speech from text."""
    logger.info("Speech generator agent is running")

    # Always extract the last HumanMessage content as the text to speak
    text = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content
            break

    if not text:
        text = "NO_TEXT_PROVIDED"

    logger.info(f"Text to be spoken: {text}")

    # Actually call the TTS tool
    audio_result = generate_speech_tool(text)  # This returns a JSON string with the audio URL
    logger.info(f"TTS tool result: {audio_result}")

    # Wrap the result in an AIMessage
    ai_message = AIMessage(content=f"Speech generated: {audio_result}", name="speech_generator")

    updated_messages = state["messages"] + [ai_message]
    return {"messages": updated_messages} 