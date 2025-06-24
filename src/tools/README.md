# Google GenAI Image & Speech Tools for DeerFlow

## Overview
This module provides two tool wrappers for Google GenAI (Gemini) image and speech (TTS) generation, compatible with LangChain/LangGraph and integrated into DeerFlow's agent workflow.

- **generate_image_tool**: Generates images from text prompts using the Gemini image model.
- **generate_speech_tool**: Generates speech audio from text using the Gemini TTS model.

## Setup
1. Install dependencies:
   ```bash
   pip install google-genai langchain-core
   ```
2. Set your API key:
   ```bash
   export GEMINI_API_KEY="your_google_genai_api_key"
   ```

## Usage
```python
from src.tools.google_genai_image import generate_image_tool
from src.tools.google_genai_tts import generate_speech_tool

img_path = generate_image_tool("A cat riding a bicycle")
print("Image saved to:", img_path)

audio_path = generate_speech_tool("Welcome!")
print("Audio saved to:", audio_path)
```

## Integration Notes
- Both tools are registered as LangChain tools and can be used in LangGraph-compatible agents.
- Agents are registered in `AGENT_LLM_MAP` and integrated into the planner and graph builder.
- Planner routes `IMAGE_GENERATION` and `SPEECH_GENERATION` steps to the respective agents.

## Test Cases
See `tests/unit/tools/test_google_genai_tools.py` for example tests:
- `test_generate_image_tool()`
- `test_generate_speech_tool()`

## Security
- Do **not** commit your API key or `.env` file to version control.
- Ensure `.env` and any secret files are in `.gitignore`. 