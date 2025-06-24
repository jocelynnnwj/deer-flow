# Google AI Studio Integration

This document describes the integration of Google AI Studio agents (Image Generation and Speech Generation) into DeerFlow.

## Overview

DeerFlow now includes two new agents powered by Google AI Studio:

1. **Image Generation Agent** - Uses Imagen-3 to generate high-quality images from text prompts
2. **Speech Generation Agent** - Uses Gemini TTS to convert text to natural-sounding speech

## Features

### Image Generation Agent
- **Model**: Imagen-3 via Google AI Studio
- **Capabilities**:
  - Generate images from detailed text descriptions
  - Support various aspect ratios (1:1, 16:9, 4:3, etc.)
  - Multiple image sizes (1024x1024, 1792x1024, etc.)
  - Different quality settings (standard, hd)
  - Optional style customization
- **Tool**: `generate_image_tool`

### Speech Generation Agent
- **Model**: Gemini TTS via Google AI Studio
- **Capabilities**:
  - Convert text to speech with natural voice synthesis
  - Support multiple voice options (alloy, echo, fable, onyx, nova, shimmer)
  - Adjustable speech speed (0.25x to 4.0x)
  - Pitch control (-20.0 to +20.0)
  - High-quality audio output
- **Tool**: `generate_speech_tool`

## Setup Instructions

### 1. API Key Configuration

Set your Google AI Studio API key in one of the following ways:

#### Option A: Environment Variable
```bash
export GOOGLE_AI_STUDIO_API_KEY="your_api_key_here"
```

#### Option B: Configuration File
Add to your `conf.yaml`:
```yaml
GOOGLE_AI_STUDIO:
  api_key: "your_api_key_here"
```

### 2. Enable Tools

Ensure the tools are enabled in your configuration:
```yaml
TOOLS:
  image_generation:
    enabled: true
  speech_generation:
    enabled: true
```

### 3. Dependencies

The required dependencies are already included in `pyproject.toml`:
- `google-generativeai>=0.8.0`
- `langchain-google-genai>=2.0.0`

## Integration Details

### Planner Integration

The new agents are integrated into the DeerFlow planner system:

- **Step Types**: Added `IMAGE_GENERATION` and `SPEECH_GENERATION` to the planner model
- **Routing**: The planner can now route tasks to the appropriate agents based on step type
- **Execution**: Agents are executed through the LangGraph workflow

### Graph Integration

The agents are integrated into the LangGraph workflow:

- **Nodes**: `image_generator_node` and `speech_generator_node`
- **Routing**: Updated `continue_to_running_research_team` function to handle new step types
- **State Management**: Agents update the state with their execution results

### Agent Configuration

Agents are configured in `src/config/agents.py`:
```python
AGENT_LLM_MAP: dict[str, LLMType] = {
    # ... existing agents ...
    "image_generator": "gemini",
    "speech_generator": "gemini",
}
```

## API Endpoints

### Image Generation
```
POST /api/image/generate
```

**Parameters**:
- `prompt` (required): Text description of the image to generate
- `aspect_ratio` (optional): Aspect ratio (default: "1:1")
- `size` (optional): Image size (default: "1024x1024")
- `style` (optional): Image style
- `quality` (optional): Quality setting (default: "standard")

**Response**: Image file with appropriate MIME type

### Speech Generation
```
POST /api/speech/generate
```

**Parameters**:
- `text` (required): Text to convert to speech
- `voice` (optional): Voice type (default: "alloy")
- `speed` (optional): Speech speed (default: 1.0)
- `pitch` (optional): Pitch adjustment (default: 0.0)

**Response**: Audio file with appropriate MIME type

## Usage Examples

### Direct Tool Usage

```python
from src.tools import generate_image_tool, generate_speech_tool

# Generate an image
image_result = generate_image_tool.invoke({
    "prompt": "A beautiful sunset over mountains",
    "aspect_ratio": "16:9",
    "size": "1792x1024"
})

# Generate speech
speech_result = generate_speech_tool.invoke({
    "text": "Welcome to our presentation!",
    "voice": "nova",
    "speed": 1.2
})
```

### Through LangGraph Workflow

The agents can be invoked through the DeerFlow workflow by including appropriate steps in the plan:

```json
{
  "step_type": "image_generation",
  "title": "Generate Product Visualization",
  "description": "Create an image of the product in use"
}
```

```json
{
  "step_type": "speech_generation", 
  "title": "Create Audio Narration",
  "description": "Convert the report summary to speech"
}
```

## Voice Options

Available voices for speech generation:

- **alloy**: Balanced, neutral voice
- **echo**: Warm, friendly voice  
- **fable**: Expressive, storytelling voice
- **onyx**: Deep, authoritative voice
- **nova**: Bright, energetic voice
- **shimmer**: Soft, gentle voice

## Error Handling

Both agents include comprehensive error handling:

- API key validation
- Network error handling
- Response validation
- Graceful degradation with informative error messages

## Testing

### Test Cases

1. **Image Generation Test**:
   ```bash
   curl -X POST "http://localhost:8000/api/image/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Generate an image of a cat"}'
   ```

2. **Speech Generation Test**:
   ```bash
   curl -X POST "http://localhost:8000/api/speech/generate" \
        -H "Content-Type: application/json" \
        -d '{"text": "Read this aloud: Welcome!"}'
   ```

### Success Checklist

- [ ] "Generate an image of a cat" → image output is returned
- [ ] "Read this aloud: Welcome!" → audio output is generated
- [ ] The planner correctly routes prompts to the new agents
- [ ] Agents are visible and callable within the LangGraph graph
- [ ] All code follows DeerFlow structure and conventions
- [ ] .env and API secrets are properly excluded via .gitignore

## Security Considerations

- API keys are stored securely and not committed to version control
- All API calls include proper error handling
- Input validation is performed on all parameters
- Rate limiting should be considered for production use

## Troubleshooting

### Common Issues

1. **API Key Not Found**: Ensure `GOOGLE_AI_STUDIO_API_KEY` is set correctly
2. **Import Errors**: Verify all dependencies are installed
3. **Model Access**: Ensure your Google AI Studio account has access to Imagen-3 and Gemini TTS
4. **Rate Limits**: Check Google AI Studio usage limits

### Debug Mode

Enable debug logging to troubleshoot issues:
```python
import logging
logging.getLogger("src.tools.google_image").setLevel(logging.DEBUG)
logging.getLogger("src.tools.google_speech").setLevel(logging.DEBUG)
```

## Future Enhancements

Potential improvements for future versions:

- Batch processing for multiple images/speech
- Advanced prompt templates
- Custom voice training
- Integration with other Google AI Studio models
- Caching for generated content
- Advanced error recovery mechanisms 