from src.agents.factory import create_agent
from src.tools.google_genai_tts import generate_speech_tool

def create_speech_generator_agent():
    return create_agent(
        agent_name="speech_generator",
        agent_type="speech_generator",
        tools=[generate_speech_tool],
        prompt_template="speech_generator"
    ) 