from src.agents.factory import create_agent
from src.tools.google_genai_image import generate_image_tool

def create_image_generator_agent():
    return create_agent(
        agent_name="image_generator",
        agent_type="image_generator",
        tools=[generate_image_tool],
        prompt_template="image_generator"
    ) 