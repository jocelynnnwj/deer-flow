from src.tools.google_genai_image import generate_image_tool
from src.tools.google_genai_tts import generate_speech_tool

def test_generate_image_tool():
    path = generate_image_tool("A cat riding a bicycle")
    assert path and (path.endswith(".png") or path.endswith(".jpg")), f"Unexpected image path: {path}"

def test_generate_speech_tool():
    path = generate_speech_tool("Welcome!")
    assert path and path.endswith(".wav"), f"Unexpected audio path: {path}" 