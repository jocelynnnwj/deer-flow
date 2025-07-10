import os
import mimetypes
import struct
import google.generativeai as genai
from langchain_core.tools import tool
import json
import uuid

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

@tool
def generate_speech_tool(text: str) -> str:
    """
    Generate speech audio using Google Gemini TTS and return a JSON string with the audio URL.
    """
    # Configure the API
    from src.config import load_yaml_config
    from pathlib import Path
    
    # Load configuration from conf.yaml
    config_path = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    config = load_yaml_config(config_path)
    
    # Try to get API key from GEMINI_MODEL
    api_key = None
    if "GEMINI_MODEL" in config and "api_key" in config["GEMINI_MODEL"]:
        api_key = config["GEMINI_MODEL"]["api_key"]
    else:
        # Fallback to environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return json.dumps({"error": "GEMINI_API_KEY not found in conf.yaml or environment variables"})
    
    genai.configure(api_key=api_key)
    
    try:
        # Use the TTS model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Generate the speech
        response = model.generate_content(text)
        
        # Check if we got audio data
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    # Save to web/public/static/ directory
                    public_dir = os.path.join("web", "public", "static")
                    os.makedirs(public_dir, exist_ok=True)
                    file_name = f"genai_tts_{uuid.uuid4().hex}.wav"
                    file_path = os.path.join(public_dir, file_name)
                    
                    # Convert to WAV format
                    data_buffer = convert_to_wav(part.inline_data.data, part.inline_data.mime_type)
                    
                    with open(file_path, "wb") as f:
                        f.write(data_buffer)
                    
                    # Return the public URL
                    url = f"/static/{os.path.basename(file_path)}"
                    return json.dumps({"type": "audio", "url": url})
        
        return json.dumps({"error": "No audio generated."})
        
    except Exception as e:
        return json.dumps({"error": f"Speech generation failed: {str(e)}"}) 