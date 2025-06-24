import os
import mimetypes
import struct
from google import genai
from google.genai import types
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
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-pro-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
    )
    file_index = 0
    file_path = None
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        part = chunk.candidates[0].content.parts[0]
        if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
            # Save to web/public/static/ directory
            public_dir = os.path.join("web", "public", "static")
            os.makedirs(public_dir, exist_ok=True)
            file_name = f"genai_tts_{uuid.uuid4().hex}.wav"
            file_path = os.path.join(public_dir, file_name)
            inline_data = part.inline_data
            data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            with open(file_path, "wb") as f:
                f.write(data_buffer)
            break
    if file_path:
        url = f"/static/{os.path.basename(file_path)}"
        return json.dumps({"type": "audio", "url": url})
    else:
        return json.dumps({"error": "No audio generated."}) 