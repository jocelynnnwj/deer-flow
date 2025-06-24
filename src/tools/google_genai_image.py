import os
import mimetypes
from google import genai
from google.genai import types
from langchain_core.tools import tool
import json
import uuid

@tool
def generate_image_tool(prompt: str) -> str:
    """
    Generate an image using Google Gemini image model and return a JSON string with the image URL.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.0-flash-preview-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
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
            file_name = f"genai_image_{uuid.uuid4().hex}"
            file_index += 1
            inline_data = part.inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            # Save to web/public/ directory
            public_dir = os.path.join("web", "public")
            os.makedirs(public_dir, exist_ok=True)
            file_path = os.path.join(public_dir, f"{file_name}{file_extension}")
            with open(file_path, "wb") as f:
                f.write(data_buffer)
            # Return the public URL
            url = f"/{file_name}{file_extension}"
            return json.dumps({"type": "image", "url": url})
    return json.dumps({"error": "No image generated."}) 