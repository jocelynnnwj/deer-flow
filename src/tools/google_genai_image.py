import os
import mimetypes
import json
import uuid
from pathlib import Path
from langchain_core.tools import tool

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    raise ImportError("google-generativeai package is required. Please install it with 'pip install google-generativeai'.")

@tool
def generate_image_tool(prompt: str, aspect_ratio: str = "1:1", size: str = "1024x1024", style: str = "", quality: str = "standard") -> str:
    """
    Generate an image using Google Gemini model and return a JSON string with the image URL and parameters used.
    """
    # Load API key from conf.yaml or env
    try:
        from src.config import load_yaml_config
        config_path = str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
        config = load_yaml_config(config_path)
        api_key = config.get("GEMINI_MODEL", {}).get("api_key") or os.environ.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return json.dumps({"error": "GEMINI_API_KEY not found in conf.yaml or environment variables"})

    # Set the API key in the environment for the SDK
    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash-preview-image-generation")  # type: ignore
        # The public SDK may not support explicit response modality selection; try with just the prompt
        response = model.generate_content(prompt)
        image_saved = False
        image_url = None
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                file_name = f"genai_image_{uuid.uuid4().hex}"
                inline_data = part.inline_data
                data_buffer = inline_data.data
                file_extension = mimetypes.guess_extension(inline_data.mime_type)
                if file_extension is None:
                    file_extension = ".png"
                public_dir = os.path.join("web", "public")
                os.makedirs(public_dir, exist_ok=True)
                file_path = os.path.join(public_dir, file_name + file_extension)
                with open(file_path, "wb") as f:
                    f.write(data_buffer)
                image_url = f"/public/{file_name}{file_extension}"
                image_saved = True
                break
        if image_saved and image_url:
            return json.dumps({
                "type": "image",
                "url": image_url,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "size": size,
                "style": style,
                "quality": quality
            })
        return json.dumps({"error": "No image generated."})
    except Exception as e:
        return json.dumps({"error": f"Image generation failed: {str(e)}"}) 