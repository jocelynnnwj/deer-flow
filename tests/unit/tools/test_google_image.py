# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
from unittest.mock import patch, MagicMock
import os
import json
from src.tools.google_genai_image import generate_image_tool

class TestGoogleImageTool:
    """Test cases for Google Image Generation Tool"""
    
    def test_image_generation_success(self):
        """Test successful image generation"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Mock the requests response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "images": [{
                    "image": {
                        "data": "base64_encoded_image_data"
                    }
                }]
            }
            
            with patch('src.tools.google_genai_image.requests.post') as mock_post:
                mock_post.return_value = mock_response
                
                # Mock file operations
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    result = generate_image_tool.invoke("a cute cat")
                    result_data = json.loads(result)
                    
                    assert result_data["type"] == "image"
                    assert "genai_image_" in result_data["url"]
                    assert result_data["url"].endswith(".jpg")
    
    def test_image_generation_no_api_key(self):
        """Test image generation with no API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = generate_image_tool.invoke("a cute cat")
            result_data = json.loads(result)
            
            assert "error" in result_data
            assert "GEMINI_API_KEY not found" in result_data["error"]
    
    def test_image_generation_api_error(self):
        """Test image generation with API error"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Mock the requests response with error
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = "Model not found"
            
            with patch('src.tools.google_genai_image.requests.post') as mock_post:
                mock_post.return_value = mock_response
                
                result = generate_image_tool.invoke("a cute cat")
                result_data = json.loads(result)
                
                assert "error" in result_data
                assert "API request failed: 404" in result_data["error"]
    
    def test_image_generation_no_images(self):
        """Test image generation when no images are returned"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Mock the requests response with no images
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"images": []}
            
            with patch('src.tools.google_genai_image.requests.post') as mock_post:
                mock_post.return_value = mock_response
                
                result = generate_image_tool.invoke("a cute cat")
                result_data = json.loads(result)
                
                assert "error" in result_data
                assert "No images generated" in result_data["error"]
    
    def test_image_generation_exception(self):
        """Test image generation with exception"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            with patch('src.tools.google_genai_image.requests.post') as mock_post:
                mock_post.side_effect = Exception("Network Error")
                
                result = generate_image_tool.invoke("a cute cat")
                result_data = json.loads(result)
                
                assert "error" in result_data
                assert "Image generation failed" in result_data["error"] 