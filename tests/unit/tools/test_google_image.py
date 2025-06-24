# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.tools.google_image import GoogleImageGenerator, generate_image_tool


class TestGoogleImageGenerator:
    """Test cases for GoogleImageGenerator class."""

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key parameter."""
        api_key = "test_api_key"
        generator = GoogleImageGenerator(api_key=api_key)
        assert generator.api_key == api_key

    def test_init_with_environment_variable(self):
        """Test initialization with environment variable."""
        api_key = "test_env_api_key"
        with patch.dict(os.environ, {"GOOGLE_AI_STUDIO_API_KEY": api_key}):
            generator = GoogleImageGenerator()
            assert generator.api_key == api_key

    def test_init_no_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google AI Studio API key is required"):
                GoogleImageGenerator()

    @patch('google.generativeai.GenerativeModel')
    def test_generate_image_success(self, mock_model_class):
        """Test successful image generation."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].inline_data = Mock()
        mock_response.candidates[0].content.parts[0].inline_data.mime_type = "image/png"
        mock_response.candidates[0].content.parts[0].inline_data.data = b"fake_image_data"
        mock_response.usage_metadata = None

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleImageGenerator(api_key="test_key")
        result = generator.generate_image("test prompt")

        # Assertions
        assert result["success"] is True
        assert result["image_data"] is not None
        assert result["mime_type"] == "image/png"
        assert result["prompt"] == "test prompt"
        mock_model_instance.generate_content.assert_called_once()

    @patch('google.generativeai.GenerativeModel')
    def test_generate_image_no_candidates(self, mock_model_class):
        """Test image generation with no candidates in response."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = []

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleImageGenerator(api_key="test_key")
        result = generator.generate_image("test prompt")

        # Assertions
        assert result["success"] is False
        assert "No image generated" in result["error"]

    @patch('google.generativeai.GenerativeModel')
    def test_generate_image_no_image_data(self, mock_model_class):
        """Test image generation with no image data in response."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = []

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleImageGenerator(api_key="test_key")
        result = generator.generate_image("test prompt")

        # Assertions
        assert result["success"] is False
        assert "No image data" in result["error"]

    @patch('google.generativeai.GenerativeModel')
    def test_generate_image_exception(self, mock_model_class):
        """Test image generation with exception."""
        # Mock the model instance to raise exception
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleImageGenerator(api_key="test_key")
        result = generator.generate_image("test prompt")

        # Assertions
        assert result["success"] is False
        assert "Image generation failed" in result["error"]


class TestGenerateImageTool:
    """Test cases for generate_image_tool function."""

    @patch('src.tools.google_image.get_image_generator')
    def test_generate_image_tool_success(self, mock_get_generator):
        """Test successful tool execution."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.generate_image.return_value = {
            "success": True,
            "image_data": "base64_encoded_data",
            "mime_type": "image/png",
            "prompt": "test prompt"
        }
        mock_get_generator.return_value = mock_generator

        # Test
        result = generate_image_tool.invoke({
            "prompt": "test prompt",
            "aspect_ratio": "1:1",
            "size": "1024x1024"
        })

        # Assertions
        assert "success" in result
        assert "image_data" in result
        mock_generator.generate_image.assert_called_once()

    @patch('src.tools.google_image.get_image_generator')
    def test_generate_image_tool_exception(self, mock_get_generator):
        """Test tool execution with exception."""
        # Mock the generator to raise exception
        mock_generator = Mock()
        mock_generator.generate_image.side_effect = Exception("Tool Error")
        mock_get_generator.return_value = mock_generator

        # Test
        result = generate_image_tool.invoke({
            "prompt": "test prompt"
        })

        # Assertions
        assert "success" in result
        assert result["success"] is False
        assert "Tool execution failed" in result["error"] 