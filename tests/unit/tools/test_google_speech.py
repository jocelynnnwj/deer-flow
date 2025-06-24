# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from src.tools.google_speech import GoogleSpeechGenerator, generate_speech_tool


class TestGoogleSpeechGenerator:
    """Test cases for GoogleSpeechGenerator class."""

    def test_init_with_api_key_parameter(self):
        """Test initialization with API key parameter."""
        api_key = "test_api_key"
        generator = GoogleSpeechGenerator(api_key=api_key)
        assert generator.api_key == api_key

    def test_init_with_environment_variable(self):
        """Test initialization with environment variable."""
        api_key = "test_env_api_key"
        with patch.dict(os.environ, {"GOOGLE_AI_STUDIO_API_KEY": api_key}):
            generator = GoogleSpeechGenerator()
            assert generator.api_key == api_key

    def test_init_no_api_key(self):
        """Test initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Google AI Studio API key is required"):
                GoogleSpeechGenerator()

    @patch('google.generativeai.GenerativeModel')
    def test_generate_speech_success(self, mock_model_class):
        """Test successful speech generation."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].inline_data = Mock()
        mock_response.candidates[0].content.parts[0].inline_data.mime_type = "audio/mp3"
        mock_response.candidates[0].content.parts[0].inline_data.data = b"fake_audio_data"
        mock_response.usage_metadata = None

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleSpeechGenerator(api_key="test_key")
        result = generator.generate_speech("test text")

        # Assertions
        assert result["success"] is True
        assert result["audio_data"] is not None
        assert result["mime_type"] == "audio/mp3"
        assert result["text"] == "test text"
        mock_model_instance.generate_content.assert_called_once()

    @patch('google.generativeai.GenerativeModel')
    def test_generate_speech_no_candidates(self, mock_model_class):
        """Test speech generation with no candidates in response."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = []

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleSpeechGenerator(api_key="test_key")
        result = generator.generate_speech("test text")

        # Assertions
        assert result["success"] is False
        assert "No audio generated" in result["error"]

    @patch('google.generativeai.GenerativeModel')
    def test_generate_speech_no_audio_data(self, mock_model_class):
        """Test speech generation with no audio data in response."""
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
        generator = GoogleSpeechGenerator(api_key="test_key")
        result = generator.generate_speech("test text")

        # Assertions
        assert result["success"] is False
        assert "No audio data" in result["error"]

    @patch('google.generativeai.GenerativeModel')
    def test_generate_speech_exception(self, mock_model_class):
        """Test speech generation with exception."""
        # Mock the model instance to raise exception
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleSpeechGenerator(api_key="test_key")
        result = generator.generate_speech("test text")

        # Assertions
        assert result["success"] is False
        assert "Speech generation failed" in result["error"]

    @patch('google.generativeai.GenerativeModel')
    def test_generate_speech_with_custom_parameters(self, mock_model_class):
        """Test speech generation with custom parameters."""
        # Mock the response
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].inline_data = Mock()
        mock_response.candidates[0].content.parts[0].inline_data.mime_type = "audio/mp3"
        mock_response.candidates[0].content.parts[0].inline_data.data = b"fake_audio_data"
        mock_response.usage_metadata = None

        # Mock the model instance
        mock_model_instance = Mock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance

        # Test
        generator = GoogleSpeechGenerator(api_key="test_key")
        result = generator.generate_speech(
            text="test text",
            voice="nova",
            speed=1.5,
            pitch=5.0
        )

        # Assertions
        assert result["success"] is True
        assert result["voice"] == "nova"
        assert result["speed"] == 1.5
        assert result["pitch"] == 5.0
        mock_model_instance.generate_content.assert_called_once()


class TestGenerateSpeechTool:
    """Test cases for generate_speech_tool function."""

    @patch('src.tools.google_speech.get_speech_generator')
    def test_generate_speech_tool_success(self, mock_get_generator):
        """Test successful tool execution."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.generate_speech.return_value = {
            "success": True,
            "audio_data": "base64_encoded_data",
            "mime_type": "audio/mp3",
            "text": "test text"
        }
        mock_get_generator.return_value = mock_generator

        # Test
        result = generate_speech_tool.invoke({
            "text": "test text",
            "voice": "alloy",
            "speed": 1.0,
            "pitch": 0.0
        })

        # Assertions
        assert "success" in result
        assert "audio_data" in result
        mock_generator.generate_speech.assert_called_once()

    @patch('src.tools.google_speech.get_speech_generator')
    def test_generate_speech_tool_exception(self, mock_get_generator):
        """Test tool execution with exception."""
        # Mock the generator to raise exception
        mock_generator = Mock()
        mock_generator.generate_speech.side_effect = Exception("Tool Error")
        mock_get_generator.return_value = mock_generator

        # Test
        result = generate_speech_tool.invoke({
            "text": "test text"
        })

        # Assertions
        assert "success" in result
        assert result["success"] is False
        assert "Tool execution failed" in result["error"] 