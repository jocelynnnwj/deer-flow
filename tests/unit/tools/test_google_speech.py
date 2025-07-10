# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import json
from src.tools.google_genai_tts import generate_speech_tool


class TestGoogleSpeechTool:
    """Test cases for Google Speech Generation Tool"""
    
    def test_speech_generation_success(self):
        """Test successful speech generation"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Mock the Google Generative AI response
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            
            # Set up the mock chain
            mock_response.candidates = [mock_candidate]
            mock_candidate.content = mock_content
            mock_content.parts = [mock_part]
            mock_part.inline_data = mock_inline_data
            mock_inline_data.data = b"fake_audio_data"
            mock_inline_data.mime_type = "audio/wav"
            
            with patch('src.tools.google_genai_tts.genai.GenerativeModel') as mock_model:
                mock_model_instance = MagicMock()
                mock_model_instance.generate_content.return_value = mock_response
                mock_model.return_value = mock_model_instance
                
                # Mock file operations
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = MagicMock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    result = generate_speech_tool.invoke("Hello, world!")
                    result_data = json.loads(result)
                    
                    assert result_data["type"] == "audio"
                    assert "genai_tts_" in result_data["url"]
                    assert result_data["url"].endswith(".wav")
    
    def test_speech_generation_no_api_key(self):
        """Test speech generation with no API key"""
        with patch.dict(os.environ, {}, clear=True):
            result = generate_speech_tool.invoke("Hello, world!")
            result_data = json.loads(result)
            
            assert "error" in result_data
            assert "GEMINI_API_KEY not found" in result_data["error"]
    
    def test_speech_generation_no_audio_generated(self):
        """Test speech generation when no audio is generated"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Mock the Google Generative AI response with no audio
            mock_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            
            mock_response.candidates = [mock_candidate]
            mock_candidate.content = mock_content
            mock_content.parts = []  # No parts with audio data
            
            with patch('src.tools.google_genai_tts.genai.GenerativeModel') as mock_model:
                mock_model_instance = MagicMock()
                mock_model_instance.generate_content.return_value = mock_response
                mock_model.return_value = mock_model_instance
                
                result = generate_speech_tool.invoke("Hello, world!")
                result_data = json.loads(result)
                
                assert "error" in result_data
                assert "No audio generated" in result_data["error"]
    
    def test_speech_generation_exception(self):
        """Test speech generation with exception"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            with patch('src.tools.google_genai_tts.genai.GenerativeModel') as mock_model:
                mock_model_instance = MagicMock()
                mock_model_instance.generate_content.side_effect = Exception("API Error")
                mock_model.return_value = mock_model_instance
                
                result = generate_speech_tool.invoke("Hello, world!")
                result_data = json.loads(result)
                
                assert "error" in result_data
                assert "Speech generation failed" in result_data["error"] 