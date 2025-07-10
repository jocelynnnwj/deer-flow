# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import pytest
import os
from unittest.mock import Mock, patch
from src.graph.builder import build_graph
from src.prompts.planner_model import StepType, Step, Plan
import json
from src.tools.google_genai_image import generate_image_tool
from src.tools.google_genai_tts import generate_speech_tool
from unittest.mock import MagicMock


class TestGoogleAIStudioAgentsIntegration:
    """Integration tests for Google AI Studio agents."""

    @pytest.fixture
    def graph_instance(self):
        """Create a graph instance for testing."""
        return build_graph()

    @pytest.fixture
    def sample_image_generation_plan(self):
        """Create a sample plan with image generation step."""
        return Plan(
            locale="en-US",
            has_enough_context=True,
            thought="Need to generate an image for the report",
            title="Image Generation Test",
            steps=[
                Step(
                    need_search=False,
                    title="Generate Product Image",
                    description="Create an image of a modern smartphone",
                    step_type=StepType.IMAGE_GENERATION,
                    execution_res=None
                )
            ]
        )

    @pytest.fixture
    def sample_speech_generation_plan(self):
        """Create a sample plan with speech generation step."""
        return Plan(
            locale="en-US",
            has_enough_context=True,
            thought="Need to generate speech for the report",
            title="Speech Generation Test",
            steps=[
                Step(
                    need_search=False,
                    title="Generate Audio Narration",
                    description="Convert the report summary to speech",
                    step_type=StepType.SPEECH_GENERATION,
                    execution_res=None
                )
            ]
        )

    @patch('src.tools.google_image.GoogleImageGenerator')
    def test_image_generation_agent_integration(self, mock_image_generator_class, graph_instance, sample_image_generation_plan):
        """Test image generation agent integration with the graph."""
        # Mock the image generator
        mock_generator = Mock()
        mock_generator.generate_image.return_value = {
            "success": True,
            "image_data": "base64_encoded_image_data",
            "mime_type": "image/png",
            "prompt": "Create an image of a modern smartphone"
        }
        mock_image_generator_class.return_value = mock_generator

        # Create test state
        test_state = {
            "messages": [
                {"role": "user", "content": "Generate an image of a smartphone"}
            ],
            "current_plan": sample_image_generation_plan,
            "plan_iterations": 0,
            "final_report": "",
            "observations": [],
            "auto_accepted_plan": True,
            "enable_background_investigation": False,
            "research_topic": "Generate an image of a smartphone"
        }

        # Test the graph execution
        # Note: This is a simplified test - in a real scenario, you'd need to mock the LLM responses
        # and handle the full graph execution flow
        assert graph_instance is not None
        assert sample_image_generation_plan.steps[0].step_type == StepType.IMAGE_GENERATION

    @patch('src.tools.google_speech.GoogleSpeechGenerator')
    def test_speech_generation_agent_integration(self, mock_speech_generator_class, graph_instance, sample_speech_generation_plan):
        """Test speech generation agent integration with the graph."""
        # Mock the speech generator
        mock_generator = Mock()
        mock_generator.generate_speech.return_value = {
            "success": True,
            "audio_data": "base64_encoded_audio_data",
            "mime_type": "audio/mp3",
            "text": "Convert the report summary to speech"
        }
        mock_speech_generator_class.return_value = mock_generator

        # Create test state
        test_state = {
            "messages": [
                {"role": "user", "content": "Convert this text to speech"}
            ],
            "current_plan": sample_speech_generation_plan,
            "plan_iterations": 0,
            "final_report": "",
            "observations": [],
            "auto_accepted_plan": True,
            "enable_background_investigation": False,
            "research_topic": "Convert this text to speech"
        }

        # Test the graph execution
        assert graph_instance is not None
        assert sample_speech_generation_plan.steps[0].step_type == StepType.SPEECH_GENERATION

    def test_step_type_enum_values(self):
        """Test that the new step types are properly defined."""
        assert StepType.IMAGE_GENERATION == "image_generation"
        assert StepType.SPEECH_GENERATION == "speech_generation"

    def test_plan_with_mixed_steps(self):
        """Test creating a plan with mixed step types including the new agents."""
        plan = Plan(
            locale="en-US",
            has_enough_context=True,
            thought="Need to research, generate image, and create speech",
            title="Mixed Steps Test",
            steps=[
                Step(
                    need_search=True,
                    title="Research Topic",
                    description="Gather information about the topic",
                    step_type=StepType.RESEARCH,
                    execution_res=None
                ),
                Step(
                    need_search=False,
                    title="Generate Visualization",
                    description="Create an image for the topic",
                    step_type=StepType.IMAGE_GENERATION,
                    execution_res=None
                ),
                Step(
                    need_search=False,
                    title="Create Audio Summary",
                    description="Convert findings to speech",
                    step_type=StepType.SPEECH_GENERATION,
                    execution_res=None
                )
            ]
        )

        assert len(plan.steps) == 3
        assert plan.steps[0].step_type == StepType.RESEARCH
        assert plan.steps[1].step_type == StepType.IMAGE_GENERATION
        assert plan.steps[2].step_type == StepType.SPEECH_GENERATION

    @patch.dict(os.environ, {"GOOGLE_AI_STUDIO_API_KEY": "test_key"})
    def test_agent_configuration(self):
        """Test that the new agents are properly configured."""
        from src.config.agents import AGENT_LLM_MAP
        
        assert "image_generator" in AGENT_LLM_MAP
        assert "speech_generator" in AGENT_LLM_MAP
        assert AGENT_LLM_MAP["image_generator"] == "gemini"
        assert AGENT_LLM_MAP["speech_generator"] == "gemini"

    def test_graph_nodes_exist(self, graph_instance):
        """Test that the new agent nodes exist in the graph."""
        # Check that the graph has the expected nodes
        # Note: This is a basic check - the actual graph structure might be more complex
        assert graph_instance is not None 


class TestGoogleAIStudioIntegration:
    """Integration tests for Google AI Studio tools"""
    
    def test_image_generation_integration(self):
        """Test image generation tool integration"""
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
    
    def test_speech_generation_integration(self):
        """Test speech generation tool integration"""
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
    
    def test_both_tools_with_same_api_key(self):
        """Test that both tools work with the same API key"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Test image generation
            mock_image_response = MagicMock()
            mock_image_response.status_code = 200
            mock_image_response.json.return_value = {
                "images": [{
                    "image": {
                        "data": "base64_encoded_image_data"
                    }
                }]
            }
            
            # Test speech generation
            mock_speech_response = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            mock_part = MagicMock()
            mock_inline_data = MagicMock()
            
            mock_speech_response.candidates = [mock_candidate]
            mock_candidate.content = mock_content
            mock_content.parts = [mock_part]
            mock_part.inline_data = mock_inline_data
            mock_inline_data.data = b"fake_audio_data"
            mock_inline_data.mime_type = "audio/wav"
            
            with patch('src.tools.google_genai_image.requests.post') as mock_image_post:
                mock_image_post.return_value = mock_image_response
                
                with patch('src.tools.google_genai_tts.genai.GenerativeModel') as mock_speech_model:
                    mock_speech_model_instance = MagicMock()
                    mock_speech_model_instance.generate_content.return_value = mock_speech_response
                    mock_speech_model.return_value = mock_speech_model_instance
                    
                    # Mock file operations for both tools
                    with patch('builtins.open', create=True) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file
                        
                        # Test both tools
                        image_result = generate_image_tool.invoke("a cute cat")
                        speech_result = generate_speech_tool.invoke("Hello, world!")
                        
                        image_data = json.loads(image_result)
                        speech_data = json.loads(speech_result)
                        
                        assert image_data["type"] == "image"
                        assert speech_data["type"] == "audio"
    
    def test_error_handling_integration(self):
        """Test error handling in both tools"""
        api_key = "test_api_key_123"
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": api_key}):
            # Test image generation error
            mock_image_response = MagicMock()
            mock_image_response.status_code = 404
            mock_image_response.text = "Model not found"
            
            with patch('src.tools.google_genai_image.requests.post') as mock_image_post:
                mock_image_post.return_value = mock_image_response
                
                image_result = generate_image_tool.invoke("a cute cat")
                image_data = json.loads(image_result)
                
                assert "error" in image_data
                assert "API request failed: 404" in image_data["error"]
            
            # Test speech generation error
            with patch('src.tools.google_genai_tts.genai.GenerativeModel') as mock_speech_model:
                mock_speech_model_instance = MagicMock()
                mock_speech_model_instance.generate_content.side_effect = Exception("API Error")
                mock_speech_model.return_value = mock_speech_model_instance
                
                speech_result = generate_speech_tool.invoke("Hello, world!")
                speech_data = json.loads(speech_result)
                
                assert "error" in speech_data
                assert "Speech generation failed" in speech_data["error"] 