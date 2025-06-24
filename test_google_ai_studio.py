#!/usr/bin/env python3
"""
Simple test script for Google AI Studio integration.
This script tests the basic functionality of the image and speech generation tools.
"""

import os
import json


def test_image_generation():
    """Test image generation functionality."""
    print("Testing Image Generation...")
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        print("❌ GOOGLE_AI_STUDIO_API_KEY not found in environment variables")
        print("   Set it with: export GOOGLE_AI_STUDIO_API_KEY='your_api_key'")
        return False
    
    try:
        # Test the tool directly
        result = generate_image_tool.invoke({
            "prompt": "A simple red circle on a white background",
            "aspect_ratio": "1:1",
            "size": "1024x1024"
        })
        
        result_data = json.loads(result)
        if result_data.get("success"):
            print("✅ Image generation successful!")
            print(f"   MIME Type: {result_data.get('mime_type')}")
            print(f"   Size: {result_data.get('size')}")
            return True
        else:
            print(f"❌ Image generation failed: {result_data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Image generation error: {str(e)}")
        return False


def test_speech_generation():
    """Test speech generation functionality."""
    print("\nTesting Speech Generation...")
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        print("❌ GOOGLE_AI_STUDIO_API_KEY not found in environment variables")
        print("   Set it with: export GOOGLE_AI_STUDIO_API_KEY='your_api_key'")
        return False
    
    try:
        # Test the tool directly
        result = generate_speech_tool.invoke({
            "text": "Hello, this is a test of the speech generation system.",
            "voice": "alloy",
            "speed": 1.0,
            "pitch": 0.0
        })
        
        result_data = json.loads(result)
        if result_data.get("success"):
            print("✅ Speech generation successful!")
            print(f"   MIME Type: {result_data.get('mime_type')}")
            print(f"   Voice: {result_data.get('voice')}")
            if result_data.get("note"):
                print(f"   Note: {result_data.get('note')}")
            return True
        else:
            print(f"❌ Speech generation failed: {result_data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Speech generation error: {str(e)}")
        return False


def test_agent_configuration():
    """Test that agents are properly configured."""
    print("\nTesting Agent Configuration...")
    
    try:
        from src.config.agents import AGENT_LLM_MAP
        
        if "image_generator" in AGENT_LLM_MAP and "speech_generator" in AGENT_LLM_MAP:
            print("✅ Agents properly configured in AGENT_LLM_MAP")
            print(f"   image_generator: {AGENT_LLM_MAP['image_generator']}")
            print(f"   speech_generator: {AGENT_LLM_MAP['speech_generator']}")
            return True
        else:
            print("❌ Agents not found in AGENT_LLM_MAP")
            return False
            
    except Exception as e:
        print(f"❌ Agent configuration error: {str(e)}")
        return False


def test_step_types():
    """Test that new step types are properly defined."""
    print("\nTesting Step Types...")
    
    try:
        from src.prompts.planner_model import StepType
        
        if hasattr(StepType, 'IMAGE_GENERATION') and hasattr(StepType, 'SPEECH_GENERATION'):
            print("✅ New step types properly defined")
            print(f"   IMAGE_GENERATION: {StepType.IMAGE_GENERATION}")
            print(f"   SPEECH_GENERATION: {StepType.SPEECH_GENERATION}")
            return True
        else:
            print("❌ New step types not found")
            return False
            
    except Exception as e:
        print(f"❌ Step types error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Google AI Studio Integration")
    print("=" * 50)
    
    tests = [
        test_agent_configuration,
        test_step_types,
        test_image_generation,
        test_speech_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Google AI Studio integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and API key.")
    
    return passed == total


if __name__ == "__main__":
    main() 