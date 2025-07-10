#!/usr/bin/env python3
"""
Test script for Google AI Studio integration
"""

import os
import google.generativeai as genai

def test_gemini_api_key():
    """Test if GEMINI_API_KEY is set correctly"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Set it with: export GEMINI_API_KEY='your_api_key'")
        return False
    
    print("✅ GEMINI_API_KEY found in environment variables")
    return True

def test_gemini_models():
    """Test listing available Gemini models"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Set it with: export GEMINI_API_KEY='your_api_key'")
        return False
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print("✅ API key configured successfully")
        
        # List available models
        print("\n📋 Available models:")
        models = genai.list_models()
        for model in models:
            print(f"  - {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                print(f"    Generation methods: {model.supported_generation_methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Gemini API key...")
    test_gemini_api_key()
    
    print("\nTesting Gemini models...")
    test_gemini_models() 