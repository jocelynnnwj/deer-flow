#!/usr/bin/env python3
"""
Simple script to test Google AI Studio API key and list available models.
"""

import os
import google.generativeai as genai

def test_api_key():
    """Test the API key and list available models."""
    
    # Get API key from environment or prompt user
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    
    if not api_key:
        print("❌ No API key found in environment variable GOOGLE_AI_STUDIO_API_KEY")
        print("\nTo get an API key:")
        print("1. Go to https://aistudio.google.com/")
        print("2. Sign in with your Google account")
        print("3. Navigate to the API section")
        print("4. Create a new API key")
        print("5. Set it as environment variable: export GOOGLE_AI_STUDIO_API_KEY='your_key_here'")
        return False
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        print(f"✅ API key configured successfully")
        
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
    test_api_key() 