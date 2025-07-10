#!/usr/bin/env python3
"""
Test script to verify API key configuration
"""

import os

def test_gemini_api_key():
    """Test if GEMINI_API_KEY is set correctly"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        print("✅ GEMINI_API_KEY found in environment variables")
        print(f"   Key: {api_key[:10]}...")
        return True
    else:
        print("❌ No API key found in environment variable GEMINI_API_KEY")
        print("\nTo fix this:")
        print("1. Get your API key from Google AI Studio")
        print("2. Go to https://aistudio.google.com/")
        print("3. Create a new API key")
        print("4. Copy the API key")
        print("5. Set it as environment variable: export GEMINI_API_KEY='your_key_here'")
        return False

if __name__ == "__main__":
    test_gemini_api_key() 