#!/usr/bin/env python3
"""
Quick test script to verify OpenAI API setup is working correctly.
Run this before running the full evaluation to avoid wasting time/money.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import sys

# Load environment variables from .env file
load_dotenv()

def test_openai_setup():
    """Test if OpenAI API is properly configured."""
    print("="*70)
    print("Testing OpenAI API Setup")
    print("="*70)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable is not set")
        print("\nPlease set it using:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\nOr add it permanently to ~/.zshrc:")
        print("  echo 'export OPENAI_API_KEY=\"your-api-key-here\"' >> ~/.zshrc")
        print("  source ~/.zshrc")
        return False
    
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connection with a simple request
    print("\nTesting API connection with a simple request...")
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello' if you can read this."}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✓ API connection successful!")
        print(f"  Test response: {result}")
        
        # Check if GPT-4 is available
        print("\nChecking GPT-4 access...")
        try:
            response_gpt4 = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Say 'Yes' if you are GPT-4."}
                ],
                max_tokens=10
            )
            print(f"✓ GPT-4 access confirmed!")
            print(f"  GPT-4 response: {response_gpt4.choices[0].message.content}")
        except Exception as e:
            print(f"⚠️  GPT-4 not accessible: {e}")
            print("\n  You can still use GPT-3.5-turbo with:")
            print("  python scripts/evaluate_all_stages.py --gpt4-model gpt-3.5-turbo")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. No API credits available")
        print("  3. Network connectivity issues")
        return False

if __name__ == "__main__":
    print("\n")
    success = test_openai_setup()
    print("\n" + "="*70)
    
    if success:
        print("✓ Setup verified! You can now run:")
        print("  python scripts/evaluate_all_stages.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    print("="*70 + "\n")
