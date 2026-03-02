import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
print(f"API Key found: {bool(api_key)}")
print(f"Key starts with: {api_key[:10]}..." if api_key else "NO KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test 1: List models
print("\n1. Testing models endpoint...")
r = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
print(f"Status: {r.status_code}")
if r.status_code == 200:
    models = [m['id'] for m in r.json().get('data', [])]
    print(f"Available models: {models[:5]}")

# Test 2: Simple chat completion
print("\n2. Testing chat completion...")
payload = {
    "model": "meta-llama/llama-guard-4-12b",
    "messages": [
        {"role": "user", "content": "Say hello"}
    ],
    "max_tokens": 50
}

r = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers=headers,
    json=payload
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:500]}")