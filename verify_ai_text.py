import requests
import random
import string

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_ai_user_" + "".join(random.choices(string.ascii_lowercase, k=5))
PASSWORD = "password123"

def verify_ai_text():
    # 1. Register/Login
    print(f"Authenticating as {USERNAME}...")
    auth_url = f"{BASE_URL}/auth/login"
    reg_url = f"{BASE_URL}/auth/register"
    
    requests.post(reg_url, json={"username": USERNAME, "password": PASSWORD, "email": f"{USERNAME}@example.com"})
    
    response = requests.post(auth_url, json={"username": USERNAME, "password": PASSWORD})
    if response.status_code != 200:
        print(f"Authentication failed: {response.text}")
        return

    token = response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    scan_url = f"{BASE_URL}/scan/ai-text"
    
    # 2. Test Human Text
    human_text = "I went to the store yesterday to buy some apples, but they were all out, so I got oranges instead."
    print(f"\nScanning Human Text: '{human_text}'")
    resp_human = requests.post(scan_url, headers=headers, json={"text": human_text})
    print(f"Response: {resp_human.json()}")
    
    # 3. Test AI Text (Standard GPT style)
    ai_text = "As an AI language model, I do not have personal experiences or feelings. However, I can explain the concept to you."
    print(f"\nScanning AI Text: '{ai_text}'")
    resp_ai = requests.post(scan_url, headers=headers, json={"text": ai_text})
    print(f"Response: {resp_ai.json()}")

if __name__ == "__main__":
    verify_ai_text()
