import requests
import sys
import os
import random
import string

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_audio_user_" + "".join(random.choices(string.ascii_lowercase, k=5))
PASSWORD = "password123"

def verify_audio():
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
    
    # 2. Use Audio File
    # Create a dummy wav file if real one is not handy?
    # Better to mock or use a small generated sine wave to ensure librosa opens it.
    
    audio_path = "test_audio.wav"
    import wave
    import struct
    import math

    # Generate 1 sec of audio
    sample_rate = 44100
    duration = 1.0
    frequency = 440.0

    print("Generating dummy audio file...")
    with wave.open(audio_path, 'w') as obj:
        obj.setnchannels(1) # mono
        obj.setsampwidth(2)
        obj.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.sin(frequency * math.pi * float(i) / float(sample_rate)))
            data = struct.pack('<h', value)
            obj.writeframesraw(data)
            
    scan_url = f"{BASE_URL}/scan/audio"
    
    try:
        print(f"Uploading {audio_path} to {scan_url}...")
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            resp = requests.post(scan_url, headers=headers, files=files)
        
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Label: {data.get('label')}")
            print(f"Confidence: {data.get('confidence_score')}")
            print("✅ Audio Scan Success")
        else:
            print("❌ Audio Scan Failed")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    verify_audio()
