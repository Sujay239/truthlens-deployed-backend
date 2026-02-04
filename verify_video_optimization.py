import requests
import sys
import os
import time
import random
import string

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_opt_user_" + "".join(random.choices(string.ascii_lowercase, k=5))
PASSWORD = "password123"

def verify_optimization():
    # 1. Register/Login
    print(f"Authenticating as {USERNAME}...")
    auth_url = f"{BASE_URL}/auth/login"
    reg_url = f"{BASE_URL}/auth/register"
    
    # Register purely to ensure user exists
    requests.post(reg_url, json={"username": USERNAME, "password": PASSWORD, "email": f"{USERNAME}@example.com"})
    
    # Login
    response = requests.post(auth_url, json={"username": USERNAME, "password": PASSWORD})
    if response.status_code != 200:
        print(f"Authentication failed: {response.text}")
        return

    token = response.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Use Existing Real Video
    # We found this file in `app/ml`
    source_video_path = os.path.join("app", "ml", "temp_8223_6994252_Generative_Female_Artificial_Intelligence_3840x2160.mp4")
    video_path = "test_real_video.mp4"
    
    if os.path.exists(source_video_path):
        import shutil
        shutil.copy(source_video_path, video_path)
        print(f"Using real video file: {source_video_path}")
    else:
        print("Warning: Real video not found, falling back to dummy (might fail scan)")
        with open(video_path, "wb") as f:
            f.write(os.urandom(1024 * 1024)) # 1MB dummy video
        
    scan_url = f"{BASE_URL}/scan/video"
    
    try:
        # 3. First Upload (Should trigger "scan")
        print("\n--- First Upload (Fresh) ---")
        start_time = time.time()
        with open(video_path, "rb") as f:
            files = {"file": (video_path, f, "video/mp4")}
            resp1 = requests.post(scan_url, headers=headers, files=files)
        
        duration1 = time.time() - start_time
        print(f"Time taken: {duration1:.2f}s")
        print(f"Status: {resp1.status_code}")
        
        if resp1.status_code != 200:
            print(f"Error: {resp1.text}")
            return
            
        data1 = resp1.json()
        id1 = data1.get("id")
        print(f"Result ID: {id1}")

        # 4. Second Upload (Should be cached)
        print("\n--- Second Upload (Cached) ---")
        start_time = time.time()
        with open(video_path, "rb") as f:
            files = {"file": (video_path, f, "video/mp4")}
            resp2 = requests.post(scan_url, headers=headers, files=files)
            
        duration2 = time.time() - start_time
        print(f"Time taken: {duration2:.2f}s")
        print(f"Status: {resp2.status_code}")
        
        data2 = resp2.json()
        id2 = data2.get("id")
        print(f"Result ID: {id2}")
        
        # 5. Verification
        print("\n--- Verification Results ---")
        if id1 == id2:
            print("✅ PASS: Result IDs match (Returned from DB)")
        else:
            print("❌ FAIL: Result IDs do not match")
            
        # Check time improvement? 
        # Since the "scan" logic in scan.py is actually just calling a mocked predict_video which might be fast,
        # the time difference might not be huge unless predict_video is slow.
        # But logically, we verified the ID match which proves DB reuse.
        
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass

if __name__ == "__main__":
    verify_optimization()
