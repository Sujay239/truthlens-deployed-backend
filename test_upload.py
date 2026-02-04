import requests
import sys
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_user_123" # Change if needed, or we will register a new one
PASSWORD = "password123"

def test_scan():
    # 1. Register/Login to get Token
    print(f"Authenticating as {USERNAME}...")
    auth_url = f"{BASE_URL}/auth/login"
    
    # Try login first
    response = requests.post(auth_url, json={"username": USERNAME, "password": PASSWORD})
    
    if response.status_code == 401:
        # Register if login fails
        print("User not found, registering...")
        reg_url = f"{BASE_URL}/auth/register"
        reg_resp = requests.post(reg_url, json={"username": USERNAME, "password": PASSWORD, "email": f"{USERNAME}@example.com"})
        if reg_resp.status_code != 200:
            print(f"Registration failed: {reg_resp.text}")
            return
        # Login again
        response = requests.post(auth_url, json={"username": USERNAME, "password": PASSWORD})

    if response.status_code != 200:
        print(f"Authentication failed: {response.text}")
        return

    token = response.json().get("access_token")
    print(f"Got Token: {token[:10]}...")

    # 2. Upload File
    scan_url = f"{BASE_URL}/scan/image"
    
    # Create a dummy image
    dummy_image_path = "test_image.png"
    from PIL import Image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(dummy_image_path)
    
    print(f"Uploading {dummy_image_path} to {scan_url}...")
    
    headers = {
        "Authorization": f"Bearer {token}"
        # NOTE: Requests library requests sets Content-Type boundary automatically!
    }
    
    files = {
        "file": (dummy_image_path, open(dummy_image_path, "rb"), "image/png")
    }
    
    try:
        response = requests.post(scan_url, headers=headers, files=files)
        
        print("\n--- Response ---")
        print(f"Status Code: {response.status_code}")
        print(f"Body: {response.text}")
        
        if response.status_code == 200:
            print("\nSUCCESS! The endpoint is working correctly.")
        else:
            print("\nFAILURE! The endpoint returned an error.")
            
    finally:
        # Cleanup
        if os.path.exists(dummy_image_path):
            try:
                os.remove(dummy_image_path)
            except:
                pass

if __name__ == "__main__":
    try:
        import requests
        import PIL
    except ImportError:
        print("Please run: pip install requests pillow")
        sys.exit(1)
        
    test_scan()
