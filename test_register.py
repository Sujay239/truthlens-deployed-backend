import requests
import random
import string

API_URL = "http://127.0.0.1:8000/auth/register"

def generate_random_string(length=8):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

username = f"user_{generate_random_string()}"
email = f"{username}@example.com"
first_name = "Test"
last_name = "User"
password = "password123"

payload = {
    "username": username,
    "email": email,
    "password": password,
    "first_name": first_name,
    "last_name": last_name,
    "full_name": f"{first_name} {last_name}"
}

print(f"Sending payload: {payload}")

try:
    response = requests.post(API_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
