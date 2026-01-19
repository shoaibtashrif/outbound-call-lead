import requests
import os
import json

api_key = os.getenv("ULTRAVOX_API_KEY")
if not api_key:
    # Fallback to the one we saw in .env if env var not set in shell
    api_key = "1vCSStI6.8VL2fVYMFN9Pokgcy3kUNReMUEeTQBlB"

url = "https://api.ultravox.ai/api/voices"
headers = {"X-API-Key": api_key}

target_id = "d5594111-ddca-442a-8796-f0fced479a03"
found = False

while url:
    print(f"Fetching {url}...")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        print(f"Error: {resp.status_code} {resp.text}")
        break
    
    data = resp.json()
    for voice in data.get("results", []):
        if voice["voiceId"] == target_id:
            print(f"FOUND VOICE! Name: {voice.get('name')}")
            found = True
            break
    
    if found:
        break
        
    url = data.get("next")

if not found:
    print("Voice ID not found in any page.")
