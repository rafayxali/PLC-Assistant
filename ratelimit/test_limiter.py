import requests
import time

URL = "http://127.0.0.1:8000/api/v1/chat"
payload = {"query": "Hello Aegis, what is the fault code for an S7-1200 CPU?"}

print("🚀 Blasting 6 rapid-fire requests to trigger the rate limiter...\n")

for i in range(1, 7):
    response = requests.post(URL, json=payload)
    print(f"Request #{i} -> Status Code: {response.status_code}")
    if response.status_code == 429:
        print(f"🛑 Success! Blocked on turn {i}: {response.json()['detail']}")

print("\n⏱️ Waiting 13 seconds for a token to refill...")
time.sleep(13)

print("🔄 Trying one more request after waiting...")
fallback_response = requests.post(URL, json=payload)
print(f"Request #7 -> Status Code: {fallback_response.status_code} (Should be back to normal)")