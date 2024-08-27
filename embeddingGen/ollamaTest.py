import requests

url = "http://localhost:11434/api/chat"
payload = {
    "model": "llama3.1",
    "messages": [
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(f"Status code: {response.status_code}")
print(f"Response: {response.json()}")
