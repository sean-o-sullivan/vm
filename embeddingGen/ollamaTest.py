import requests

url = "http://127.0.0.1:11434/v1/chat/"

payload = {
	"model": "llama-3.1-8b",
	"prompt": "Hello, how are you?"
}

headers = {
	"Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)


print(f"status code: {response.status_code}")

print(f"response: {response.text}")

