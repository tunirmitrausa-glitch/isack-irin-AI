import requests
from config import NVIDIA_API_KEY, JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

api_key = "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"

url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "stream": False,
    "messages": [
        {"role": "user", "content": "Say hello and tell me you're ready to help build Isack."}
    ]
}

response = requests.post(url, headers=headers, json=payload)
data = response.json()
answer = data["choices"][0]["message"]["content"]
print(answer)
