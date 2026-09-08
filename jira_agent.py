import requests
from requests.auth import HTTPBasicAuth
from config import NVIDIA_API_KEY, JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

# Which Jira project to summarize
JIRA_PROJECT = "SCRUM"

# Pull all issues from the project
url = f"https://{JIRA_SITE}/rest/api/3/search/jql"
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
headers = {"Accept": "application/json"}
params = {"jql": f"project = {JIRA_PROJECT}", "fields": "summary,status"}

response = requests.get(url, headers=headers, auth=auth, params=params)
data = response.json()

issue_summary = ""
for issue in data["issues"]:
    key = issue["key"]
    summary = issue["fields"]["summary"]
    status = issue["fields"]["status"]["name"]
    issue_summary += f"{key}: {summary} — [{status}]\n"

# Now send this to NVIDIA's model to summarize
nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"

nvidia_headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

nvidia_payload = {
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "stream": False,
    "messages": [
        {"role": "user", "content": f"Here is our current sprint data:\n\n{issue_summary}\n\nGive me a short, plain-English summary of where the sprint stands."}
    ]
}

nvidia_response = requests.post(
    nvidia_url, headers=nvidia_headers, json=nvidia_payload)
nvidia_data = nvidia_response.json()
answer = nvidia_data["choices"][0]["message"]["content"]

print(answer)
