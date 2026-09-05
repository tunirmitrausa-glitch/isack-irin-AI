import requests
from requests.auth import HTTPBasicAuth
from config import NVIDIA_API_KEY, JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

# Jira credentials
jira_email = "tunirmitra101@gmail.com"
jira_token = "ATATT3xFfGF07h0N22hzu2yLqvoK_cY4sc9-YFNitN1XzL_oObHld9ZNyGuMVe3CyjGJvB-92O3n7LHebj7DnKf6xS6zeyLA2gik0SexoKy2juWyvsdUFnGEbRZIPFALEVntgnaf1qlKXQ-h9lk_3ca2qnlsP13D1wK0B3PhsKzbuDEt0wHTBF8=3CE6FA5D"
jira_site = "tunirmitra101.atlassian.net"

# Pull all issues from the SCRUM project
url = f"https://{jira_site}/rest/api/3/search/jql"
auth = HTTPBasicAuth(jira_email, jira_token)
headers = {"Accept": "application/json"}
params = {"jql": "project = SCRUM", "fields": "summary,status"}

response = requests.get(url, headers=headers, auth=auth, params=params)
data = response.json()

issue_summary = ""
for issue in data["issues"]:
    key = issue["key"]
    summary = issue["fields"]["summary"]
    status = issue["fields"]["status"]["name"]
    issue_summary += f"{key}: {summary} — [{status}]\n"

# Now send this to NVIDIA's model to summarize
nvidia_api_key = "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"

nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"

nvidia_headers = {
    "Authorization": f"Bearer {nvidia_api_key}",
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
