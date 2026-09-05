import requests
from requests.auth import HTTPBasicAuth
from config import JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
headers = {"Accept": "application/json", "Content-Type": "application/json"}

# Which ticket to move, and to which status
issue_key = "SCRUM-19"   # change this to a real ticket key from your board
new_status = "Task In Progress"  # the exact column name you want to move it to

# Step 1: find the available transitions for this issue
transitions_url = f"https://{JIRA_SITE}/rest/api/3/issue/{issue_key}/transitions"
response = requests.get(transitions_url, headers=headers, auth=auth)
transitions = response.json()["transitions"]

print("Available transitions:")
for t in transitions:
    print(f"- {t['name']} (id: {t['id']})")

# Step 2: find the transition ID that matches the status we want
transition_id = None
for t in transitions:
    if t["name"].lower() == new_status.lower():
        transition_id = t["id"]

if transition_id is None:
    print(
        f"Could not find a transition to '{new_status}'. Check the exact column name.")
else:
    # Step 3: perform the transition
    payload = {"transition": {"id": transition_id}}
    move_response = requests.post(
        transitions_url, headers=headers, auth=auth, json=payload)
    if move_response.status_code == 204:
        print(f"Successfully moved {issue_key} to '{new_status}'.")
    else:
        print(
            f"Failed to move ticket. Status: {move_response.status_code}, Response: {move_response.text}")
