import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import requests
import json
from requests.auth import HTTPBasicAuth
from config import NVIDIA_API_KEY, JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

# Step 1: Record your voice
duration = 5
sample_rate = 44100

print("Recording... speak now.")
recording = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1)
sd.wait()
write("voice_command.wav", sample_rate, recording)
print("Recording finished.")

# Step 2: Transcribe with Whisper
print("Transcribing...")
model = whisper.load_model("base")
result = model.transcribe("voice_command.wav")
spoken_text = result["text"]
print(f"You said: {spoken_text}")

# Step 3: Ask NVIDIA's model to extract the ticket key and target status
nvidia_url = "https://integrate.api.nvidia.com/v1/chat/completions"
nvidia_headers = {
    "Authorization": f"Bearer {NVIDIA_API_KEY}",
    "Content-Type": "application/json"
}

extraction_prompt = f"""Extract the Jira ticket key and target status from this spoken command.
Respond ONLY with valid JSON in this exact format, nothing else:
{{"issue_key": "SCRUM-19", "status": "Task In Progress"}}

Valid statuses are exactly: "To Do", "Task In Progress", "Completed"

Command: "{spoken_text}"
"""

nvidia_payload = {
    "model": "nvidia/nemotron-3-super-120b-a12b",
    "stream": False,
    "messages": [{"role": "user", "content": extraction_prompt}]
}

nvidia_response = requests.post(
    nvidia_url, headers=nvidia_headers, json=nvidia_payload)
nvidia_data = nvidia_response.json()
extracted_text = nvidia_data["choices"][0]["message"]["content"]

print(f"Extracted: {extracted_text}")

# Parse the JSON the model returned
command = json.loads(extracted_text)
issue_key = command["issue_key"]
new_status = command["status"]

# Step 4: Move the ticket in Jira
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
jira_headers = {"Accept": "application/json",
                "Content-Type": "application/json"}

transitions_url = f"https://{JIRA_SITE}/rest/api/3/issue/{issue_key}/transitions"
response = requests.get(transitions_url, headers=jira_headers, auth=auth)
transitions = response.json()["transitions"]

transition_id = None
for t in transitions:
    if t["name"].lower() == new_status.lower():
        transition_id = t["id"]

if transition_id is None:
    print(f"Could not find a transition to '{new_status}'.")
else:
    payload = {"transition": {"id": transition_id}}
    move_response = requests.post(
        transitions_url, headers=jira_headers, auth=auth, json=payload)
    if move_response.status_code == 204:
        print(f"Successfully moved {issue_key} to '{new_status}'.")
    else:
        print(
            f"Failed. Status: {move_response.status_code}, Response: {move_response.text}")
