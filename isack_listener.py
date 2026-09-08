import os
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import whisper
import requests
import json
from requests.auth import HTTPBasicAuth
from openwakeword.model import Model

from config import NVIDIA_API_KEY, JIRA_EMAIL, JIRA_TOKEN, JIRA_SITE

SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms chunks, openwakeword's expected input size
COMMAND_DURATION = 6  # seconds to record after wake word fires

# Whichever trained wake word models are present in this folder get loaded.
# Drop a new one (e.g. hey_irin.onnx) in here and it'll be picked up
# automatically, no code changes needed.
CANDIDATE_MODELS = ["hey_isack.onnx", "hey_irin.onnx"]
WAKEWORD_MODELS = [m for m in CANDIDATE_MODELS if os.path.exists(m)]
if not WAKEWORD_MODELS:
    raise FileNotFoundError(
        f"None of the expected wake word models were found: {CANDIDATE_MODELS}")
WAKE_THRESHOLD = 0.5

print("Loading wake word model(s)...")
oww_model = Model(
    wakeword_models=WAKEWORD_MODELS, inference_framework="onnx")

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

jira_auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_TOKEN)
jira_headers = {"Accept": "application/json",
                "Content-Type": "application/json"}


def handle_command(wakeword):
    """Record a command, transcribe it, extract a Jira action, and execute it."""
    print(f"'{wakeword}' detected! Listening for your command...")
    recording = sd.rec(int(COMMAND_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    wavfile.write("voice_command.wav", SAMPLE_RATE, recording)

    print("Transcribing...")
    result = whisper_model.transcribe("voice_command.wav")
    spoken_text = result["text"].strip()
    print(f"You said: {spoken_text}")

    if not spoken_text:
        print("Didn't catch anything, going back to listening.")
        return

    extraction_prompt = f"""Extract the Jira ticket key and target status from this spoken command.
Respond ONLY with valid JSON in this exact format, nothing else:
{{"issue_key": "SCRUM-19", "status": "Task In Progress"}}

Valid statuses are exactly: "To Do", "Task In Progress", "Completed"

Command: "{spoken_text}"
"""

    nvidia_response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "stream": False,
            "messages": [{"role": "user", "content": extraction_prompt}]
        }
    )
    response_json = nvidia_response.json()
    if "choices" not in response_json:
        print(f"NVIDIA API error: {response_json}")
        return
    extracted_text = response_json["choices"][0]["message"]["content"]
    print(f"Extracted: {extracted_text}")

    try:
        command = json.loads(extracted_text)
    except json.JSONDecodeError:
        print("Couldn't understand that as a Jira command.")
        return

    issue_key = command.get("issue_key")
    new_status = command.get("status")
    if not issue_key or not new_status:
        print("Missing ticket or status, skipping.")
        return

    transitions_url = f"https://{JIRA_SITE}/rest/api/3/issue/{issue_key}/transitions"
    response = requests.get(
        transitions_url, headers=jira_headers, auth=jira_auth)
    transitions = response.json().get("transitions", [])

    transition_id = None
    for t in transitions:
        if t["name"].lower() == new_status.lower():
            transition_id = t["id"]

    if transition_id is None:
        print(
            f"Could not find a transition to '{new_status}' for {issue_key}.")
        return

    move_response = requests.post(
        transitions_url, headers=jira_headers, auth=jira_auth,
        json={"transition": {"id": transition_id}}
    )
    if move_response.status_code == 204:
        print(f"Successfully moved {issue_key} to '{new_status}'.")
    else:
        print(
            f"Failed to move ticket. Status: {move_response.status_code}, Response: {move_response.text}")


def main():
    wake_names = ", ".join(f"'{os.path.splitext(m)[0]}'" for m in WAKEWORD_MODELS)
    print(f"Listening for {wake_names}... (Ctrl+C to stop)")
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype='int16', channels=1)
    stream.start()
    try:
        while True:
            audio_bytes, _ = stream.read(CHUNK_SIZE)
            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            prediction = oww_model.predict(audio)

            triggered = next(
                (wakeword for wakeword, score in prediction.items()
                 if score > WAKE_THRESHOLD),
                None)

            if triggered:
                stream.stop()
                handle_command(triggered)
                if hasattr(oww_model, "reset"):
                    oww_model.reset()
                time.sleep(1)
                stream.start()
                print(f"Listening for {wake_names} again...")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
