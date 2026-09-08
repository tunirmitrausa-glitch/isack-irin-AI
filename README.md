# Isack

A voice-activated project management assistant. Say a wake word, speak a command like
*"move SCRUM-19 to in progress"*, and Isack transcribes it, figures out what you meant,
and moves the ticket on your Jira board — hands-free.

I built this to see how far I could get combining local wake-word detection, speech
transcription, and an LLM-based intent extractor into one low-latency voice pipeline,
using Jira as a concrete, real target system to act on.

## How it works

```
 microphone
     |
     v
 openWakeWord  --- listens continuously for "Hey Isack" / "Hey Irin"
     |
     v  (wake word detected)
 record ~6s of audio
     |
     v
 Whisper (local)  --- transcribes speech to text
     |
     v
 NVIDIA NIM (LLM)  --- extracts {"issue_key": ..., "status": ...} as JSON
     |
     v
 Jira REST API  --- looks up valid transitions, then performs the move
     |
     v
 ticket moved on your board
```

- **Wake word detection** runs continuously and fully locally via
  [openWakeWord](https://github.com/dscripka/openWakeWord), using two custom-trained
  ONNX models (`hey_isack.onnx`, `hey_irin.onnx`) so no audio leaves your machine until
  a wake word actually fires.
- **Transcription** runs locally via [OpenAI Whisper](https://github.com/openai/whisper)
  (`base` model).
- **Intent extraction** sends the transcribed text to an LLM hosted on
  [NVIDIA NIM](https://build.nvidia.com/) (`nvidia/nemotron-3-super-120b-a12b`), which is
  prompted to return strict JSON with the target ticket key and status.
- **Execution** hits the [Jira REST API v3](https://developer.atlassian.com/cloud/jira/platform/rest/v3/)
  to fetch that issue's valid transitions and perform the one matching the requested status.

## Project layout

| File | Purpose |
|---|---|
| `isack_listener.py` | **Main entry point.** Full pipeline: wake word -> record -> Whisper -> NIM -> Jira. |
| `detect_isack.py` | Standalone wake-word test tool — just prints a detection and confidence score, no recording/Jira. Useful for tuning models/thresholds. |
| `automatic_model_training_clean.ipynb` | Colab notebook that trains the custom wake-word models from scratch (synthetic TTS speech + noise/room-impulse augmentation), exported to ONNX. |
| `config.py` | Your local API keys/site (gitignored — see Setup). |
| `config.py.example` | Template for `config.py`. |
| `hey_isack.onnx`, `hey_irin.onnx` (+ `.onnx.data`) | Trained wake-word models, checked in since they're small (~200KB each). |
| `jira_agent.py` | Prototype: pulls all issues in a Jira project and asks the LLM for a plain-English sprint summary. No voice involved. |
| `move_ticket.py` | Minimal script to move one hardcoded ticket to one hardcoded status — useful for testing the Jira transition logic in isolation. |
| `voice_jira.py` | Earlier prototype of the full pipeline, before wake-word gating was added (records a fixed 5s clip immediately on run). |
| `voice_test.py` | Earliest prototype — just records audio and transcribes it with Whisper, no Jira/LLM. |
| `first_script.py` | The very first smoke test against the NVIDIA NIM chat completions endpoint. |

The prototype scripts (`first_script.py`, `jira_agent.py`, `move_ticket.py`,
`voice_jira.py`, `voice_test.py`) are kept as-is to show how the project evolved from a
single API call into the full voice pipeline; `isack_listener.py` is the one to actually run.

## Setup

**Requirements:**
- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) on your PATH (required by Whisper) — `brew install ffmpeg` on macOS
- A working microphone

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/isack.git
cd isack
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure your API keys**

```bash
cp config.py.example config.py
```

Then edit `config.py`:
- `NVIDIA_API_KEY` — get one for free at [build.nvidia.com](https://build.nvidia.com/) (NVIDIA NIM).
- `JIRA_EMAIL` — the email you log into Jira with.
- `JIRA_TOKEN` — create one at [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens).
- `JIRA_SITE` — your Jira Cloud domain, e.g. `your-domain.atlassian.net`.

`config.py` is gitignored and should never be committed.

**3. Wake word models**

`hey_isack.onnx` and `hey_irin.onnx` are included in this repo and ready to use. If you
want a different wake word, open `automatic_model_training_clean.ipynb` in Google Colab
(needs a GPU runtime), change the target phrase, and run it top to bottom — it walks
through generating synthetic training clips and exporting a new `.onnx` model.

## Running it

```bash
python isack_listener.py
```

Say **"Hey Isack"** or **"Hey Irin"**, wait for the prompt, then say something like:

> "Move SCRUM-19 to in progress"

Isack will transcribe it, ask the LLM to extract the ticket key and target status, and
move the ticket on your board.

Valid statuses are currently hardcoded to exactly `"To Do"`, `"Task In Progress"`, and
`"Completed"` in the extraction prompt in `isack_listener.py` — edit that list to match
your own board's column names.

To just test wake-word detection without doing anything else, run `detect_isack.py`.

## Known limitations

- **Wake word accuracy**: the models are trained on synthetic TTS speech with fairly
  modest training targets (60% accuracy / 25% recall thresholds, 1,000 synthetic clips
  per phrase), so expect occasional missed or false-positive detections rather than
  production-grade reliability.
- **Fixed listening window**: after a wake word fires, it records a fixed 6-second clip
  regardless of how long you actually talk.
- **Single-user Jira auth**: uses one hardcoded email/API-token pair from `config.py` —
  there's no multi-user auth or OAuth flow.
- **Board-specific status names**: the three valid statuses are hardcoded strings that
  have to exactly match your Jira board's transition names.
- **No conversational context**: each command is transcribed and parsed independently;
  there's no follow-up/clarification if the LLM extraction is ambiguous or wrong.
- **JSON parsing is best-effort**: if the LLM doesn't return valid JSON, the command is
  silently skipped rather than retried.

## Built with

- [openWakeWord](https://github.com/dscripka/openWakeWord) — on-device wake word detection
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) — synthetic TTS speech for wake-word training data
- [OpenAI Whisper](https://github.com/openai/whisper) — local speech-to-text
- [NVIDIA NIM](https://build.nvidia.com/) — hosted LLM inference for intent extraction
- [Jira REST API](https://developer.atlassian.com/cloud/jira/platform/rest/v3/) — ticket transitions
