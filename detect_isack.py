import os
import openwakeword
from openwakeword.model import Model
import numpy as np
import sounddevice as sd

# Load whichever trained wake word models are present in this folder.
# Drop a new one (e.g. hey_irin.onnx) in here and it'll be picked up
# automatically, no code changes needed.
CANDIDATE_MODELS = ["hey_isack.onnx", "hey_irin.onnx"]
wakeword_models = [m for m in CANDIDATE_MODELS if os.path.exists(m)]
if not wakeword_models:
    raise FileNotFoundError(
        f"None of the expected wake word models were found: {CANDIDATE_MODELS}")

model = Model(
    wakeword_models=wakeword_models,
    inference_framework="onnx"
)

sample_rate = 16000
chunk_size = 1280  # 80ms chunks, openwakeword's expected input size

wake_names = ", ".join(f"'{os.path.splitext(m)[0]}'" for m in wakeword_models)
print(f"Listening for {wake_names}... (Ctrl+C to stop)")


def callback(indata, frames, time, status):
    audio = np.frombuffer(indata, dtype=np.int16)
    prediction = model.predict(audio)
    for wakeword, score in prediction.items():
        if score > 0.5:
            print(f"Wake word '{wakeword}' detected! (confidence: {score:.2f})")


with sd.RawInputStream(
    samplerate=sample_rate,
    blocksize=chunk_size,
    dtype='int16',
    channels=1,
    callback=callback
):
    while True:
        sd.sleep(1000)
