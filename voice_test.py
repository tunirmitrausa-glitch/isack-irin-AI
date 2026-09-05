import sounddevice as sd
from scipy.io.wavfile import write
import whisper

# Record audio
duration = 5  # seconds
sample_rate = 44100

print("Recording... speak now.")
recording = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate, channels=1)
sd.wait()
write("voice_command.wav", sample_rate, recording)
print("Recording finished.")

# Transcribe with Whisper
print("Transcribing...")
model = whisper.load_model("base")
result = model.transcribe("voice_command.wav")

print("You said:")
print(result["text"])
