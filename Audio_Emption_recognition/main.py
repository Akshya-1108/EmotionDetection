import sounddevice as sd
import numpy as np
import queue
import keyboard
import matplotlib.pyplot as plt
import librosa
import torch
from transformers import *

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

def predict_emotion(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)
        predicted_label = torch.argmax(predictions, dim=-1)
        emotion = model.config.id2label[predicted_label.item()]
    return emotion


q = queue.Queue()
fs = 44100  # Sample rate
channels = 1
frames = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

print("Press and hold the 'r' key to record audio. Release to stop.")

recording = False

with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
    while True:
        if keyboard.is_pressed('r') and not recording:
            print("Recording started...")
            recording = True
        elif not keyboard.is_pressed('r') and recording:
            print("Recording stopped.")
            break
        sd.sleep(100)


while not q.empty():
    frames.append(q.get())

# print(len(frames))

if frames:
    audio_data = np.concatenate(frames, axis=0)
    print(f"Recorded {len(audio_data)/fs:.2f} seconds of audio.")
else:
    print("No audio recorded.")

time = np.linspace(0, len(audio_data) / fs, num=len(audio_data))

plt.figure(figsize=(12, 4))
plt.plot(time, audio_data)
plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

audio_data = audio_data.flatten()

emotion = predict_emotion(audio_data)
print(f"Predicted emotion: {emotion}")
 

