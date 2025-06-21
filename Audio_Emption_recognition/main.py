import sounddevice as sd
import numpy as np
import queue
import keyboard
import matplotlib.pyplot as plt
import librosa
import torch
from transformers import *

# Load model and feature extractor with error handling
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
    model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
except Exception as e:
    print(f"Error loading model or feature extractor: {e}")
    exit(1)

def predict_emotion(audio, rate=16000):
    try:
        inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(inputs.input_values)
            predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1)
            emotion = model.config.id2label[predicted_label.item()]
        return emotion
    except Exception as e:
        print(f"Error during emotion prediction: {e}")
        return None

q = queue.Queue()
fs = 44100 
channels = 1
frames = []

def callback(indata, frames, time, status):
    if status:
        print(f"Sounddevice status: {status}")
    q.put(indata.copy())

print("Press and hold the 'r' key to record audio. Release to stop.")

recording = False

try:
    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        while True:
            if keyboard.is_pressed('r') and not recording:
                print("Recording started...")
                recording = True
            elif not keyboard.is_pressed('r') and recording:
                print("Recording stopped.")
                break
            sd.sleep(100)
except Exception as e:
    print(f"Error with audio input stream: {e}")
    exit(1)

while not q.empty():
    frames.append(q.get())

if not frames:
    print("No audio recorded. Please try again.")
    exit(1)

try:
    audio_data = np.concatenate(frames, axis=0)
    print(f"Recorded {len(audio_data)/fs:.2f} seconds of audio.")
except Exception as e:
    print(f"Error processing audio frames: {e}")
    exit(1)

time = np.linspace(0, len(audio_data) / fs, num=len(audio_data))

try:
    plt.figure(figsize=(12, 4))
    plt.plot(time, audio_data)
    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error displaying waveform: {e}")

audio_data = audio_data.flatten()

# Resample to 16kHz for model compatibility
try:
    audio_data_16k = librosa.resample(audio_data, orig_sr=fs, target_sr=16000)
except Exception as e:
    print(f"Error during resampling: {e}")
    exit(1)

emotion = predict_emotion(audio_data_16k)
if emotion is not None:
    print(f"Predicted emotion: {emotion}")
else:
    print("Emotion prediction failed.")
