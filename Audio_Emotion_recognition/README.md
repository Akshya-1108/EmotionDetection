# Real-Time Speech Emotion Recognition with Wav2Vec2

This project captures audio from your microphone in real time, visualizes the waveform, and predicts the emotion expressed in the speech using a transformer-based model (`r-f/wav2vec-english-speech-emotion-recognition`).
Recording is controlled by pressing and holding the `r` key—recording starts when pressed and stops when released.

---

## Features

- **Real-time audio recording** via microphone
- **Waveform visualization** using `matplotlib`
- **Speech emotion recognition** using a pre-trained Wav2Vec2 transformer model
- **Robust error handling** for common issues (no audio, device errors, etc.)

---

## Requirements

- Python 3.8+
- [sounddevice](https://python-sounddevice.readthedocs.io/)
- numpy
- keyboard
- matplotlib
- librosa
- torch
- transformers

Install dependencies with:

```bash
pip install sounddevice numpy keyboard matplotlib librosa torch transformers
```

> **Note:**
> - You may need administrator/root privileges for `keyboard` on some systems.
> - Ensure your microphone is connected and accessible.

---

## Usage

1. **Run the script:**

```bash
python your_script_name.py
```

2. **Recording:**
    - Press and hold the `r` key to start recording.
    - Release the `r` key to stop recording.
3. **Output:**
    - The script will display a plot of your recorded audio waveform.
    - It will predict and print the detected emotion.

---

## Example Output

```
Press and hold the 'r' key to record audio. Release to stop.
Recording started...
Recording stopped.
Recorded 2.45 seconds of audio.
Predicted emotion: angry
```

And a waveform plot will appear.

---

## Troubleshooting

- **No audio recorded:**
Ensure your microphone is connected and not muted.
- **Permission errors:**
Try running the script as administrator/root if you encounter issues with keyboard or microphone access.
- **Model download issues:**
Ensure you have a stable internet connection on first run.

---

## Testing

- Try recording in different environments (quiet, noisy).
- Test with different speakers and emotions.
- Use short and long recordings to check robustness.

---

## License

This project is for educational and research purposes.
Model and code use open-source licenses; see individual package documentation for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/)
- [r-f/wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition)
- [Librosa](https://librosa.org/)
- [Matplotlib](https://matplotlib.org/)

---

**Feel free to fork, modify, and contribute!**

---
