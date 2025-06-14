# Emotion Detection using Transformers

This project is a comprehensive multi-model emotion recognition system built using transformer models. It consists of three independent modules:

1. **Text Emotion Recognition**
2. **Audio Emotion Recognition**
3. **Facial Emotion Recognition**

Each module is located in its own subdirectory and comes with a dedicated `README.md` file containing detailed setup and usage instructions.

---

 

---

## 🔗 Sub-Modules Documentation

### 📝 [Text Emotion Recognition](Text_Emotion_recognition/README.md )
Detects emotions from raw text using transformer-based NLP models. Trained on emotion-labeled datasets like GoEmotions.

### 🔊 [Audio Emotion Recognition]( )
Analyzes emotional tone in audio speech using acoustic features and transformer-based audio models (e.g., Wav2Vec).

### 😐 [Facial Emotion Recognition](Facial_Emotion_recognition)
Uses deep learning and transformers to classify facial expressions into emotion categories.

---

## 🛠 Requirements

- Python 3.7+
- `torch`, `transformers`, `scikit-learn`
- Submodule-specific dependencies listed in each subfolder

Install base requirements:

```bash
pip install torch transformers scikit-learn
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Contribution

Feel free to contribute to any module or improve the integration across modalities. Each module can be used independently or combined into a unified emotion recognition pipeline.
