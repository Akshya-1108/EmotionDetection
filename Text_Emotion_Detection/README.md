# Emotion Classifier using Transformers

This is a simple Python-based emotion classifier that uses a pre-trained Hugging Face transformer model to predict emotions from text input.

## 🚀 Features

- Uses Hugging Face's `transformers` library.
- Supports any compatible emotion classification model (default: `j-hartmann/emotion-english-distilroberta-base`).
- Can run on both CPU and GPU (CUDA support).
- Easy to extend or integrate with web or desktop apps.

## 📁 Project Structure

```
emotion-classifier/
│
├── model/
│   └── model.py               # Contains the EmotionClassifier class
│
├── main.py                    # Entry point to input text and get emotion predictions
│
├── README.md                  # Project documentation
│
└──requirements.txt            # project requirements
```
## 🧠 How the Model Predicts Emotion

The model uses a transformer-based neural network (`DistilRoBERTa`) trained on the [GoEmotions dataset](https://github.com/google-research/goemotions), which consists of over 58k English Reddit comments annotated with 27 emotion categories (later mapped to a smaller set like joy, sadness, fear, etc.).

### 🔍 Emotion Prediction Logic

- The input text is **tokenized** using a pre-trained tokenizer.
- The tokens are passed through a **transformer model**, which outputs raw scores (logits).
- A **softmax** layer converts these logits into probabilities across the emotion categories.
- The emotion with the **highest probability** is selected as the predicted label.

### ⚠️ Important Note on Accuracy

Since emotions like **fear** and **sadness** can sometimes overlap in language, the model may misclassify them, especially when:
- The sentence lacks **explicit fear-related keywords** (like "scared", "panic", "anxious").
- The phrasing is more introspective or passive, which the model maps to **sadness** instead.

To improve accuracy for edge cases:
- Use stronger, more emotion-specific phrasing in input text.
- Fine-tune the model with additional annotated examples.
- Consider using a **multi-label classification model** if overlapping emotions need to be captured.
"""

## 🧠 Model Used

- [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
  - Trained for emotion classification on English text.
  - Outputs labels like: `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`, etc.

## 🛠️ Installation

```bash
git clone https://github.com/your-username/emotion-classifier.git
cd emotion-classifier
pip install torch transformers
```

## ▶️ Usage

Run the script:

```bash
python main.py
```

Enter any English sentence when prompted:

```
Text Here... --> I am feeling really great and grateful today!
Predicted Emotion: joy
```

## 🧩 Sample Inputs & Predictions

| Input Text                                      | Predicted Emotion |
|------------------------------------------------|-------------------|
| "I am very happy and excited!"                 | joy               |
| "This is frustrating and annoying!"            | anger             |
| "I feel like I'm all alone."                   | sadness           |
| "I'm nervous about the interview tomorrow."    | fear              |


## 📚 Requirements

- Python 3.7+
- `torch`
- `transformers`

Install using pip:

```bash
pip install torch transformers
```

## 📌 Notes

- Make sure you have internet access the first time you run the code to download the model from Hugging Face.
- Once downloaded, the model is cached for offline use.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.