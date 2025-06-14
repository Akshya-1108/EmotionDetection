# Facial Emotion Detection

This project implements a real-time facial emotion detection system using a pretrained model from Hugging Face and OpenCV for face detection and video processing.

---

## Prerequisites

Python 3.8+
A webcam connected to your system
Required Python packages:pip install torch transformers opencv-python pillow

---

## Installation

Clone the repository:
```
git clone [repository](https://github.com/Akshya-1108/EmotionDetection)a# 🎭 Real-Time Facial Emotion Detection with Transformers
```
This project implements real-time facial emotion recognition using a pre-trained Vision Transformer model (`dima806/facial_emotions_image_detection`) from HuggingFace, along with OpenCV for capturing video feed and detecting faces.

---

## 🚀 Features

- 🧠 Emotion classification using a transformer-based image classification model.
- 📷 Real-time webcam feed with OpenCV.
- 🧍 Face detection using Haar Cascade.
- ⚡ CUDA GPU acceleration if available.
- 🐍 Clean Pythonic implementation.

---

## 🧠 Emotions Detected

The model can recognize the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 📦 Requirements

Ensure your environment meets the following requirements:

- Python **3.7–3.10** (MediaPipe and Transformers compatibility)
- `torch`
- `transformers`
- `opencv-python`
- `Pillow`

Install dependencies using pip:

```bash
pip install torch torchvision transformers opencv-python pillow
```

---

## 🧪 How to Run

1. Save the Python code below in a file named `emotion_detection.py`
2. Open terminal and run:

```bash
python emotion_detection.py
```

3. Press `Q` to quit the application.

---

## 📁 Folder Structure

```
project/
├── main.py
├── README.md
└── haarcascade
    └──haar_cascade.xml        
```

---

## 📖 Model Details

- **Model Name:** [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection)
- **Type:** Vision Transformer (ViT)
- **Interface:** HuggingFace `transformers` library
- **Processor:** `AutoImageProcessor`

---

## 📝 License

This project is released under the **MIT License**.

---
cd <repository-directory>
```


Install the dependencies:
```
pip install -r requirements.txt
```

Ensure you have a working webcam connected.


## Usage

Run the script:
```
python emotion_detection.py
```

The webcam feed will open, displaying detected faces with emotion labels.
```
Press q to quit the application.
```

How It Works

* **Model**: Uses the dima806/facial_emotions_image_detection pretrained model from Hugging Face for emotion classification.
* **Face Detection**: Utilizes OpenCV's Haar Cascade Classifier (haarcascade_frontalface_default.xml) to detect faces in the webcam feed.
* **Processing**: Each detected face is processed, converted to RGB, and passed through the model to predict emotions.
* **Display**: Bounding boxes and emotion labels are drawn on the video feed in real-time.

## Notes

The script assumes the presence of a webcam (index 0). Modify cv2.VideoCapture(0) if using a different camera index.
Ensure sufficient lighting and clear visibility for accurate face detection and emotion classification.
The model runs on GPU if available (cuda), otherwise defaults to CPU.

---

## Troubleshooting

* **Webcam** not detected: Check the camera index or ensure the webcam is properly connected.
* **Model** download issues: Verify internet connectivity, as the script downloads the pretrained model on first run.
* **Performance**: For faster processing, ensure a compatible GPU is available with CUDA support.

---

License

This project is licensed under the MIT License.
=======
## ⚠️⚠️PROJECT IN PROCESS⚠️⚠️
>>>>>>> e5262632b16055f02f4e79cd3936332d74609ea8
