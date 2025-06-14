import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model import EmotionClassifier

model_name = "j-hartmann/emotion-english-distilroberta-base"
prompt = input("Text Here... -->")

model = EmotionClassifier(model_name)

pred = model.predict(prompt)
print("Predicted Emotion:", pred)