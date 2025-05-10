from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd

# Load pre-trained model and tokenizer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline for emotion detection
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# Sample social media texts
texts = [
    "I just got a promotion at work! Feeling amazing!",
    "I'm so sad and exhausted right now.",
    "Why is everyone so fake? I'm sick of it.",
    "This is the best day of my life!",
    "I feel nothing anymore. Just... empty."
]

# Analyze emotions
results = []
for text in texts:
    scores = emotion_classifier(text)[0]
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    top_emotion = sorted_scores[0]
    results.append({"Text": text, "Emotion": top_emotion['label'], "Confidence": round(top_emotion['score'], 3)})

# Display results in a DataFrame
df = pd.DataFrame(results)
print(df)