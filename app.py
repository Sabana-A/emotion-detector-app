
import streamlit as st
import joblib
import string
from nltk.corpus import stopwords

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set of stopwords
stop_words = set(stopwords.words('english'))

# Clean function
def clean_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🧠 Emotion Detector from Text")
st.write("Enter a sentence to find out the emotion!")

# Input text
user_input = st.text_area("Your Text")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        # Optional emoji dictionary
        emojis = {
            'happy': '😊', 'sadness': '😢', 'anger': '😡',
            'fear': '😨', 'love': '❤️', 'surprise': '😲'
        }
        
        st.success(f"**Predicted Emotion:** {prediction} {emojis.get(prediction, '')}")
