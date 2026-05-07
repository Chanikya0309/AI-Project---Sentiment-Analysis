import streamlit as st
import pickle
import re

# Load saved model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Prediction function
def predict_sentiment(text):
    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)

    return prediction[0]

# Streamlit UI
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):

    result = predict_sentiment(user_input)

    if result == "positive":
        st.success("Positive 😊")

    else:
        st.error("Negative 😠")