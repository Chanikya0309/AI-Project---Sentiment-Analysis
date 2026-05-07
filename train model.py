import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Bigger dataset
data = {
    "text": [

        # Positive reviews
        "I love this product",
        "This is amazing",
        "Very happy with service",
        "I liked the movie",
        "Excellent work",
        "Fantastic experience",
        "This is very good",
        "Awesome product",
        "I enjoyed this",
        "Best purchase ever",
        "Super quality",
        "Highly recommended",
        "Very satisfied",
        "Wonderful app",
        "Great support team",
        "Absolutely loved it",
        "The product works perfectly",
        "Very impressive",
        "Brilliant performance",
        "Good experience overall",

        # Negative reviews
        "Worst experience ever",
        "I hate this",
        "Very bad product",
        "I disliked the movie",
        "Terrible service",
        "Very poor quality",
        "This is awful",
        "Not good at all",
        "Waste of money",
        "Extremely disappointing",
        "Horrible experience",
        "Very unhappy",
        "Pathetic support",
        "The app crashes often",
        "Bad customer service",
        "Completely useless",
        "Low quality product",
        "Very frustrating",
        "I regret buying this",
        "Disappointed with performance"
    ],

    "sentiment": [

        # Positive labels
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",

        # Negative labels
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Clean reviews
df["text"] = df["text"].apply(clean_text)

# Features and labels
X = df["text"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Better ML model
model = LogisticRegression()

# Train model
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")