# Sentiment Analysis Project

-- Overview

The Sentiment Analysis Project is a Natural Language Processing (NLP) based web application that predicts whether a given text or review expresses a positive or negative sentiment. The project uses Machine Learning techniques to analyze textual data and classify user opinions in real time.

This application is built using Python, Scikit-learn, and Streamlit, making it lightweight, interactive, and easy to deploy.

The project demonstrates the complete Machine Learning workflow including:

1. Data preprocessing
2. Text cleaning
3. Feature extraction using TF-IDF
4. Model training
5. Model evaluation
6. Real-time prediction through a web interface

-- Project Objectives

Build a real-time sentiment prediction system
Understand NLP preprocessing techniques
Learn text vectorization using TF-IDF
Train and evaluate machine learning classification models
Deploy an ML model using Streamlit

-- Tech Stack
-- Programming Language

Python

-- Libraries & Frameworks

Pandas
NumPy
Scikit-learn
Regex (re)
Pickle
Streamlit

-- Machine Learning & NLP

TF-IDF Vectorization
Logistic Regression
Text Classification
Natural Language Processing (NLP)

-- Features

1. Real-time sentiment prediction
2. Interactive web interface using Streamlit
3. NLP-based text preprocessing
4. TF-IDF feature extraction
5. Machine Learning model training
6. Model serialization using Pickle
7. Clean and lightweight UI
8. Easy deployment and scalability

-- Project Workflow

1. Data Collection

A dataset containing positive and negative reviews is created and processed for training.

2. Text Preprocessing

The text data is cleaned by:

Converting text to lowercase
Removing special characters
Removing unnecessary symbols using Regular Expressions

3. Feature Extraction using TF-IDF

The cleaned text is converted into numerical vectors using TF-IDF (Term Frequency - Inverse Document Frequency).
TF-IDF helps the model understand the importance of words in a sentence while reducing the impact of very common words.

4. Model Training

The project uses Logistic Regression for sentiment classification.
The model learns patterns from training data and predicts whether a review is:

Positive
Negative

5. Model Evaluation

The model performance is evaluated using:
Accuracy Score
Classification Report
Precision
Recall
F1-Score

6. Model Saving

7. Streamlit Deployment

A Streamlit web application is created to allow users to:
Enter custom text
Click Predict
Instantly view sentiment results

-- Challenges Faced

Small datasets caused incorrect predictions for unseen words
Limited vocabulary reduced model accuracy
Handling text preprocessing and vectorization effectively
Improving prediction accuracy for real-world reviews

-- Improvements Implemented

Increased dataset size
Used TF-IDF vectorization
Switched from Naive Bayes to Logistic Regression
Added train-test split for better evaluation
Improved preprocessing pipeline

-- Learning Outcomes

Through this project, I learned:
NLP preprocessing techniques
Text vectorization methods
Machine Learning model training
Model deployment using Streamlit
End-to-end ML workflow implementation
Real-world challenges in sentiment analysis systems

-- Author
Chanikya Posham
Aspiring Data Scientist & AI/ML Enthusiast
