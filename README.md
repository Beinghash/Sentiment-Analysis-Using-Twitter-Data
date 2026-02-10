# 🐦 Sentiment Analysis using Twitter Data

## 📌 Project Overview

This project implements a sentiment analysis system for Twitter data using Natural Language Processing (NLP) and Machine Learning techniques. 
The system analyzes user-provided tweet text and classifies it as either positive or negative sentiment. A trained machine learning model is integrated into a Flask-based web application 
to enable real-time sentiment prediction through an interactive user interface.

## 🎯 Objectives

- To understand and apply text preprocessing techniques for NLP

- To convert raw text data into numerical features using TF-IDF

- To train and evaluate a machine learning classification model

- To deploy the trained model using Flask

- To build an end-to-end ML pipeline from data → model → web app

## 🛠️ Tools & Technologies Used

| Category              | Tools / Technologies |
|----------------------|---------------------|
| Programming Language | Python 3 |
| Web Framework        | Flask |
| NLP Library          | NLTK |
| Machine Learning     | Scikit-learn |
| Feature Extraction  | TF-IDF Vectorizer |
| Data Handling        | Pandas, NumPy |
| Model Persistence   | Pickle |
| Development Tools   | Jupyter Notebook, VS Code |
| Environment Manager | Anaconda |


## 📊 Dataset Information

The dataset contains Twitter tweets and corresponding sentiment labels

Due to its large size, the dataset is not included directly in this repository

📥 Dataset Download Link (Google Drive):
👉 [Add your Google Drive link here]

After downloading, place the CSV file in this directory before running the notebook.

## ⚙️ Step-by-Step Overflow

The methodology of this project follows a systematic end-to-end machine learning pipeline, starting from data preprocessing and model training to deployment as a web application using Flask.

1. Dataset Acquisition

The dataset consists of Twitter tweets labeled for sentiment analysis, where each tweet is associated with a sentiment class:

0 → Negative (Hate Speech)

1 → Positive (Non-Hate Speech)

Due to the large size of the dataset, it is hosted externally and provided through a Google Drive link. The dataset is loaded into the system using the Pandas library for further processing.

2. Text Preprocessing

Raw Twitter data contains noise such as URLs, hashtags, mentions, emojis, and stopwords. To ensure consistency and improve model performance, a custom text preprocessing pipeline is implemented using regular expressions and NLTK.

The preprocessing steps include:

Converting all text to lowercase

Removing URLs and web links

Removing user mentions (@username) and hashtags

Removing numbers and special characters

Tokenizing text into words

Removing English stopwords using NLTK

Rejoining cleaned words into a single string

This preprocessing ensures that the input text during training and prediction follows the same format, which is critical for reliable sentiment classification.

3. Feature Extraction using TF-IDF

Machine learning models cannot process raw text directly. Therefore, the cleaned text is transformed into numerical features using the Term Frequency–Inverse Document Frequency (TF-IDF) technique.

Key characteristics of TF-IDF in this project:

Assigns higher weight to important words

Reduces the impact of frequently occurring but less informative words

Limits vocabulary size to the top 5000 features for efficiency

The TF-IDF vectorizer is fitted only on the training data to learn vocabulary and IDF values, and then applied to the test data to avoid data leakage.

4. Train–Test Split

The dataset is divided into:

80% training data

20% testing data

This split ensures that the model learns patterns from unseen data and allows for proper evaluation of generalization performance.

A fixed random state is used to maintain reproducibility of results.

5. Model Selection and Training

A Logistic Regression classifier is selected for this project due to its:

High efficiency on high-dimensional sparse text data

Fast training time

Strong baseline performance in sentiment analysis tasks

The model is trained using the TF-IDF feature vectors generated from the training dataset. During training, the model learns the relationship between word importance and sentiment labels.

6. Model Evaluation

After training, the model is evaluated on the test dataset using:

Accuracy

Precision

Recall

F1-score

These metrics provide a comprehensive understanding of the model’s classification performance, especially for imbalanced sentiment data.

7. Model Persistence

To enable reuse of the trained model during deployment, the following components are saved using Pickle:

The trained TF-IDF vectorizer

The trained Logistic Regression model

Saving both components ensures that the same vocabulary and learned parameters are used during real-time predictions, maintaining consistency between training and deployment.

8. Web Application Development using Flask

The trained model is integrated into a Flask-based web application. Flask serves as a bridge between the frontend and the machine learning model.

Key responsibilities of Flask include:

Rendering HTML templates

Handling user authentication (signup and login)

Receiving tweet input from the user

Preprocessing the input text

Converting text to TF-IDF features

Generating sentiment predictions

Displaying results dynamically on the web interface

9. Real-Time Prediction Workflow

When a user submits a tweet through the web interface:

The input is sent from HTML to Flask via a POST request

Flask preprocesses the input text

The saved TF-IDF vectorizer converts text into numerical features

The trained machine learning model predicts the sentiment

The numeric prediction is mapped to a human-readable label

The result is displayed back to the user on the result page

10. System Deployment (Local)

The Flask application runs locally on a development server, allowing users to interact with the sentiment analysis model in real time via a web browser. This deployment demonstrates the practical applicability of machine learning models in real-world web applications.

## 🧪 Results

The model successfully classifies tweets into:

- Positive (Non-Hate Speech)

- Negative (Hate Speech)

Logistic Regression provided:

- Fast training time

- Good accuracy for text-based classification

- The web interface allows real-time sentiment prediction

 ## 🌐 How to Run the Project Locally

Clone the repository

git clone <repository-link>


Open Anaconda Prompt and navigate to the project folder

cd D:\Mini-Project\MPNLP04\Code


Install required dependencies

conda install numpy scipy scikit-learn
pip install flask nltk


Run the Flask application

python app.py


Open browser and visit

http://127.0.0.1:5000/

## 📄 Documentation & Research Paper

📘 Project Documentation is included in this folder

📑 IEEE Research Paper related to this project is included in this folder

These files provide:

- Detailed system explanation

- Literature survey

- Experimental analysis

- Conclusions and future scope

## 🔮 Future Enhancements

- Integration with Twitter API for live tweet analysis

- Support for multiclass sentiment analysis

- Deployment on cloud platforms (Render / AWS / Heroku)

- Use of advanced models like LSTM / BERT

## 🎓 Academic Note

This project is developed as part of a college mini project for academic learning purposes and demonstrates the practical application of Machine Learning, NLP, and Web Deployment
