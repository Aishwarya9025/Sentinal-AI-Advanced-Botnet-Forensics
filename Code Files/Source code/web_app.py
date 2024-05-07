""" 


    PLEASE NOTE:

    This is an interactive web app created with StreamLit.

    It's hosted on Heroku here:
    https://hate-speech-predictor.herokuapp.com/

    If you use any of this code, please credit with a link to my website:
    https://www.sidneykung.com/


"""

# importing relevant python packages
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# modeling
from sklearn import svm
# sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# model_results = st.beta_container()
# sentiment_analysis = st.beta_container()


user_text = st.text_input('Enter Tweet', max_chars=280)

model_results = st.container()
sentiment_analysis = st.container()
with model_results:
    st.subheader('Prediction:')
    if user_text:
        
        user_text = re.sub('[%s]' % re.escape(
            string.punctuation), '', user_text)
        # tokenizing
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(user_text)
        # removing stop words
        stopwords_removed = [token.lower()
                             for token in tokens if token.lower() not in stop_words]
        # taking root word
        lemmatizer = WordNetLemmatizer()
        lemmatized_output = []
        for word in stopwords_removed:
            lemmatized_output.append(lemmatizer.lemmatize(word))

        # instantiating count vectorizor
        count = CountVectorizer(stop_words=stop_words)
        X_train = pickle.load(open(
            r'C:\Users\ANGELINE\OneDrive\Documents\project code\Source code\pickle/X_train_2.pkl', 'rb'))
        X_test = lemmatized_output
        X_train_count = count.fit_transform(X_train)
        X_test_count = count.transform(X_test)

        # loading in model
        final_model = pickle.load(open(
            r'C:\Users\ANGELINE\OneDrive\Documents\project code\Source code\pickle/final_log_reg_count_model.pkl', 'rb'))

        # apply model to make predictions
        prediction = final_model.predict(X_test_count[0])

        # if prediction == 0:
        #     st.subheader('**Not Hate Speech**')
        # else:
        #     st.subheader('**Hate Speech**')
        #st.text('')

with sentiment_analysis:
    if user_text:
       
        analyzer = SentimentIntensityAnalyzer()
        # the object outputs the scores into a dict
        sentiment_dict = analyzer.polarity_scores(user_text)
        if sentiment_dict['compound'] >= 0.05:
            category = ("**Positive ✅**")
        elif sentiment_dict['compound'] <= - 0.05:
            category = ("**Negative 🚫**")
        else:
            category = ("**Neutral ☑️**")

        # score breakdown section with columns
        breakdown, graph = st.columns(2)
        with breakdown:
            # printing category
            st.write("Your Tweet is rated as", category)
            # printing overall compound score
            st.write("**Compound Score**: ", sentiment_dict['compound'])
            # printing overall compound score
            st.write("**Polarity Breakdown:**")
            st.write(sentiment_dict['neg']*100, "% Negative")
            st.write(sentiment_dict['neu']*100, "% Neutral")
            st.write(sentiment_dict['pos']*100, "% Positive")
        with graph:
            sentiment_graph = pd.DataFrame.from_dict(
                sentiment_dict, orient='index').drop(['compound'])
            st.bar_chart(sentiment_graph)

