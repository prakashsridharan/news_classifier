from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
from news_classifier_utils import *

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)

nltk.download('wordnet')
nltk.download('stopwords')

stopwords_list = nltk.corpus.stopwords.words("english")


@app.route('/')
def hello_world():
    return 'News Classifier!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model.pkl')
    vectorizer_news_text = joblib.load('vectorizer_news_text.pkl')
    vectorizer_link = joblib.load('vectorizer_link.pkl')
    vectorizer_news_text = joblib.load('vectorizer_news_text.pkl')

    to_predict_list = request.form.to_dict()

    #cleaning the input data
    review_text = preprocess_data(to_predict_list['review_text'], stem_flag=False, lemm_flag=True, stopwords_to_be_removed=stopwords_list)
    news_text_vect = vectorizer_news_text.transform([review_text])
    pred = clf.predict(news_text_vect)
    if pred[0]:
        prediction = "0"
    else:
        prediction = "Not 0"

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
