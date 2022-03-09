from flask import Flask, jsonify, request
import numpy as np
#from sklearn.externals 
import joblib
from news_classifier_utils import *

import warnings

# For handling data
import scipy

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
    #with warnings.catch_warnings():
    	#warnings.simplefilter("ignore", category=UserWarning)
    clf = joblib.load('./model_files/model.pkl')
    vectorizer_news_text = joblib.load('./model_files/vectorizer_news_text.pkl')
    vectorizer_link = joblib.load('./model_files/vectorizer_link.pkl')
    vectorizer_authors = joblib.load('./model_files/vectorizer_authors.pkl')

    to_predict_list = request.form.to_dict()

    #cleaning the input data
    review_text = preprocess_data(to_predict_list['review_text'], stem_flag=False, lemm_flag=True, stopwords_to_be_removed=stopwords_list)
    print ("to_predict_list['review_text'] ---->", to_predict_list['review_text'])
    print(" cleaned review_text--->", review_text)
    news_text_vect = vectorizer_news_text.transform([review_text])
    link_vect = vectorizer_link.transform(["NA"])
    authors_vect = vectorizer_authors.transform(["UNKNOWN_AUTHOR"]) 

    test_vect = scipy.sparse.hstack([news_text_vect, link_vect, authors_vect
		 , 
                 np.array([0]), np.array([0]), 
                 np.array([0]), np.array([0])
                 #, np.array(df_test["sentence_count"][:,None])
                 ])

    pred = clf.predict(test_vect)
    print ("predicted category ------->  --->  >>>",int(pred[:1]))
    pred_cat =  get_category_name(int(pred[:1]))
    #pred_cat =  get_category_name(1)

    return jsonify({
		  # 'review news text': to_predict_list['review_text']
		   'prediction': pred_cat
		})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
