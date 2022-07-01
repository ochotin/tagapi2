#import json
from flask import Flask, request, render_template
#import nltk
#from nltk.tokenize import word_tokenize
import sklearn
import joblib
# import pickle


app = Flask(__name__)

def text_cleaner2(x):
    x = x.split(' ')
    return x

vectorizer = joblib.load("./New_tfidf_vectorizer_1.joblib")
model = joblib.load("./New_model_1.joblib")
multilabel_binarizer = joblib.load("./New_multilabel_binarizer_1.joblib")

@app.route('/')
def loadPage():
     return render_template('index.html')

@app.route('/tag_pred', methods=['POST', 'GET'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        Question = request.form.get('Question')
        Question_clean = text_cleaner2(Question)
        X_tfidf = vectorizer.transform([Question_clean])
        # predict = model.predict(X_tfidf)
        # tags_prediction = multilabel_binarizer.inverse_transform(predict)
        return render_template('index.html', tags_prediction = X_tfidf)
    return render_template('index.html')

if __name__ == "__main__":
        app.run()