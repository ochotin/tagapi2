#import json
from flask import Flask, request, render_template
import nltk
from nltk.tokenize import word_tokenize
# import pickle


app = Flask(__name__)

def text_cleaner2(x):
    # x = word_tokenize(x)
    return x


@app.route('/')
def loadPage():
     return render_template('index.html')

@app.route('/tag_pred', methods=['POST', 'GET'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        Question = request.form.get('Question')
        Question_clean = text_cleaner2(Question)
        return render_template('index.html', tags_prediction=Question)
    return render_template('index.html')

if __name__ == "__main__":
        app.run()