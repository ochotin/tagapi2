#import json
import requests
from flask import Flask, request, render_template
import numpy as np
import re
# from bs4 import BeautifulSoup
#import spacy
#import en_core_web_sm
# import nltk
from nltk.corpus import stopwords
# from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import word_tokenize
from joblib import load

#nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
#nltk.download('all') 

app = Flask(__name__)
# Cleaning function for new question

# Load pre-trained models
#model_path = "C:/Users/Houda/Documents/OpenClassrooms/P5/"
print("before loading .........")
vectorizer = load("./New_tfidf_vectorizer_1.joblib")
print("after loading 1 .........")
model = load("./New_model_1.joblib")
print("after loading 2 .........")
multilabel_binarizer = load("./New_multilabel_binarizer_1.joblib")
print("after loading 3 .........")


#def text_cleaner(x, nlp, pos_list, lang="english"):
def text_cleaner(x, lang):
    """Function allowing to carry out the preprossessing on the textual data. 
        It allows you to remove extra spaces, unicode characters, 
        English contractions, links, punctuation and numbers.
        
        The re library for using regular expressions must be loaded beforehand.
        The SpaCy and NLTK librairies must be loaded too. 
    Parameters
    ----------------------------------------
    nlp : spacy pipeline
        Load pipeline with options.
        ex : spacy.load('en', exclude=['tok2vec', 'ner', 'parser', 
                                'attribute_ruler', 'lemmatizer'])
    x : string
        Sequence of characters to modify.
    pos_list : list
        List of POS to conserve.
    ----------------------------------------
    """
    # Remove POS not in "NOUN", "PROPN"
    # x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x

# nlp = spacy.load('en_core_web_sm')
pos_list = ["NOUN","PROPN"]


@app.route('/')
def loadPage():
     return render_template('index.html')

@app.route('/tag_pred', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        Question = request.form.get('Question')
        Question_clean = text_cleaner(Question, "english")
        X_tfidf = vectorizer.transform([Question_clean])
        predict = model.predict(X_tfidf)
        tags_prediction = multilabel_binarizer.inverse_transform(predict)
        return render_template('index.html', tags_prediction=tags_prediction)

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>Question: <input type="text" name="Question"></label></div>
               
               <input type="submit" value="Submit">
           </form>'''
           
           
if __name__ == "__main__":
    try:
        app.run()  # nosec debug
    except Exception as exception:
        raise exception
		
