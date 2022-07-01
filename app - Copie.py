#import json
import requests
from flask import Flask, request, render_template
import numpy as np
import re
# from bs4 import BeautifulSoup
#import spacy
#import en_core_web_sm
import nltk
# from nltk.corpus import stopwords
# from nltk.stem import wordnet
# from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import word_tokenize
# from joblib import load
import pickle

#nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
#nltk.download('all') 

app = Flask(__name__)
# Cleaning function for new question


#def text_cleaner(x, nlp, pos_list, lang="english"):
def text_cleaner(x):
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
    # x = x.encode("ascii", "ignore").decode()
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
    # stop_words = stopwords.words("english")
    # Remove stop words
    # x = [word for word in x if word not in stop_words 
    #     and len(word)>2]
    # Lemmatizer
    # wn = nltk.WordNetLemmatizer()
    # x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x
	
def text_cleaner2(x):
    x = word_tokenize(x)
    return x

# nlp = spacy.load('en_core_web_sm')
# pos_list = ["NOUN","PROPN"]

# Load pre-trained models
# with open('New_tfidf_vectorizer.pkl', "rb") as fp:   # Unpickling
#    vectorizer = pickle.load(fp)
    
# with open('New_model.pkl', "rb") as fp:   # Unpickling
#     model = pickle.load(fp)
    
# with open('New_multilabel_binarizer.pkl', "rb") as fp:   # Unpickling
#    multilabel_binarizer = pickle.load(fp)

# with open('New_X_tfidf.pkl', "rb") as fp:   # Unpickling
#    New_X_tfidf = pickle.load(fp)
	
print("after loading .........")


@app.route('/')
def loadPage():
     return render_template('index.html')

@app.route('/tag_pred', methods=['POST', 'GET'])
def form_example():
    # handle the POST request
    if request.method == 'POST':
        Question = request.form.get('Question')
        print("------------------ question : ", Question)
        Question_clean = text_cleaner2(Question)
        # X_tfidf = vectorizer.transform([Question_clean]) 
        # predict = model.predict(X_tfidf)
        # predict = model.predict(New_X_tfidf)		
        # tags_prediction = multilabel_binarizer.inverse_transform(predict)
        # tags_prediction = "Python ... 2"
        tags_prediction = Question_clean
        return render_template('index.html', tags_prediction=tags_prediction)
#    return '''
#           <form method="POST">
#               <div><label>Question: <input type="text" name="Question"></label></div>
#               
#               <input type="submit" value="Submit">
#           </form>'''
           
           
# app.run(debug=True)

if __name__ == "__main__":
        app.run()
