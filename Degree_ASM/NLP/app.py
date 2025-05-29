from flask import Flask, render_template, url_for, request, session
from flask import jsonify
from werkzeug.utils import secure_filename
import os
import string
import re 
import sys
import pdfquery
#from pypdf import PdfReader 
from pdfquery import PDFQuery
import flask
import requests
#import pandas as pd
import nltk
#nltk.download()
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import spacy 
import requests 
# from spacy import displacy
# from bs4 import BeautifulSoup
# import torch # type: ignore
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from spire.doc import *
from spire.doc.common import *
# pip install tensorflow
# pip install tf-keras
text = ""
summary = ""


app = Flask(__name__, static_url_path='', static_folder='static')
app.secret_key = 'your_secret_key'

# Route for uploading a file
@app.route('/upload-file', methods=['POST'])
def upload_file():
    global text, summary
    text = ""
    summary = ""
    # Perform action 1 (e.g., retrieve data from database)
    previous_data = session.get('previous_data', None)

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Get the file type 
    split_tup = os.path.splitext(file.filename)
    file_extension = split_tup[1]
    print(file.filename)

    if file_extension == ".pdf":
        # creating a pdf reader object 
        pdf = pdfquery.PDFQuery(file.filename)
        pdf.load()
        text = pdf.pq('LTTextLineHorizontal').text()
        # text = textpreprocessing(text)
        # print(text)

        # Store the uploaded text in the session

        return render_template('game_main_page.html', previous_data=previous_data, text=text)

    elif file_extension == ".docx":
        document = Document()
        # Load the document
        document.LoadFromFile(file.filename, FileFormat.Docx)

        # Get the text from the document
        text = document.GetText()

        return render_template('game_main_page.html', text=text, summary=summary)


    


# Function for text preprocessing
def textpreprocessing(text):
    
    lower_text = text.lower()

    # Tokenize the text into words
    tokenize_text = word_tokenize(lower_text)

    # Define punctuation marks to remove (excluding full stop)
    punctuation_to_remove = ''.join([char for char in string.punctuation if char != '.'])

    # Remove punctuation marks except for full stop
    cleaned_words = [word for word in tokenize_text if word not in punctuation_to_remove]

    #NLTK STOP WORD
    text_stopwords = [word for word in cleaned_words if not word in stopwords.words()]

    lemmatizer=WordNetLemmatizer()
    lemmatized_content = []

    for word in text_stopwords:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_content.append(lemmatized_word)
    
    # Joining the lemmatized words back into a string
    lemmatized_text = ' '.join(lemmatized_content)
    return lemmatized_text 



# Function for text summarization
@app.route('/summary', methods=['POST'])
def summary():
    global text, summary
    text_for_summary = text
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(text_for_summary, max_length=130, min_length=30, do_sample=False)

    return render_template('game_main_page.html', text=text, summary=summary)




@app.route('/get', methods=['get', 'POST'])
def chat():
    msg = request.form["msg"]
    intput = msg
    return QandA(intput)


def QandA(input):

    global text
    model_name = "deepset/tinyroberta-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    res = nlp(input, text)
    print(res)

    print(res['answer'])

    return res['answer']








# TAI QI ZHENG
@app.route("/")
def home():
    return render_template('game_main_page.html')

@app.route("/chatbox")
def chatbox():
    return render_template('chatbox.html')



# TAN XUE WEN
@app.route("/game_main_page_TANXUEWEN")
def game_main_page_TANXUEWEN():
    return render_template('game_main_page_TANXUEWEN.html')

@app.route("/chatbox_TANXUEWEN")
def chatbox_TANXUEWEN():
    return render_template('chatbox_TANXUEWEN.html')



# KONG KAI LE
@app.route("/game_main_page_KONGKAILE")
def game_main_page_KONGKAILE():
    return render_template('game_main_page_KONGKAILE.html')

@app.route("/chatbox_KONGKAILE")
def chatbox_KONGKAILE():
    return render_template('chatbox_KONGKAILE.html')




if __name__ == '__main__':
    app.run(debug=True)