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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy 
import requests 
from spacy import displacy
from bs4 import BeautifulSoup
from tkinter import *
from tkinter import messagebox 
from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter import simpledialog
import torch # type: ignore
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


text = ""
summary = ""


app = Flask(__name__, static_url_path='', static_folder='static')
app.secret_key = 'your_secret_key'

# Route for uploading a file
@app.route('/upload-file', methods=['POST'])
def upload_file():
    # Perform action 1 (e.g., retrieve data from database)
    previous_data = session.get('previous_data', None)

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    pdf = pdfquery.PDFQuery(file.filename)
    pdf.load()
    text = pdf.pq('LTTextLineHorizontal').text()
    text = textpreprocessing(text)

    session['previous_data'] = previous_data
    session['text'] = text
    return render_template('game_main_page.html', previous_data=previous_data, text=text)


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

    previous_data = session.get('previous_data', None)

    global text, summary
    print("\n summary \n")
    nlp = spacy.load("en_core_web_sm")
    # Process the text with spaCy
    doc = nlp(text)
    
    # Calculate the importance score for each sentence based on the sum of token weights
    sentence_importance = {}
    for sent in doc.sents:
        sentence_importance[sent] = sum(token.vector_norm for token in sent if not token.is_stop)
    
    # Sort sentences by importance score
    sorted_sentences = sorted(sentence_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Set the number of sentences for the summary
    num_sentences = 5  # You can adjust this number as needed
    
    # Generate the summary
    summary = " ".join(sent.text for sent, _ in sorted_sentences[:num_sentences])

    session['previous_data'] = previous_data
    session['summary'] = summary

    return render_template('game_main_page.html', previous_data=previous_data, summary=summary)


def QandA():
    global text, summary
    userinput = simpledialog.askstring('Q&A', 'What you want to ask?')

    userinput = str(userinput)
    summary = str(text)

    model_name = 'deepset/roberta-base-squad2'

    nlp = pipeline('question-answering', model = model_name, tokenizer = model_name)
    
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     question_answerer = pipeline('question-answering', model=model_name,)
#     question_answerer[{'question': userinput, 'context': text}]
    
    inputs0 = tokenizer(userinput, summary, return_tensors="pt")
    output0 = model(**inputs0)

    
    answer_start_idx = torch.argmax(output0.start_logits)
    anser_end_idx = torch.argmax(output0.end_logits)
    
    answer_tokens = inputs0.input_ids[0, answer_start_idx: anser_end_idx + 1]
    answer = tokenizer.decode(answer_tokens)
    messagebox.showinfo("OutPut", "ques: {} \n answer: {}".format(userinput, answer))

    print("ques: {} \n answer: {}".format(userinput, answer))



@app.route("/")
def home():
    return render_template('game_main_page.html')



if __name__ == '__main__':
    app.run(debug=True)