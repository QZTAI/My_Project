import tokenize
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
from nltk.translate.bleu_score import modified_precision
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import spacy 
import requests 
from fractions import Fraction
# from spacy import displacy
# from bs4 import BeautifulSoup
# import torch # type: ignore
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from spire.doc import * 
from spire.doc.common import * 
# pip install tensorflow
# pip install tf-keras
# pip install textblob
# pip install allennlp
# pip install scipy=1.12.0
# pip install torch
from textblob import TextBlob
# import allennlp
# from allennlp.predictors import Predictor
# from bert_score import score
# from evaluate import load
# bertscore = load("bertscore")
# Pratical 11
from rouge_score import rouge_scorer
import gensim
from accessory_functions import google_vec_file

text = ""
summary = ""
selected_model = 0

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
        text = textpreprocessing(text)
        # print(text)

        # Store the uploaded text in the session

        return render_template('Main.html', previous_data=previous_data, text=text)

    elif file_extension == ".docx":
        document = Document()
        # Load the document
        document.LoadFromFile(file.filename, FileFormat.Docx)

        # Get the text from the document
        text = document.GetText()
        text = textpreprocessing(text)

        return render_template('Main.html', text=text, summary=summary)


    
def remove_text_after_word(text, word):
    index = text.find(word)
    if index != -1:
        return text[:index]
    else:
        return text

# Function for text preprocessing
def textpreprocessing(text):

    text_to_remove = "Evaluation Warning: The document was created with Spire.Doc for Python."
    cleaned_text = text.replace(text_to_remove, '')

    word_to_remove_after = "REFERENCES"
    cleaned_text = remove_text_after_word(cleaned_text, word_to_remove_after)
    
    # Regular expression pattern to match citations (Format for citation = [1])
    citation_pattern = r'\[[0-9]+\]'
    # Replace citations with an empty string
    cleaned_text = re.sub(citation_pattern, '', cleaned_text)

    # Regular expression pattern to match APA style citations (Format for citation = (Name, year))
    apa_citation_pattern = r'\(\w+,\s*\d{4}\)'
    # Replace APA style citations with an empty string
    cleaned_text = re.sub(apa_citation_pattern, '', cleaned_text)

    # Cleaning Text (Remove Email / Remove URL)
    # Regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Replace URLs with an empty string
    cleaned_text = re.sub(url_pattern, '', cleaned_text)

    # Cleaning Text (Remove Email / Remove URL)
    # Regular expression pattern to match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    # Replace email addresses with an empty string
    cleaned_text = re.sub(email_pattern, '', cleaned_text)

    # Replace the ? and ! to .
    cleaned_text = cleaned_text.replace('!', '.')
    cleaned_text = cleaned_text.replace('?', '.')

    # Regular expression pattern to match punctuation marks except period
    punctuation_pattern = r'[^\w\s\.\s]'
    # Replace punctuation marks except period with an empty string
    cleaned_text = re.sub(punctuation_pattern, '', cleaned_text)

    # Lowercasing
    cleaned_text = cleaned_text.lower()

    # Tokenize the text into words
    tokenize_text = word_tokenize(cleaned_text)

    # NLTK STOP WORD
    # text_stopwords = [word for word in tokenize_text if not word in stopwords.words()]

    lemmatizer=WordNetLemmatizer()
    lemmatized_content = []

    for word in tokenize_text:
        lemmatized_word = lemmatizer.lemmatize(word)
        lemmatized_content.append(lemmatized_word)
    
    # Joining the lemmatized words back into a string
    cleaned_text = ' '.join(lemmatized_content)
    return cleaned_text 

def tokenize_and_lemmatize(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

# =============================== summarisation ==================================================
# Evaluating for Summarisation
def Evaluating_Summary(generated_text):
    # From CHATGPT
    reference_text1 = "Subject: Apology for Recent Absence from Lectures Dear Dr. Rashidah, I hope this email finds you well. I apologize for my recent absence from your lectures and for not informing you sooner. Unfortunately, I've been unable to access my student email since January 2021 due to technical issues. Regarding my absences, on February 17, 2021, I was unwell but did not seek medical attention, resulting in no medical certificate. On February 24, 2021, I couldn't attend class due to my sister's wedding, and though I intended for a friend to inform you, it didn't happen as planned. Additionally, my return to the university after the wedding was delayed due to heavy rain, causing me to miss a subsequent flight. I'm aware I missed a listening test on February 24, 2021, and I'm concerned about its impact on my grade. Would it be possible to retake the test? I suggest Friday morning or an alternative project as options. I understand the importance of this test and am committed to making up for the missed opportunity. Once again, I apologize for any inconvenience caused and assure you of my commitment to attending all future classes and communicating promptly if unable to do so. I eagerly await your response. Thank you for your understanding."
    reference_text2 = "The paper discusses the importance of early detection of brain tumors and the role of machine learning algorithms, particularly in segmenting, detecting, and classifying brain tumors using FMRI or MRI images. It reviews various methods and approaches employed in this field, including preprocessing, segmentation, feature extraction, and classification. The common procedure for these algorithms involves preprocessing the image to remove noise, segmenting the image to identify potential tumor regions, and then classifying features such as intensity, shape, and texture of these regions. Despite the advancements in machine learning approaches for brain tumor detection, many of these methods have not been widely adopted yet, indicating the ongoing need for research in this area. The paper presents a comprehensive survey of existing literature, categorizing approaches into segmentation and detection/classification. Segmentation methods include techniques like fuzzy c-means clustering, adaptive k-means clustering, and multi-kernel learning, while detection/classification methods range from support vector machines to convolutional neural networks. Overall, the paper concludes that while many approaches have achieved high accuracies, there is still room for improvement and potential for combining existing methods to further enhance performance. The integration of deep learning techniques with precise feature extraction methods shows promise for future advancements in brain tumor detection and classification."


    reference_text1 = tokenize_and_lemmatize(reference_text1)
    reference_text2 = tokenize_and_lemmatize(reference_text2)
    generated_text = tokenize_and_lemmatize(generated_text)

    cleaned_text1 = ' '.join(reference_text1)
    cleaned_text2 = ' '.join(reference_text2)
    cleaned_text3 = ' '.join(generated_text)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores1 = scorer.score(cleaned_text1, cleaned_text3)
    scores2 = scorer.score(cleaned_text2, cleaned_text3)

    print("Using Rouge Scorer to evaluate PDF 1:")
    print(scores1)
    print("Using Rouge Scorer to evaluate PDF 2:")
    print(scores2)


# ============== START TAI QI ZHENG
def split_sentences(text):
    # Split the text by periods
    sentences = text.split('.')

    # Remove any leading or trailing whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences

# Function for text summarisation
@app.route('/summary_TAIQIZHENG', methods=['POST'])
def summary():
    global text, summary
    combined_sentences = []

    text_for_summary = split_sentences(text)
    num = len(text_for_summary)

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    for i in range(0, num, 10):
        combined_sentences.append(' '.join(text_for_summary[i:i+10]))
    
    for x in range(len(combined_sentences)):
        print(len(combined_sentences))
        summary_cell = summarizer(combined_sentences[x], max_length=130, min_length=30, do_sample=False)
        # Ensure that summary_cell contains only strings
        summary_cell = [sent['summary_text'] for sent in summary_cell]  # Extract 'summary_text' from each dictionary
        summary += ' '.join(summary_cell)
    
    result = Evaluating_Summary(summary)
    print("This is the Evaluating for summary by using Word2Vec : ")
    print(result)
    return render_template('game_main_page_TAIQIZHENG.html', text=text, summary=summary)
# ============== END   TAI QI ZHENG

# ============== START TAN XUE WEN
@app.route('/summary_TANXUEWEN', methods=['POST'])
def nltk_summarizer():
    global text, summary
    summary = nltk_summarizer(text)
    result = Evaluating_Summary(summary)
    print("This is the Evaluating for summary by using Word2Vec : ")
    print(result)
    return render_template('game_main_page_TANXUEWEN.html', text=text, summary=summary)

def nltk_summarizer(text):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    sumValues = 0
    summary = ""
    # Create a frequency table for words
    freqTable = dict()
    # Populate the frequency table
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Calculate the score for each sentence based on word frequencies
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    # Calculate the average sentence score
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.1 * average)):
            summary += " " + sentence
    return summary

# ============== END   TAN XUE WEN

# ============== START KONG KAI LE
# Function to summarize text using TextBlob
@app.route('/summary_KONGKAILE', methods=['POST'])
def summarize_text():
    global text, summary
    num_sentences = 10
    blob = TextBlob(text)
    sentences = blob.sentences
    
    # Calculate polarity for each sentence
    sentence_polarities = [(sentence, sentence.sentiment.polarity) for sentence in sentences]
    
    # Sort sentences by polarity
    sorted_sentences = sorted(sentence_polarities, key=lambda x: x[1], reverse=True)
    
    # Select top sentences as summary
    top_sentences = [sentence for sentence, polarity in sorted_sentences[:num_sentences]]
    
    # Join top sentences to create the summary
    summary = ' '.join(map(str, top_sentences))

    result = Evaluating_Summary(summary)
    print("This is the Evaluating for summary by using Word2Vec : ")
    print(result)
    return render_template('game_main_page_KONGKAILE.html', text=text, summary=summary)

# ============== END   KONG KAI LE
# =============================== summarisation ==================================================



# =========================== Get data ========================================
@app.route('/process_model_TAIQIZHENG', methods=['POST'])
def process_model_TAIQIZHENG():
    global selected_model
    selected_model = request.form['model']  
    print(selected_model)
    return render_template('chatbox_TAIQIZHENG.html')

@app.route('/get_TAIQIZHENG', methods=['get', 'POST'])
def chat_TAIQIZHENG():
    global text, selected_model
    msg = request.form["msg"]
    intput = msg
    print(selected_model)
    if selected_model == "1":
        print(1)
        return deepset_tinyroberta_squad2(intput, text)
    elif selected_model == "2":
        print(2)
        return deepset_roberta_base_squad2(intput, text)
    else:
        return deepset_tinyroberta_squad2(intput, text)


@app.route('/process_model_TANXUEWEN', methods=['POST'])
def process_model_TANXUEWEN():
    global selected_model
    selected_model = request.form['model']  
    print(selected_model)
    return render_template('chatbox_TANXUEWEN.html')

@app.route('/get_TANXUEWEN', methods=['get', 'POST'])
def chat_TANXUEWEN():
    global text, selected_model
    msg = request.form["msg"]
    intput = msg
    print(selected_model)
    if selected_model == "1":
        print(1)
        return QandA_BERT(intput, text)
    elif selected_model == "2":
        print(2)
        return QandA_DistilBERT(intput, text)
    else:
        QandA_BERT(intput, text)

@app.route('/process_model_KONGKAILE', methods=['POST'])
def process_model_KONGKAILE():
    global selected_model
    selected_model = request.form['model']  
    print(selected_model)
    return render_template('chatbox_KONGKAILE.html')

@app.route('/get_KONGKAILE', methods=['get', 'POST'])
def chat_KONGKAILE():
    global text, selected_model
    msg = request.form["msg"]
    intput = msg
    print(selected_model)
    if selected_model == "1":
        print(1)
        return deepset_electra_base_squad2(intput, text)
    elif selected_model == "2":
        print(2)
        return etalab_ia(intput, text)
    else:
        return deepset_electra_base_squad2(intput, text)
    

# =========================== Get data ========================================

# Performance Measurement for Q&A
references0 = "he wanted to say sorry for missing her classes"
references1 = "he went home to attend his sister wedding in his village"
references2 = "of the bad weather"
references3 = "to take the listening test"
references4 = "to come early for all Dr Rashidahs class"

references5 = "Review of Brain Tumour Segmentation, Detection and Classification Algorithms in fMRI Images"
references6 = "malignant tumor and benign tumor"
references7 = "functional magnetic resonance imaging fmri or magnetic resonance imaging mri"
references8 = "headaches, nausea and problems concerning balance, hearing and vision."
references9 = "researchers all over the world"

#Q01 what is the reason he writing this email
#Q02 why he missed the second class
#Q03 why he missed the flight
#Q04 what he request in email
#Q05 what he promising

#Q06 title for this paper
#Q07 What is the main type of brain tumour
#Q08 How the doctors detect and classify tumours
#Q09 Symptoms for brain tumors
#Q10 This paper is to motivate who

def Evaluation_QandA(predictions):
    global references0, references1, references2, references3, references4, references5, references6, references7, references8, references9
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    AIans = " ".join(predictions)

    scores0 = scorer.score(AIans, references0)
    scores1 = scorer.score(AIans, references1)
    scores2 = scorer.score(AIans, references2)
    scores3 = scorer.score(AIans, references3)
    scores4 = scorer.score(AIans, references4)
    scores5 = scorer.score(AIans, references5)
    scores6 = scorer.score(AIans, references6)
    scores7 = scorer.score(AIans, references7)
    scores8 = scorer.score(AIans, references8)
    scores9 = scorer.score(AIans, references9)


    print("======================================")
    print("Question 1")
    print(scores0)
    print("======================================")
    print("Question 2")
    print(scores1)
    print("======================================")
    print("Question 3")
    print(scores2)
    print("======================================")
    print("Question 4")
    print(scores3)
    print("======================================")
    print("Question 5")
    print(scores4)
    print("======================================")
    print("Question 6")
    print(scores5)
    print("======================================")
    print("Question 7")
    print(scores6)
    print("======================================")
    print("Question 8")
    print(scores7)
    print("======================================")
    print("Question 9")
    print(scores8)
    print("======================================")
    print("Question 10")
    print(scores9)
    print("======================================")

# =============================== Q&A ==================================================
# ============== START TAI QI ZHENG
def deepset_tinyroberta_squad2(input, text):

    errorReplay = "I don't know what 7 you say"

    model_name = "deepset/tinyroberta-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

    input = textpreprocessing(input)

    res = nlp(input, text)
    print(input)
    print(res)

    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)

    score = res['score']
    #return res['answer']
    if float(score) < 0.01 and float(score) > 5:
        return errorReplay
    else:
        return res['answer']

def deepset_roberta_base_squad2(input, text):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    res = nlp(input, text)
    print(input)
    print(res)
    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)
    return res['answer']
# ============== END   TAI QI ZHENG

# ============== START TAN XUE WEN

# BERT
def QandA_BERT(input, text):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    input = textpreprocessing(input)
    res = nlp(question=input, context=text)
    print(input)
    print(res)

    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)
    return res['answer']

# DISTILBERT
def QandA_DistilBERT(input, text):
    model_name = "distilbert-base-uncased-distilled-squad"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    input = textpreprocessing(input)
    res = nlp(question=input, context=text)
    print(input)
    print(res)

    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)
    return res['answer']

# ============== END   TAN XUE WEN

# ============== START KONG KAI LE
def deepset_electra_base_squad2(input, text):
    model_name = "deepset/electra-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    input = textpreprocessing(input)
    res = nlp(question=input, context=text)
    print(input)
    print(res)

    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)
    return res['answer']

def etalab_ia(input, text):
    model_name = "etalab-ia/camembert-base-squadFR-fquad-piaf"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    input = textpreprocessing(input)
    res = nlp(question=input, context=text)
    print(input)
    print(res)

    prediction = textpreprocessing(res['answer'])
    prediction = tokenize_and_lemmatize(prediction)
    Evaluation_QandA(prediction)
    print(prediction)
    return res['answer']
# ============== END   KONG KAI LE
# =============================== Q&A ==================================================

# =============================== QUESTION GENERATION ==================================================
def AnswerChecker(userinput, aianswer):
    userinput = tokenize_and_lemmatize(userinput)
    aianswer = tokenize_and_lemmatize(aianswer)
    print(userinput)
    print(aianswer)
    google_model = gensim.models.KeyedVectors.load_word2vec_format(google_vec_file, binary=True)
    results = google_model.n_similarity(userinput, aianswer)

    print(results)
    return results

@app.route('/UserInputText1', methods=['GET', 'POST'])
def index1():
    global text, question1, question2, question3, answer1, answer2, answer3, status1, status2, status3, user_input1, user_input2, user_input3
    status1 = ""
    user_input1 = ""

    if request.method == 'POST':

        #=================Answer Checker=================
        user_input1 = request.form['questionbox1']
        result = AnswerChecker(user_input1, answer1)
        if result <= 1:
            status1 = "Good job"
        elif result < 0.75:
            status1 = "The Answer should be no problem"
        elif result < 0.5:
            status1 = "The Answer mybe no detail enought"
        elif result < 0.25:
            status1 = "You need study more"
        else:
            status1 = "You need study more"
        #=================Answer Checker=================

        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)
    else:
        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)

@app.route('/UserInputText2', methods=['GET', 'POST'])
def index2():
    global text, question1, question2, question3, answer1, answer2, answer3, status1, status2, status3, user_input1, user_input2, user_input3
    status2 = ""
    user_input2 = ""

    if request.method == 'POST':

        #=================Answer Checker=================
        user_input2 = request.form['questionbox2']
        result = AnswerChecker(user_input2, answer2)
        if result <= 1:
            status2 = "Good job"
        elif result < 0.75:
            status2 = "The Answer should be no problem"
        elif result < 0.5:
            status2 = "The Answer mybe no detail enought"
        elif result < 0.25:
            status2 = "You need study more"
        else:
            status2 = "You need study more"
        #=================Answer Checker=================

        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)
    else:
        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)

@app.route('/UserInputText3', methods=['GET', 'POST'])
def index3():
    global text, question1, question2, question3, answer1, answer2, answer3, status1, status2, status3, user_input1, user_input2, user_input3
    status3 = ""
    user_input3 = ""

    if request.method == 'POST':

        #=================Answer Checker=================
        user_input3 = request.form['questionbox3']
        result = AnswerChecker(user_input3, answer3)
        if result <= 1:
            status3 = "Good job"
        elif result < 0.75:
            status3 = "The Answer should be no problem"
        elif result < 0.5:
            status3 = "The Answer mybe no detail enought"
        elif result < 0.25:
            status3 = "You need study more"
        else:
            status3 = "You need study more"
        #=================Answer Checker=================

        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)
    else:
        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3, status1=status1, status2=status2, status3=status3, user_input1=user_input1, user_input2=user_input2, user_input3=user_input3)


def QGT5L(text):
    tokenizer = AutoTokenizer.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
    model = AutoModelForSeq2SeqLM.from_pretrained("potsawee/t5-large-generation-squad-QuestionAnswer")
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    question_answer = question_answer.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
    question, answer = question_answer.split(tokenizer.sep_token)
    return question, answer


# =============================== QUESTION GENERATION ==================================================

question1 = ""
question2 = ""
question3 = ""
answer1 = ""
answer2 = ""
answer3 = ""
status1 = ""
status2 = ""
status3 = ""
user_input1 = ""
user_input2 = ""
user_input3 = ""

@app.route("/")
def home():
    global text, summary
    return render_template('Main.html', text=text, summary=summary)
# Externel
@app.route("/question_generation")
def question_generation():
    global text, question1, question2, question3, answer1, answer2, answer3
    combined_sentences = []
    questions = []
    answers = []
    question1 = ""
    question2 = ""
    question3 = ""
    answer1 = ""
    answer2 = ""
    answer3 = ""

    if text == "":
        return render_template('question_generation.html')
    else:
        text_for_summary = split_sentences(text)
        num = len(text_for_summary)
        for i in range(0, num, 5):
            combined_sentences.append('. '.join(text_for_summary[i:i+5]))
        for x in range(len(combined_sentences)):
            question, answer = QGT5L(combined_sentences[x])
            print(question)
            print(answer)
            questions.append(question)
            answers.append(answer)
        question1 = questions[0]
        question2 = questions[1]
        question3 = questions[2]
        answer1 = answers[0]
        answer2 = answers[1]
        answer3 = answers[2]
        return render_template('question_generation.html', text=text,question1=question1, answer1=answer1, question2=question2, answer2=answer2, question3=question3, answer3=answer3)

# TAI QI ZHENG
@app.route("/game_main_page_TAIQIZHENG")
def game_main_page_TAIQIZHENG():
    global text, summary
    return render_template('game_main_page_TAIQIZHENG.html', text=text, summary=summary)

@app.route("/chatbox_TAIQIZHENG")
def chatbox():
    return render_template('chatbox_TAIQIZHENG.html')



# TAN XUE WEN
@app.route("/game_main_page_TANXUEWEN")
def game_main_page_TANXUEWEN():
    global text, summary
    return render_template('game_main_page_TANXUEWEN.html', text=text, summary=summary)

@app.route("/chatbox_TANXUEWEN")
def chatbox_TANXUEWEN():
    return render_template('chatbox_TANXUEWEN.html')



# KONG KAI LE
@app.route("/game_main_page_KONGKAILE")
def game_main_page_KONGKAILE():
    global text, summary
    return render_template('game_main_page_KONGKAILE.html', text=text, summary=summary)

@app.route("/chatbox_KONGKAILE")
def chatbox_KONGKAILE():
    return render_template('chatbox_KONGKAILE.html')




if __name__ == '__main__':
    app.run(debug=True)