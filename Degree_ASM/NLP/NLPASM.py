

import easygui 

import os

import string

import sys

import pdfquery
from pypdf import PdfReader 
from pdfquery import PDFQuery

import flask

import requests

import pandas as pd

import nltk
#nltk.download()
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


import spacy 

import requests 
from spacy import displacy
from bs4 import BeautifulSoup

from tkinter import *

from tkinter import ttk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkPDFViewer import tkPDFViewer as pdf 
from PIL import Image, ImageTk
import PyPDF2

#===============================================================
#===============================================================


from flask import Flask, render_template 
  
app = Flask(__name__) 
  
# Define a route for the root URL ('/') 
@app.route('/') 
def index(): 
    # Fetch data from the database and prepare for rendering 
    data = get_data_from_database()  # Replace this with your actual data retrieval logic 
    # Render the 'index.html' template and pass the retrieved data for rendering 
    return render_template('index.html', data=data) 
  
# Placeholder for fetching data from the database 
def get_data_from_database(): 
    # Replace this function with your actual logic to retrieve data from the database 
    # For now, returning a sample data 
    return {'message': 'Hello, data from the database!'} 
  
if __name__ == '__main__': 
    # Run the Flask application 
    app.run(debug=True) 






#===============================================================
#===============================================================
pages = []
text = ""
#===========
root = Tk()
root.geometry('700x500+400+100')
#===========
icon = PhotoImage(file = "pdf_icon.png")
root.iconphoto(False, icon)
root.configure(bg="white")
root.title("PDF")
#===========
root.columnconfigure(0, weight = 1)
root.columnconfigure(1, weight = 3)
root.rowconfigure(0, weight = 1)
#===========
# Create text widget and specify size.
right_content = Text(root)
right_content.grid(row = 0, column = 1)

T = Text(root)
T.grid(row = 0, column = 0)

#frm = ttk.Frame(root, padding=10)
#frm.grid()

# Read File
def openfile():
    global text
    text_output = ""
    filename = filedialog.askopenfilename(initialdir="/Documents", title="Upload a PDF file", filetype=[("pdf files", "*.pdf")])
      
    # creating a pdf reader object 
    #reader = PdfReader(filename) 
    pdf = pdfquery.PDFQuery("Malaysia.pdf")
    pdf.load()
    text = pdf.pq('LTTextLineHorizontal').text()

    # printing number of pages in pdf file 
    #print(len(reader.pages)) 
  
    # getting a specific page from the pdf file 
    
    #for i in range(len(reader.pages)):
    #    page = reader.pages[i] 
  
        # extracting text from page 
    #    messy_text = page.extract_text() 
    #    text_output += messy_text
        
    #text = text_output.replace("\n", " ")
    T.insert(END, text)
    textpreprocessing()
    print(text)

    
def textpreprocessing():
    global text
    #nlp = spacy.load("en_core_web_sm")
    #pd.set_option("display.max_rows", 200)
    #doc = nlp(text)
    #displacy.render(doc, style="ent")
    
    #for ent in doc.ents:
    #    print(ent.text, ent.start_char, ent.end_char, ent.label_)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    text = word_tokenize(text)
    
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words
    text= [word for word in text if not word in all_stopwords]

    # initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # apply lemmatization function to column of dataframe
    text = [lemmatizer.lemmatize(token) for token in text]



        

# Define a function to close the window
def close():
   root.destroy()



#Create a Menu
my_menu= Menu(root)
root.config(menu=my_menu)
#Add dropdown to the Menus
file_menu=Menu(my_menu,tearoff=False)
my_menu.add_cascade(label="File",menu= file_menu)
file_menu.add_command(label="Open",command=openfile)
file_menu.add_command(label="Clear")
file_menu.add_command(label="Quit",command=close)

 
# Create a Button
btn = Button(root, text = 'Click me !', bd = '5', command = root.destroy) 
btn.grid(row = 0, column = 1)

root.mainloop()

