from flask import Flask, render_template, request, redirect, url_for, session
from pathlib import Path
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords

import spacy
from spacy import displacy
from IPython.core.display import display, HTML

import numpy as np

from keras.models import Model, Sequential
from keras import models
from tensorflow import keras
import pickle
import pandas as pd
import html
import string
from Privacy_Dictionary.liwc_class import Liwc

def extract_features(user_story):
    PERSONS = ['I', 'ME', 'MY', 'MINE', 'YOU', 'YOUR', 'YOURS', 'HE', 'SHE', 
           'HIS', 'HER', 'HIM', 'THEY', 'THEM', 'THEMSELVES', 'OUR', 'WE']
    LOCATION = ['LOC', 'GPE', 'ORG', 'FAC', 'CARDINAL']
    ALL_ENTITIES = LOCATION + ['PERSON', 'HEALTH', 'MONEY', 'DATE', 'TIME']
    
    doc = nlp(u''+user_story)
    
    text_tokenized = []
    modified_tokens = []
    modified_tokens_dep = []
    modified_tokens_pos = []
        
    for t in doc:
        if str(t) not in punctuation:
            if t.ent_type_ == '':
                if t.text.upper() in PERSONS:
                    modified_tokens.append('PERSON')
                    text_tokenized.append(t.text)
                else:
                    modified_tokens.append(t.text)
                    text_tokenized.append(t.text)
            else:
                modified_tokens.append(t.ent_type_)
                text_tokenized.append(t.text)
    
    for t in doc:
        if str(t) not in punctuation:
            modified_tokens_dep.append(t.dep_)
 
    for t in doc:
        if str(t) not in punctuation:
            modified_tokens_pos.append(t.pos_)
    
    c,keywords, words_category = dictionary.parse(user_story.lower().split(' '))
    categories_list = [list(i) for i in c.items()]
    if str(keywords) == "[]":
        keywords = "none"
        categories_list = "none"

    return modified_tokens, modified_tokens_dep, modified_tokens_pos, categories_list, keywords, text_tokenized, words_category


def prepare_input_privacy(sentence):
    
    max_length=558
    output = disclo_cnn.layers[-4].output
    disclo_cnn_cutted = Model(disclo_cnn.input, output)

    disclo_cnn_cutted.trainable = False
    for layer in disclo_cnn_cutted.layers:
        layer.trainable = False
    modified_tokens, modified_tokens_dep, modified_tokens_pos, counter_list, k, k_grams, words_category = extract_features(sentence)
    

    encoded1 = tokenizer.texts_to_sequences(modified_tokens[:])
    encoded2 = tokenizer.texts_to_sequences(modified_tokens_dep[:])
    encoded3 = tokenizer.texts_to_sequences(modified_tokens_pos[:])
    
    encoded1_x = np.zeros((1,max_length))
    count=0
    for i in encoded1:
        if len(i) >0:
            encoded1_x[0,count]=i[0]
        count+=1

    encoded2_x = np.zeros((1,max_length))
    count=0
    for i in encoded2:
        if len(i) >0:
            encoded2_x[0,count]=i[0]
        count+=1

    encoded3_x = np.zeros((1,max_length))
    count=0
    for i in encoded3:
        if len(i) >0:
            encoded3_x[0,count]=i[0]
        count+=1
    encoded4_x = np.zeros((1,max_length))
    count=0
    for i in tokenizer.texts_to_sequences(str(counter_list)):
        if len(i) >0:
            encoded4_x[0,count]=i[0]
        count+=1
    encoded5_x = np.zeros((1,max_length))
    count=0
    for i in tokenizer.texts_to_sequences(str(k)):
        if len(i) >0:
            encoded5_x[0,count]=i[0]
        count+=1
    output_disclo = disclo_cnn_cutted.predict([encoded1_x,encoded2_x,encoded3_x])
    df_coeff = pd.DataFrame(
    {'word': k_grams
    })
    
    
    return encoded4_x, encoded5_x, output_disclo, df_coeff, k, words_category

def weights_to_HTML(df_coeff, keywords):
    #Highlight for privacy words
    highlighted_text = pd.DataFrame(columns=["token"])
    count=0
    for word in df_coeff['word']:
        if count<(len(df_coeff['word'])):
    
            if ((word in keywords) & (word not in string.punctuation)):
                highlighted_text.loc[count] = '<span style="background-color:rgba(207, 0, 15,0.8);color:#000000;">' + html.escape(word) + '</span>'
            else:
                highlighted_text.loc[count] = '<span style="color:#000000;">' + word + '</span>'
    
        count+=1
    
    return highlighted_text

app = Flask(__name__, template_folder='templates')
# Configure a secret SECRET_KEY
app.config['SECRET_KEY'] = 'mysecretkey'


# LOAD THE MODELS, DICTIONARY AND TRANSFORMERS!
dictionary = Liwc('Privacy_Dictionary/privacydictionary_TAWC.dic')
disclo_cnn = keras.models.load_model('Disclosure_CNN/cnn.h5')
with open('Disclosure_CNN/transformer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
nlp = spacy.load('Disclosure_NER')
privacy_detector = keras.models.load_model('Privacy_Detector_TL/privacy_detector.h5')

user_story = ''


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/prediction',methods = ['POST'])
def prediction():
    
    global user_story
    
    user_story = str(request.form['user_story'])
    input1,input2,input3, df_coeff, keywords, words_category= prepare_input_privacy(user_story)

    prediction = privacy_detector.predict([input1,input2,input3])
    
    explain = weights_to_HTML(df_coeff, keywords)
    

    return render_template('prediction.html',result = prediction, explain = explain, words_category = words_category)



if __name__ == '__main__':
    app.run(debug=True)