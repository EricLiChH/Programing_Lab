import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intends = json.loads(open('intends.json').read())

words=pickle.load(open("words.pkl", "rb"))
classes=pickle.load(open("learns.pkl", "rb"))
model=load_model('chatting_model.h5')


def opti_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in words]
    # divide sentence and lemmatize each word
    return sentence_words

def store_words(sentence):
    sentence_words = opti_sentence(sentence)
    container = [0]*len(sentence_words)
    for word_s in sentence_words:
        for i, word in enumerate(words):
            if word_s == word:
                container[i] = 1

    return np.array(container)


def predict(sentence):
    bow=store_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
