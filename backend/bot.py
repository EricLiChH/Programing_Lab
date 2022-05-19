import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intends = json.loads(open('intends.json',encoding='utf-8').read())

words=pickle.load(open("words.pkl", "rb"))
learns=pickle.load(open("learns.pkl", "rb"))
model=load_model('chatting_model.h5')


def opti_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # divide sentence and lemmatize each word
    return sentence_words

def store_words(sentence):
    sentence_words = opti_sentence(sentence)
    container = [0]*len(words)
    for word_s in sentence_words:
        for i, word in enumerate(words):
            if word_s == word:
                container[i] = 1

    return np.array(container)


def predict(sentence):
    bow=store_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intends':learns[r[0]], 'probability':str(r[1])})
    return return_list

def getresponse(intend_list,intend_json):
    tag=intend_list[0]['intends']
    list_intends=intend_json['intends']
    for i in list_intends:
        if i['tag']==tag:
            result=random.choice(i['response'])
            break
    return result

def test_bot():
    print('now using your bot!')
    while True:
        message=input("user:")
        ints=predict(message)
        res=getresponse(ints,intends)
        print("bot:"+res)

test_bot()
