import random
import json
import re
import pickle
import socket
import numpy as np
import requests as req
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
    # check if in the words list
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
    host = 'localhost'
    port = 8001
    sock = socket.socket()
    sock.bind(("", port))
    sock.listen(5)

    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36'
    headers={'User-Agent':user_agent}

    while True:
        calc=False
        con, address = sock.accept()
        data = con.recvfrom(1024)
        message = str(data[0], 'utf-8')

        # 计算器
        if re.search('^[0-9+\-][0-9+\-*/\.]*', message):
            try:
                res= message + '=' + str(eval(message))
            except:
                res = 'In this age, still doing traditonal math?'
        # 百科
        elif message.split()[0] == "百科":
            r = req.get('https://baike.baidu.com/item/' + message.split()[1], headers=headers)
            try:
                r.encoding = 'utf-8'
                regex = re.compile('<div class="lemma-summary" label-module="lemmaSummary">(\s*)<div class="para" label-module="para">([\s\S]*?)</div>(\s*)</div>')
                res = re.findall(regex, r.text)[0][1]
            except:
                res = '好难啊我有点看不懂'
        # 普通对话
        else:
            ints = predict(message)
            res = getresponse(ints,intends)
        
        con.sendall(bytes(res, 'utf-8'))
        con.close()

test_bot()
