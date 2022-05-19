<<<<<<< HEAD
import pickle
import random
import json

# TODO: reduce the stem of every word
import nltk
import numpy
import numpy as np
from nltk.stem import WordNetLemmatizer

import numpy as py

import torch

from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def function_start() -> None:
    print("THE BOT IS BEING TRAINED")


def function_training() -> None:
    lemmatizer = WordNetLemmatizer()

    intends = json.loads(open('intends.json',encoding='utf-8').read())

    words = []
    learns = []
    documents = []
    ignores = [',', '.', '!', '?', '，', '。']

    for intent in intends['intends']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            documents.append((word, intent['tag']))
            if intent['tag'] not in learns:
                learns.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignores]
    words = sorted(list(set(words)))
    # print(words)
    learns = sorted(set(learns))

    pickle.dump(words, open('words.pkl ', 'wb'))
    pickle.dump(learns, open('learns.pkl ', 'wb'))
    # change to learns

    training = []
    output_emp = [0] * len(learns)

    for document in documents:
        bags = []

        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word_pattern.lower()) for word_pattern in word_patterns]

        for word in words:
            if word in word_patterns:
                bags.append(1)
            else:
                bags.append(0)

        output_line = list(output_emp)
        output_line[learns.index(document[1])] = 1

        training.append([bags, output_line])

    random.shuffle(training)
    training = numpy.array(training)

    training_x = list(training[:, 0])
    training_y = list(training[:, 1])

    chatting_model = Sequential()
    chatting_model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(64, activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(len(training_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    chatting_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = chatting_model.fit(np.array(training_x), np.array(training_y), epochs=114514, batch_size=5, verbose=1)
    chatting_model.save('chatting_model.h5',hist)

def function_down():
    print("TRAINING DONE")


def main():
    function_start()
    i = 0
    j = 0
    # while(i < 1919810):
    #     i += 1
    #     while(j < 114514):
    #         j += 1
    function_training()
    function_down()

=======
import pickle
import random
import json

# TODO: reduce the stem of every word
import nltk
import numpy
import numpy as np
from nltk.stem import WordNetLemmatizer

import numpy as py

import torch

from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


def function_start() -> None:
    print("THE BOT IS BEING TRAINED")


def function_training() -> None:
    lemmatizer = WordNetLemmatizer()

    intends = json.loads(open('intends.json',encoding='utf-8').read())

    words = []
    learns = []
    documents = []
    ignores = [',', '.', '!', '?', '，', '。']

    for intent in intends['intends']:
        for pattern in intent['patterns']:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            documents.append((word, intent['tag']))
            if intent['tag'] not in learns:
                learns.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignores]
    words = sorted(list(set(words)))
    # print(words)
    learns = sorted(set(learns))

    pickle.dump(words, open('words.pkl ', 'wb'))
    pickle.dump(learns, open('learns.pkl ', 'wb'))
    # change to learns

    training = []
    output_emp = [0] * len(learns)

    for document in documents:
        bags = []

        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word_pattern.lower()) for word_pattern in word_patterns]

        for word in words:
            if word in word_patterns:
                bags.append(1)
            else:
                bags.append(0)

        output_line = list(output_emp)
        output_line[learns.index(document[1])] = 1

        training.append([bags, output_line])

    random.shuffle(training)
    training = numpy.array(training)

    training_x = list(training[:, 0])
    training_y = list(training[:, 1])

    chatting_model = Sequential()
    chatting_model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(64, activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(len(training_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    chatting_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = chatting_model.fit(np.array(training_x), np.array(training_y), epochs=114514, batch_size=5, verbose=1)
    chatting_model.save('chatting_model.h5',hist)

def function_down():
    print("TRAINING DONE")


def main():
    function_start()
    i = 0
    j = 0
    # while(i < 1919810):
    #     i += 1
    #     while(j < 114514):
    #         j += 1
    function_training()
    function_down()

>>>>>>> f0a32ca24e4c108db5376d2be3347eb6627f26f1
main()