import pickle
import random
import json

# TODO: reduce the stem of every word
import nltk
import numpy
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
    lemmatizer = WordNetLemmatizer

    intends = json.loads(open('intends.json').read())

    words = []
    learns = []
    documents = []
    ignores = [',', '.', '!', '?', '，', '。']

    for intend in intends['intends']:
        for pattern in intend['patterns']:
            wordList = nltk.word_tokenize(pattern)
            words.append(wordList)
            documents.append((wordList, intend['tags']))
            if intend['tags'] not in learns:
                learns.append(intend['tags'])

    words = [lemmatizer.lemmatize(word) for word in words if word not in ignores]
    words = sorted(set(words))

    learns = sorted(set(learns))

    pickle.dump(words, open('words.pkl ', 'wb'))
    pickle.dump(words, open('learns.pkl ', 'wb'))

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

    chatting_model = Sequential
    chatting_model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(64, activation='relu'))
    chatting_model.add(Dropout(0.5))
    chatting_model.add(Dense(len(training_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, mementum=0.9, nesterov=True)
    chatting_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    chatting_model.fit(numpy.array(training_x), numpy.array(training_y), epochs=200, batch_size=5, verbose=1)
    chatting_model.save('chatting_model.model')

    print("TRAINING DONE")


function_start()
function_training()
