import random
import pickle
import numpy as np
import qalsadi.lemmatizer
import nltk
import pandas as pd
import tensorflow
import keras
import json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import fbeta_score



import tflearn

# ---------- READ RESPONSES FILE --------------
with open("Bot responses.json",'r', encoding='utf-8') as file:
    data = json.load(file)
try:
# ---------- UPLOAD, BAG_OF_WORDS, LABELS, TRAINING X AND Y--------------

    with open('data.pickle','rb') as f:
        words, classes, training, output_row = pickle.load(f)
except:
# ---------- CLEAN THE DATA IF THE EXTRACTED FEATURE DOSEN'T EXISTS --------------

    lemmer = qalsadi.lemmatizer.Lemmatizer()
    df = pd.read_csv("Medical Chatbot data.csv")

    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.','؟']

    for i in range (df['message'].count()):
        word = nltk.word_tokenize(df['message'][i])
        words.extend(word)
        documents.append((word, df['intent'][i]))
        if df['intent'][i] not in classes:
            classes.append(df['intent'][i])
    words = [lemmer.lemmatize(w) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))


    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, classes, training, output_row),f)


train_x = list(training[:, 0])
train_y = list(training[:, 1])

# ---------- TFLEARN MODEL --------------
# tensorflow.compat.v1.reset_default_graph()
# net = tflearn.input_data(shape=[None, len(train_x[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(train_y[0]), activation="softmax")
# net = tflearn.regression(net)
# model = tflearn.DNN(net)
try:
# ---------- UPLOAD THE MODEL --------------

    model = keras.models.load_model('Lina.dnn')
    # model.load('Lina.tflearn')
except:
# ---------- KERAS MODEL --------------

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # compile model
    model.compile(optimizer=sgd,
                      loss='categorical_crossentropy',
                      metrics=["accuracy"])

# ---------- FIT TFLEARN MODEL --------------

    # model.fit(np.array(train_x), np.array(train_y), n_epoch=100, batch_size=8, show_metric=True)

    model.save('Lina.dnn1')


# ---------- EXTRACT THE WORDS OF EACH CLASS AFTER LEMMETIZING THEM --------------
def bag_of_words(sentence, words):
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmer.lemmatize(word) for word in sentence_words]
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# ---------- PREDICT THR CLASS OF THE MESSAGE --------------
def predict_class(sentence, words):
    result = bag_of_words(sentence, words)
    res = model.predict(np.array([result]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = ""
    for r in results:
        return_list = classes[r[0]]
    return return_list


# ---------- FUNCTION THAT TAKE THE MESSAGE, APPLY THE CLEANING AND PREDICTION ON IT. THEN, CHOSSE RANDOM RESPONSE FROM RESPONSES FILE --------------
def chat(inp):
    while True:
        inp_clean = bag_of_words(inp, words)
        res = model.predict(np.array([inp_clean]))[0]
        res_index=np.argmax(res)
        print(res[res_index])
        if res[res_index]>0.8:
            result_class = predict_class(inp,words)


            for tg in data["intents"]:
                print(result_class)
                if result_class == tg['Intent']:
                    response = tg['purpose']

            return (random.choice(response))
        else:
            return ("مفهمتش قصدك, ممكن تسأل بطريقة تانية؟")


# ---------- CALCULATE THE ACCUARECY OF THE MODEL WITH TEST DATASET --------------
def calc_Acc():
    df2=pd.read_csv("test sheet2.csv")

    y_pred2 = [predict_class(df2['message'][i], words) for i in range(len(df2['message']))]
    print(fbeta_score(df2['intent'].values, y_pred2, average='micro', beta=0.5))
    print(y_pred2)



