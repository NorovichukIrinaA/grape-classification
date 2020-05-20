#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dropout


# Прогноз сорта по всем

def def_model_fragment():
    model = Sequential() 
    model.add(Dense(15*10, input_dim=2, activation='relu'))
    model.add(Dense(9*5, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',   metrics=["accuracy"])  
    return model


def def_model_sort():
    model = Sequential()  
    model.add(Dense(15*10, input_dim=2, activation='relu'))
    model.add(Dense(9*6, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',   metrics=["accuracy"])  
    return model

def def_model_fragment():
    model = Sequential() 
    model.add(Dense(15*10, input_dim=2, activation='relu'))
    model.add(Dense(9*5, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',   metrics=["accuracy"])   
    return model


# парамерт дата фрейм
def predict_Al_Cr_fragment_and_sort(X_test):
    header =  ['Al','Cr']
    header_sort = ['Cabernet %', 'Riesling %']
    header_fragment = ['peel %', 'pit %', 'puree %']
    sort_ = ['Cabernet', 'Riesling']
    fragment_ = ['peel', 'pit', 'puree'] 
    # создание преобразователя номера класса @predict_classes в сорт @fragment_
    label = LabelEncoder()
    label2 = LabelEncoder()
    l_sort = label.fit(sort_)
    l_fragment = label2.fit(fragment_)


    # DataFrame cохранение исходного @X_test
    df_predicting = pd.DataFrame(X_test.values, columns=header)
    X = X_test.values
    
    # Предсказание класса  sort
    X_predict_classes_sort = model_predict_classes_sort(X)
    # преобразование номера класса в sort_ (Место)
    X_predict_classes_sort = l_sort.inverse_transform(X_predict_classes_sort)

    # Предсказание класса  fragment
    X_predict_classes_fragment = model_predict_classes_fragment(X)
    # преобразование номера класса в fragment (сорт)
    X_predict_classes_fragment = l_fragment.inverse_transform(X_predict_classes_fragment)

    # Предсказание вероятности класса sort
    X_predict_sort = model_predict_sort(X)
    X_predict_sort = np.around((np.array(X_predict_sort) * 100), decimals=-1)

    # DataFrame @X_predict_sort
    df_predict_sort = pd.DataFrame((np.array(X_predict_sort)), columns=header_sort)

    # Предсказание вероятности класса fragment
    X_predict_fragment = model_predict_fragment(X)
    X_predict_fragment = np.around((np.array(X_predict_fragment) * 100), decimals=-1)

    # DataFrame @X_predict_fragment
    df_predict_None = pd.DataFrame((np.array(X_predict_fragment)), columns=header_fragment)

    # DataFrame общий - сорт, зона, исходные данные
    df = pd.concat([(pd.DataFrame(X_predict_classes_fragment, columns=['fragment'])),
                    (pd.DataFrame(X_predict_fragment, columns=header_fragment)),
                    (pd.DataFrame(X_predict_classes_sort, columns=['sort'])),
                    (pd.DataFrame(X_predict_sort, columns=header_sort)),
                    df_predicting], axis=1)
    return df


def model_predict_classes_sort(xtest):
    model = def_model_sort()
    model.load_weights('Sort_Al_Cr.h5')

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict_classes(xtest, batch_size=1)
    return predict_2


def model_predict_sort(xtest):
    model = def_model_sort()
    model.load_weights('Sort_Al_Cr.h5')

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict(xtest, batch_size=1)
    return predict_2


def model_predict_classes_fragment(xtest):
    model = def_model_fragment()
    model.load_weights('Fragment_Al_Cr.h5')

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict_classes(xtest, batch_size=1)
    return predict_2


def model_predict_fragment(xtest):
    model = def_model_fragment()
    model.load_weights('Fragment_Al_Cr.h5')

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict(xtest, batch_size=1)
    return predict_2


# In[ ]:




