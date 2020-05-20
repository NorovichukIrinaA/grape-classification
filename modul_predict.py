#!/usr/bin/env python
# coding: utf-8

# In[23]:


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

def def_model_all_Fragments_Variety():
    model = Sequential()
    model.add(Dense(6*67, input_dim=17, activation='sigmoid')) 
    model.add(Dense(8*17, activation='tanh'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',   metrics=["accuracy"]) 
    return model
 

# парамерт дата фрейм
def predict_All_Fragments_Variety(X_test):
    header = ['Al','Ba','Cr','Cu','Fe','Li','Mn','Na','Ni','Pb','Rb','Sr','Ti','Zn','Ca','K_','Mg']
    header_sort = ['Cabernet peel %', 'Cabernet pit %', 'Cabernet puree %', 'Riesling peel %', 'Riesling pit %', 'Riesling puree %']
    Sort = ['Cabernet peel', 'Cabernet pit', 'Cabernet puree', 'Riesling peel', 'Riesling pit', 'Riesling puree']
 
    label = LabelEncoder() 
    l_sort = label.fit(Sort)
    
    # DataFrame cохранение исходного @X_test
    df_predicting = pd.DataFrame(X_test.values, columns=header)
    X = X_test.values
    
    # Предсказание класса  
    X_predict_classes = model_predict_classes_Fragments_Variety(X)
    # преобразование номера класса в название
    X_predict_classes = l_sort.inverse_transform(X_predict_classes)
 
    # Предсказание вероятности класса  
    X_predict = model_predict_Fragments_Variety(X)
    X_predict = np.around((np.array(X_predict) * 100), decimals=-1)

    # DataFrame @X_predict
    df_predict = pd.DataFrame((np.array(X_predict)), columns=header_sort)

    # DataFrame общий - сорт, зона, исходные данные
    df = pd.concat([(pd.DataFrame(X_predict_classes, columns=['Sort'])),
                    (pd.DataFrame(X_predict, columns=header_sort)),  
                    df_predicting], axis=1)
    return df


def model_predict_classes_Fragments_Variety(xtest):
    model = def_model_all_Fragments_Variety()
    model.load_weights("_all.h5")

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict_classes(xtest, batch_size=1)
    return predict_2


def model_predict_Fragments_Variety(xtest):
    model = def_model_all_Fragments_Variety()
    model.load_weights("_all.h5")

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict(xtest, batch_size=1)
    return predict_2 


# In[ ]:


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

def def_model_all_Fragments_Variety():
    model = Sequential()
    model.add(Dense(6*67, input_dim=17, activation='sigmoid')) 
    model.add(Dense(8*17, activation='tanh'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',   metrics=["accuracy"]) 
    return model
 

# парамерт дата фрейм
def predict_All_Fragments_Variety(X_test):
    header = ['Al','Ba','Cr','Cu','Fe','Li','Mn','Na','Ni','Pb','Rb','Sr','Ti','Zn','Ca','K_','Mg']
    header_sort = ['Cabernet peel %', 'Cabernet pit %', 'Cabernet puree %', 'Riesling peel %', 'Riesling pit %', 'Riesling puree %']
    Sort = ['Cabernet peel', 'Cabernet pit', 'Cabernet puree', 'Riesling peel', 'Riesling pit', 'Riesling puree']
 
    label = LabelEncoder() 
    l_sort = label.fit(Sort)
    
    # DataFrame cохранение исходного @X_test
    df_predicting = pd.DataFrame(X_test.values, columns=header)
    X = X_test.values
    
    # Предсказание класса  
    X_predict_classes = model_predict_classes_Fragments_Variety(X)
    # преобразование номера класса в название
    X_predict_classes = l_sort.inverse_transform(X_predict_classes)
 
    # Предсказание вероятности класса  
    X_predict = model_predict_Fragments_Variety(X)
    X_predict = np.around((np.array(X_predict) * 100), decimals=-1)

    # DataFrame @X_predict
    df_predict = pd.DataFrame((np.array(X_predict)), columns=header_sort)

    # DataFrame общий - сорт, зона, исходные данные
    df = pd.concat([(pd.DataFrame(X_predict_classes, columns=['Sort'])),
                    (pd.DataFrame(X_predict, columns=header_sort)),  
                    df_predicting], axis=1)
    return df


def model_predict_classes_Fragments_Variety(xtest):
    model = def_model_all_Fragments_Variety()
    model.load_weights("_all.h5")

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict_classes(xtest, batch_size=1)
    return predict_2


def model_predict_Fragments_Variety(xtest):
    model = def_model_all_Fragments_Variety()
    model.load_weights("_all.h5")

    if xtest.size / 2 == 1:
        xtest = np.reshape(xtest, (1, -1))

    predict_2 = model.predict(xtest, batch_size=1)
    return predict_2 

