from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam,SGD
import matplotlib as plt
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import load_model
from keras.models import load_model
import pickle

# To Diable Warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


dataset = pd.read_csv('D:\ddf.csv')
X = dataset.drop('level',axis=1)
Y = dataset['level']

dataset2 =  pd.read_csv('d:/dd.csv')
XX = dataset2.drop('level',axis=1)
YY =  dataset2['level']

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

encoder2 = LabelEncoder()
encoder2.fit(YY)
encoded_YY = encoder.transform(YY)
dummy_yy = np_utils.to_categorical(encoded_YY)

def baseline_model():
	model = Sequential()
	model.add(Dense(22, input_dim=11, activation='relu'))
	model.add(Dense(11, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model


filename = 'Saved_Model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.predict(XX)
print("\n\nThe Predicted level of the User is ")
print(encoder.inverse_transform(result))

print("\n\nThe  Actual level of the User is ")
original =  encoder.inverse_transform(np.argmax(dummy_yy,axis=1))
print(original)