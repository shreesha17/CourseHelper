from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam,SGD
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pickle


# To Diable Warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset = pd.read_csv('D:\ddd.csv')
X = dataset.drop('level',axis=1)
Y = dataset['level']

dataset2 =  pd.read_csv('d:/dd2.csv')
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
	model.add(Dense(22, input_dim=6, activation='relu'))
	model.add(Dense(11, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=90, batch_size=10, verbose=1)


#X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1)
#estimator.fit(X_train, Y_train)
estimator.fit(X, dummy_y)
predictions = estimator.predict(XX)

predictions  = encoder.inverse_transform(predictions)
original =  encoder.inverse_transform(np.argmax(dummy_yy,axis=1))

#print (np.argmax(Y_test,axis=1))
print(predictions)
print(original)

results=accuracy_score(original,predictions)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#encoder.inverse_transform(predictions)
#y_test = encoder.inverse_transform(Y_test)

#print(predictions)
#accuracy_score(y_test,y_pred)
#print(accuracy_score)
#print(encoder.inverse_transform(predictions))








#estimator.fit(X, dummy_y)
#predictions  =  estimator.predict(XX)
#print(encoder.inverse_transform(predictions))
filename = 'Saved_Model.sav'
pickle.dump(estimator, open(filename, 'wb'))

#kfold = KFold(n_splits=3, shuffle=True, random_state=7)
#results = cross_val_score(estimator, X, dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#model.fit(X, dummy_y, epochs = 80, batch_size=10)

#model  =  baseline_model()