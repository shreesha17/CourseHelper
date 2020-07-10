import pickle
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


dataset =  pd.read_csv('d:/ddd.csv')
X =  dataset.drop('level',axis=1)
Y =  dataset['level']
print(dataset.corr()['level'].sort_values())