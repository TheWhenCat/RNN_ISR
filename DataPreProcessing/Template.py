#Data Preprocessing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, -1].values
print(Y)


#Dealing with Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer.fit(X[0:, 3:4])
X[0:, 3:4] = imputer.transform(X[0:, 3:4])
imputer.fit(X[0:, 7:13])
X[0:, 7:13] = imputer.transform(X[0:, 7:13])


#Encodcing Categorical Data
#Not working come back to this
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X = X[0:, :]
X[0:, 2] = labelencoder_X.fit_transform(X[0:, 2])
X[0:, 4] = labelencoder_X.fit_transform(X[0:, 4])
X[0:, 5] = labelencoder_X.fit_transform(X[0:, 5])
onehotencoder = OneHotEncoder(categorical_features = [2,4,5])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Splitting the dataset into Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Rescalign the Features (evening them out)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(X)

#Part 2 - ANN!

#Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

#Add the input layer and hidden layer #1
classifier.add(Dense(output_dim=6, init="uniform", activation = "relu", input_dim=11))

#Adding 2nd Hidden Layer
classifier.add(Dense(output_dim=6, init="uniform", activation = "relu"))

#Adding Output Layer
classifier.add(Dense(output_dim=1, init="uniform", activation = "sigmoid"))

#Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

#Fitting ANN to Training Set
classifier.fit(X_train, Y_train, batch_size=10, epochs=200)