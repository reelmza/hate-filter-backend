import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import joblib

import nltk
import re

# Training Logic
data=pd.read_csv("datasets/dataset.csv")

# Convert columns to unicode string
x=np.array(data["Message"])
y=np.array(data["Category"])

cv=CountVectorizer()
X = cv.fit_transform(x)

# Train model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

# Persit model to a file
joblib.dump(model, 'model/model_v1')
