import pandas as pd
import numpy as np

from typing import Union
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
import joblib

import nltk
import re

# Training Logic
data=pd.read_csv("datasets/dataset.csv")

x=np.array(data["Message"])

cv=CountVectorizer()
cv.fit_transform(x)

# Load stored model
md = joblib.load('model/model_v1')


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search/{query}")
def read_item(query: str, q: Union[str, None] = None):
    # Predict sentiment
    mystr=query

    # Transform query to array 
    mystr=cv.transform([mystr]).toarray()
    output = md.predict((mystr))

    # Print result to console
    print(output)
    print(query)

    # return a respose to client
    return {"search": output[0], "q": q}