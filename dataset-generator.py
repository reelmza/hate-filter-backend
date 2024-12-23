import pandas as pd
import numpy as np

import nltk
import re

# Download stopwords
nltk.download('stopwords')

# Clean, Remove Stopwords & Format Dataset
from nltk.corpus import stopwords
stopwords=set(stopwords.words("english"))
stemmer=nltk.SnowballStemmer("english")

# Regular expression to clean data
def clean (text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('<.?>+','', text)
    text = re.sub(r'[^\w\s]','', text) 
    text = re.sub('\n', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text =" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    return text

# Read unclean dataset
data=pd.read_csv("datasets/raw_dataset.csv")

# Create new row with grading for each data
data["Category"]=data["Label"].map({0:"Normal", 1:'Hate'})

# Extract only Message & Category from unclean dataset
data = data[["Message", "Category"]]

# Clean Message column 
data["Message"] = data["Message"].apply(clean)

# Write dataset to an excel file
data.to_csv('datasets/dataset.csv', index=False)
print(data.head())