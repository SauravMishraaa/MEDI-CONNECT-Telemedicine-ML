# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:58:18 2023

@author: pc
"""

from fastapi import FastAPI
import uvicorn
from Predictions import predict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app=FastAPI()

@app.get('/')
def index():
    return{'Hello':'message'}

@app.post('/predict')
def disease_severity(data:predict):
    input_text=data.text
    tfidf_vectorizer=TfidfVectorizer(max_features=1000)
    input_texts = [input_text]
    input = tfidf_vectorizer.fit_transform(input_texts)
    # input=tfidf_vectorizer.transform(input_text.astype(str))
    svm_classifier=SVC(kernel='linear')
    target_labels = [0,1,2] 
    svm_classifier.fit(input,target_labels)
    prediction=svm_classifier.predict(input)
    input=tfidf_vectorizer.transform([input_text])

    print(f"Prediction :{prediction[0]}")
    return{'Prediction':prediction}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
