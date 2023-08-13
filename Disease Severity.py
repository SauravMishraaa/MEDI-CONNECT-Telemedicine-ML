# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 12:25:14 2023

@author: pc
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


disease_pred=pd.read_csv('pred_status.csv')

disease_pred.head()

disease_pred['Status']=disease_pred['Status'].map({'medium severity':1,'medium severity:':1,'low severity':0,'Low severity':0,'low severity\r':0,'high severity':2})

disease_pred.drop('Description',axis=1,inplace=True)

X=disease_pred['Doctor']+''+disease_pred['Patient']

y=disease_pred['Status']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


X_train = X_train.fillna('')
X_test = X_test.fillna('')


tfidf_vectorizer=TfidfVectorizer(max_features=1000)
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train.astype(str))
X_test_tfidf=tfidf_vectorizer.transform(X_test.astype(str))


svm_classifier=SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf,y_train)

predictions=svm_classifier.predict(X_test_tfidf)

text="Hi,I m XXXX,I am ulcer patient ,I did my endoscopy last yr in July and took medication till Feb after that I left the medicine and started taking DGL and probiotic for a month after that I did endoscopy again and found out that only superficial scars are left ..but after endoscopy I really felt great pain the stomach again and mild pain after few days.my doc told me not to take high protein diets.but I ignored and took it and now I sometimes feel Pricky feeling in my stomach .I m worried whether my ulcer is coming back"

input_text=tfidf_vectorizer.transform([text])

prediction=svm_classifier.predict(input_text)

print(f"Prediction :{prediction[0]}")
