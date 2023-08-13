# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:27:38 2023

@author: pc
"""
from typing import Annotated
from fastapi import FastAPI, Form
import uvicorn
import asyncio
from Summary import summarization
import pickle
from transformers import pipeline
app=FastAPI()

pickle_in=open('summarizer_pipeline.pkl','rb')
summarizer=pickle.load(pickle_in)


@app.post('/summary')
def summary(data:summarization):
    input_text = data.document
    summarizer = pipeline(
        task="summarization",
        model="t5-small",
        min_length=20,
        max_length=40,
        truncation=True,
        model_kwargs={"cache_dir": 'C://Users//pc//Desktop//New folder//Cache Folder'},
    ) 
    
    #input_text = "Enter the text you want to summarize: "
    summarize=summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    bullet_points = summarize.split(". ")

    for point in bullet_points:
        
        print(f"- {point}")
    return {'summary':summarize}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)