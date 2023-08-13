# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 17:27:22 2023

@author: pc
"""

import pickle
from transformers import pipeline
import streamlit as st

def main():
    st.title("Text Summerization")

summarizer = pipeline(
    task="summarization",
    model="t5-small",
    min_length=20,
    max_length=40,
    truncation=True,
    model_kwargs={"cache_dir": 'C://Users//pc//Desktop//New folder//Cache Folder'},
) 


with open('summarizer_pipeline.pkl', 'wb') as f:
    pickle.dump(summarizer, f)
    

with open('summarizer_pipeline.pkl', 'rb') as f:
    loaded_summarizer = pickle.load(f)
    

input_text = st.text_area("Enter the text you want to summarize: ",height=200)

# Generate the summary
if st.button("Summarize"):
    if input_text:

            summary = loaded_summarizer(input_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            
            st.subheader("Patient-Summary: ")
            
            bullet_points = summary.split(". ")

            for point in bullet_points:
    
                  print(f"- {point}")
    else:
         st.warning("Please enter text  to summarize: ")

# print()
# print("Patient-Summary:", summary)
# print()
# print("Doctor-Summary",summary)
