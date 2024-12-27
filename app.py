

import streamlit as st
# import pandas as pd
# import numpy as np
import gdown

from transformers import pipeline

st.title("Fine Tuning BERT for Twitter Tweets for Multi Class Sentiment Classification")

@st.cache_resource
def loadModelFilesFromDrive(modelPth):
    url = "https://drive.google.com/drive/folders/10Q7Om1zf2lWSti8jPOx728QNrSAmA6NF?usp=drive_link"
    output_dir = modelPth
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    return output_dir


model = loadModelFilesFromDrive("bert-base-uncased-sentiment-model")
classifier = pipeline('text-classification', model= model)

text = st.text_area("Enter some text")

if st.button("Predict"):
    result = classifier(text)
    st.write(result)