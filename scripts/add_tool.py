#import webscraper
import os
import keras
import pickle
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

high_probability_threshold = 80
mid_probability_threshold = 70
low_probability_threshold = 60

def add_tool():
    #tool_description = webscraper.get_tool_description(tool_website)
    description = ["A password manager"]

    high_probability_cis_controls = []
    mid_probability_cis_controls = []
    low_probability_cis_controls = []

    for filename in os.listdir('models'):
        with open(f'models/{filename}', 'rb') as tm:
            new_pipe = pickle.load(tm)
    
        prob = max(new_pipe.predict_proba(description)[0])
        print(prob)
        label = new_pipe.predict(description)[0]
        if prob >= 0.80:
            high_probability_cis_controls.append(label)
        elif prob >= 0.70:
            mid_probability_cis_controls.append(label)
        else:
            low_probability_cis_controls.append(label)
    
    print(f'high: {high_probability_cis_controls}, mid: {mid_probability_cis_controls}, low: {low_probability_cis_controls}')

add_tool()