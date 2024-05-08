#import webscraper
import os
import keras
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
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

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform([description])

    models = {}
    for filename in os.listdir('models'):
        model_name = filename.split('.')[0]
        models[model_name] = keras.models.load_model(os.path.join('models', filename))
        models[model_name].predict(description)
    for filename in os.listdir('models'):
        with open(filename, 'rb') as tm:
            current_pipe = pickle.load(tm)
    
        prob = new_pipe.predict_proba(description)[0]
        label = label_encoder.inverse_transform(new_pipe.predict(description))
        if prob >= 80:
            high_probability_cis_controls.append(predictions[model_name])
        elif prob >= 70:
            mid_probability_cis_controls.append(predictions[model_name])
        else:
            low_probability_cis_controls.append(predictions[model_name])
    
    return high_probability_cis_controls, mid_probability_cis_controls, low_probability_cis_controls

add_tool()