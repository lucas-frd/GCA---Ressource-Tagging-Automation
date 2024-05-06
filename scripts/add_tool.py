import webscraper
import os
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

high_probability_threshold = 80
mid_probability_threshold = 70
low_probability_threshold = 60

def add_tool(tool_website):
    tool_description = webscraper.get_tool_description(tool_website)

    high_probability_cis_controls = []
    mid_probability_cis_controls = []
    low_probability_cis_controls = []

    models = {}
    for filename in os.listdir('models'):
        model_name = filename.split('.')[0]
        models[model_name] = tf.keras.models.load_model(os.path.join('saved_models', filename))

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform([description])

    # Predict with each model
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict(X_tfidf)[0]
        if predictions[model_name].items()[1] >= 80:
            high_probability_cis_controls.append(predictions[model_name])
        elif predictions[model_name].items()[1] >= 70:
            mid_probability_cis_controls.append(predictions[model_name])
        else:
            low_probability_cis_controls.append(predictions[model_name])
    
    return high_probability_cis_controls, mid_probability_cis_controls, low_probability_cis_controls