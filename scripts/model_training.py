import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def train_model(control1, control2):

    df = pd.read_csv('/Users/lucasfernandes/Desktop/GCA - Ressource Tagging Automation/data/tools/Consolidated_Tools.csv')

    sentences = {}
    for index, row in df.iterrows():
        tool = row['Long Description']
        if tool is None:
            continue
        controls = [control.strip() for control in str(row['CIS v8 Control Area']).split(",")]
        if controls == ['nan']:
            continue
        sentences[tool] = controls

    # Define the CIS controls you're interested in
    cis_controls_of_interest = [str(control1), str(control2)]
    filtered_tools = []

    # Iterate through each sentence and compare the CIS controls
    for sentence, controls in sentences.items():
        # Preprocess controls
        preprocessed_controls = [preprocess_control(control) for control in controls]
        if any(control in cis_controls_of_interest for control in preprocessed_controls):
            mapped_controls = list(set([preprocess_control(control) for control in controls if preprocess_control(control) in cis_controls_of_interest]))
            print(f"{sentence}: {', '.join(controls)} => {', '.join(mapped_controls)}")
            filtered_tools.append({'Long Description': sentence, 'CIS v8 Control Area': ', '.join(mapped_controls)})

    df = pd.DataFrame(filtered_tools)
    # Remove rows with NaN values only from column 'A'
    df = df.dropna(subset=['Long Description'])
    print(df["CIS v8 Control Area"].value_counts())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df["Long Description"], df['CIS v8 Control Area'], test_size=0.2, random_state=42)

    model = MultinomialNB()
    tf_vect = TfidfVectorizer()
    pipe = Pipeline([("vectorizer", tf_vect), ("classifier", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    with open(f'models/classifier_{control1}_{control2}', 'wb') as picklefile:
        pickle.dump(pipe, picklefile)

# Function to preprocess control numbers
def preprocess_control(control):
    # If the control number starts with a number followed by a dot, extract only the integer part
    if "." in control:
        return control.split(".")[0]
    return control


train_model(17, 18)

