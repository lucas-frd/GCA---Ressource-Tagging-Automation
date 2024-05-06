import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

def train_model(data, control1, control2):
    # Read the data
    df = pd.read_csv('data/filtered_descriptions.csv')

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['Long Description'])

    # Convert string labels to numerical format
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df['CIS v8 Control Area'])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels_encoded, test_size=0.2, random_state=42)

    # Define neural network architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    # Make predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}, Recall: {recall}, F1-score: {f1}')

    save_dir = 'models'
    model.save(f'models/model_CIS{control1}_CIS{control2}.h5')



# Read the CSV file into a DataFrame
df = pd.read_csv('data/WIP - CIS Control Mappings Updated - cleaned_consolidated_tools_2023_12_06.xlsx - Sheet1.csv', usecols=['Name', 'CIS v8 Control Area', 'Long Description'])

df1 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 1")]
df2 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 2")]
df3 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 3")]
df4 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 4")]
df5 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 5")]
df6 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 6")]
df7 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 7")]
df8 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 8")]
df9 = df[df["CIS v8 Control Area"].astype(str).str.contains(" 9")]
df10 = df[df["CIS v8 Control Area"].astype(str).str.contains("10")]
df11 = df[df["CIS v8 Control Area"].astype(str).str.contains("11")]
df12 = df[df["CIS v8 Control Area"].astype(str).str.contains("12")]
df13 = df[df["CIS v8 Control Area"].astype(str).str.contains("13")]
df14 = df[df["CIS v8 Control Area"].astype(str).str.contains("14")]
df15 = df[df["CIS v8 Control Area"].astype(str).str.contains("15")]
df16 = df[df["CIS v8 Control Area"].astype(str).str.contains("16")]
df17 = df[df["CIS v8 Control Area"].astype(str).str.contains("17")]
df18 = df[df["CIS v8 Control Area"].astype(str).str.contains("18")]

df1.loc[:, "CIS v8 Control Area"] = 1
df2.loc[:, "CIS v8 Control Area"] = 2
df3.loc[:, "CIS v8 Control Area"] = 3
df4.loc[:, "CIS v8 Control Area"] = 4
df5.loc[:, "CIS v8 Control Area"] = 5
df6.loc[:, "CIS v8 Control Area"] = 6
df7.loc[:, "CIS v8 Control Area"] = 7
df8.loc[:, "CIS v8 Control Area"] = 8
df9.loc[:, "CIS v8 Control Area"] = 9
df10.loc[:, "CIS v8 Control Area"] = 10
df11.loc[:, "CIS v8 Control Area"] = 11
df12.loc[:, "CIS v8 Control Area"] = 12
df13.loc[:, "CIS v8 Control Area"] = 13
df14.loc[:, "CIS v8 Control Area"] = 14
df15.loc[:, "CIS v8 Control Area"] = 15
df16.loc[:, "CIS v8 Control Area"] = 16
df17.loc[:, "CIS v8 Control Area"] = 17
df18.loc[:, "CIS v8 Control Area"] = 18

# Concatenate the filtered DataFrames
result_df = pd.concat([df1, df3])

# Drop duplicates based on the "Long Description" column to keep only unique descriptions
result_df = result_df.drop_duplicates(subset="Long Description")

# Reset index
result_df.reset_index(drop=True, inplace=True)

print(result_df["CIS v8 Control Area"].value_counts())

# Save the result DataFrame to a CSV file
result_df.to_csv("data/filtered_descriptions.csv", index=False)

train_model("data/filtered_descriptions.csv", 1, 3)

