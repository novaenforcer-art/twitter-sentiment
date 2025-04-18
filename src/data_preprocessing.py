import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

train_data = pd.read_csv('data/raw/train.csv')
test_data = pd.read_csv('data/raw/test.csv')
#         print(f"error. Failed to save data to {data_path}")

nltk.download('stopwords')
nltk.download('wordnet')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    return' '.join([word for word in text.split() if word not in stopwords.words('english')])

def removing_numbers(text):
    return''.join([word for word in text.split() if not word.isdigit()])

def lower_case(text):
    return' '.join([word.lower() for word in text.split()])

def removing_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_urls(text):
    return re.sub(r'http\S+', '', text)

def remove_small_sentences(text):
    for i in range(len(df)):
        if len(df['content'][i].split()) < 3:
            df.drop(i, inplace=True)

def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(remove_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

data_path = os.path.join("data", "processed")

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'))
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'))

