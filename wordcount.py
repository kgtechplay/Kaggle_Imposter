import pandas as pd
import spacy
from collections import Counter
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

# Load the CSV
data = pd.read_csv('processed_train_data.csv')
print(data.isna().sum())
data = data.fillna('')
# Assuming 'data' is your DataFrame with 'id', 'real_text', and 'fake_text'
word_counts = []

for _, row in data.iterrows():
    idx = row['id']
    real_text = row['real_text']
    fake_text = row['fake_text']
    
    # Count words by splitting on whitespace
    real_word_count = len(real_text.split())
    fake_word_count = len(fake_text.split())
    
    # Store results
    word_counts.append({
        'id': idx,
        'real_word_count': real_word_count,
        'fake_word_count': fake_word_count
    })

# Convert to DataFrame
word_count_df = pd.DataFrame(word_counts)

# Print first few rows
print(word_count_df.head())

# create csv
word_count_df.to_csv('word_count.csv', index=False)