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

word_freq = []

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load your cleaned data
data = pd.read_csv('processed_train_data.csv')
data = data.fillna('')

# Storage for word frequency results
word_freqs = []

# Process each article
for _, row in data.iterrows():
    idx = row['id']
    real_text = row['real_text']
    fake_text = row['fake_text']

    # Process real and fake text using spaCy
    real_doc = nlp(real_text)
    fake_doc = nlp(fake_text)

    # Filter nouns (common or proper) and make lowercase for counting
    real_nouns = [token.text.lower() for token in real_doc if token.pos_ in ['NOUN', 'PROPN']]
    fake_nouns = [token.text.lower() for token in fake_doc if token.pos_ in ['NOUN', 'PROPN']]

    # Count frequencies
    real_counter = Counter(real_nouns)
    fake_counter = Counter(fake_nouns)

    # Get most common noun and count (default to '', 0 if empty)
    real_top = real_counter.most_common(1)[0] if real_counter else ('', 0)
    fake_top = fake_counter.most_common(1)[0] if fake_counter else ('', 0)

    # Store results
    word_freqs.append({
        'id': idx,
        'real_top_noun': real_top[0],
        'real_noun_count': real_top[1],
        'fake_top_noun': fake_top[0],
        'fake_noun_count': fake_top[1]
    })

# Convert to DataFrame and save to CSV
word_freq_df = pd.DataFrame(word_freqs)
word_freq_df.to_csv('word_freq.csv', index=False)

print("✅ Saved word frequencies to word_freq.csv")