
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

#language detection code

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

languages = []

# Process each article
for _, row in data.iterrows():
    idx = row['id']
    real_text = row['real_text']
    fake_text = row['fake_text']

    # Initialize lists for each article
    real_languages = []
    fake_languages = []

    # Split into sentences using simple punctuation rule
    real_sentences = re.split(r'[.!?]', real_text)
    fake_sentences = re.split(r'[.!?]', fake_text)

    # Detect languages safely for real_text
    for sentence in real_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 10:
            try:
                lang = detect(sentence)
                real_languages.append(lang)
            except LangDetectException:
                continue

    # Detect languages safely for fake_text
    for sentence in fake_sentences:
        sentence = sentence.strip()
        if len(sentence) >= 10:
            try:
                lang = detect(sentence)
                fake_languages.append(lang)
            except LangDetectException:
                continue

    # Count language occurrences
    real_lang_counts = dict(Counter(real_languages))
    fake_lang_counts = dict(Counter(fake_languages))

   # Store results
    languages.append({
        'id': idx,
        'real_language': real_languages,
        'fake_language': fake_languages,
        'real_unique_languages': list(set(real_languages)),
        'fake_unique_languages': list(set(fake_languages)),
        'real_lang_counts': real_lang_counts,
        'fake_lang_counts': fake_lang_counts
    })

# Convert to DataFrame and save to CSV
language_df = pd.DataFrame(languages)
language_df.to_csv('language_analysis.csv', index=False)

print("✅ Saved language detection results to language_analysis.csv")
