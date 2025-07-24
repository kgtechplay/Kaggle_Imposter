import pandas as pd
import spacy
from collections import Counter
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

# Load the CSVs
data1 = pd.read_csv('processed_train_data.csv')
data2 = pd.read_csv('word_count.csv')
# Only columns 6 and 7 (real_lang_counts, fake_lang_counts) from data3
cols_to_use = ['id', 'real_lang_counts', 'fake_lang_counts']
data3 = pd.read_csv('language_analysis.csv', usecols=cols_to_use)
data4 = pd.read_csv('word_freq.csv')

# Merge the CSVs
merged_data = pd.merge(data1, data2, on='id', how='left')
merged_data = pd.merge(merged_data, data3, on='id', how='left')
merged_data = pd.merge(merged_data, data4, on='id', how='left')

# merged_data = merged_data.fillna('')

print(merged_data.loc[80:86])

# Save the merged CSV
merged_data.to_csv('merged_train_data.csv', index=False)

# Print the merged row for id 83 as a table with header
row_83 = merged_data[merged_data['id'] == 83]
if not row_83.empty:
    print('\nMerged data for id=83:')
    print(row_83.to_string(index=False))
else:
    print('No row found with id=83')