import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pprint
import json

# Load CSV
data = pd.read_csv('processed_test_data.csv').fillna('')

# Load later
reference_vector = np.load('reference_vector.npy')

# Store similarity results
similarity_index = []

#store file
output_file = []

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute similarity for each row
for _, row in data.iterrows():
    idx = row['id']
    file1_text = row['file1_text']
    file2_text = row['file2_text']

       # Get embeddings for the texts
    file1_embedding = model.encode([file1_text])  
    file2_embedding = model.encode([file2_text])

    # Compute cosine similarity with reference
    sim_file1 = cosine_similarity(file1_embedding, reference_vector).item()
    sim_file2 = cosine_similarity(file2_embedding, reference_vector).item()

    # Store results (not storing embeddings here, just scores)
    similarity_index.append({
        'id': idx,
        'score_file1': sim_file1,
        'score_file2': sim_file2
    })

    # Store output (for submission)
    output_file.append({
        'id': idx,
        'real_text_id': 1 if sim_file1 > sim_file2 else 2
    })

# Convert to DataFrame
output_df = pd.DataFrame(output_file)
output_df.to_csv('output.csv', index=False)

print("✅ OUTPUT CREATED")

