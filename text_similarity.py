import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load CSV
data = pd.read_csv('processed_train_data.csv').fillna('')

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pick first 5 real_texts as reference
reference_texts = data['real_text'].iloc[[5, 7, 19, 41, 51, 72, 88, 94]].tolist()
ref_embeddings = model.encode(reference_texts)
reference_vector = np.mean(ref_embeddings, axis=0).reshape(1, -1)

# Store similarity results
similarity_index = []

# Compute similarity for each row
for _, row in data.iterrows():
    idx = row['id']
    real_text = row['real_text']
    fake_text = row['fake_text']

    # Get embeddings for the texts
    real_embedding = model.encode([real_text])  # shape (1, 384)
    fake_embedding = model.encode([fake_text])

    # Compute cosine similarity with reference
    sim_real = cosine_similarity(real_embedding, reference_vector).item()
    sim_fake = cosine_similarity(fake_embedding, reference_vector).item()

    # Store results (not storing embeddings here, just scores)
    similarity_index.append({
        'id': idx,
        'similarity_real': sim_real,
        'similarity_fake': sim_fake
    })

# Convert to DataFrame
similarity_df = pd.DataFrame(similarity_index)
similarity_df.to_csv('similarity_scores.csv', index=False)

np.save('reference_vector.npy', reference_vector)

print("✅ Similarity scores saved to similarity_scores.csv")
