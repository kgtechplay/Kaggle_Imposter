import os
import pandas as pd
import pprint
import json

# Path setup - corrected to match actual structure
base_folder = 'fake-or-real-the-impostor-hunt/train'
csv_path = 'fake-or-real-the-impostor-hunt/train.csv'


# Read CSV
df = pd.read_csv(csv_path)

# Storage for processed data
data = []

# Iterate through each row in the CSV
for _, row in df.iterrows():
    idx = row['id']
    real_id = row['real_text_id']
    
    # Build folder and file paths
    folder_name = f'article_{idx:04d}'
    folder_path = os.path.join(base_folder, folder_name)
    
    file1_path = os.path.join(folder_path, 'file_1.txt')
    file2_path = os.path.join(folder_path, 'file_2.txt')
    
    # Read files
    with open(file1_path, 'r', encoding='utf-8') as f:
        file1_text = f.read()
    with open(file2_path, 'r', encoding='utf-8') as f:
        file2_text = f.read()
    
    # Assign real and fake text
    real_text = file1_text if real_id == 1 else file2_text
    fake_text = file2_text if real_id == 1 else file1_text
    
    # Store data
    data.append({
        'id': idx,
        'real_text': real_text,
        'fake_text': fake_text
    })

# Convert to DataFrame if needed
processed_df = pd.DataFrame(data)

# create csv
processed_df.to_csv('processed_train_data.csv', index=False)