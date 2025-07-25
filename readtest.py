import os
import pandas as pd
import pprint
import json

# Path setup - corrected to match actual structure
base_folder = 'fake-or-real-the-impostor-hunt/test'



# Storage for processed data
data = []

# Iterate through all the folders
for idx in range(1068):
        
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
    
   
    # Store data
    data.append({
        'id': idx,
        'file1_text': file1_text,
        'file2_text': file2_text
    })

# Convert to DataFrame if needed
processed_df = pd.DataFrame(data)

# create csv
processed_df.to_csv('processed_test_data.csv', index=False)