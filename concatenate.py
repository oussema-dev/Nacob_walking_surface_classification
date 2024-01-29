"""This module concatenates all paricipant data and saves it to numpy and csv files"""

import os
import pandas as pd

# Create data directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

folder_path = os.getcwd()
output_file = os.path.join(folder_path, data_dir, "concatenated_data.csv")

# Initialize an empty DataFrame to store concatenated data
concatenated_data = pd.DataFrame()

# Iterate through each subfolder in the specified directory
for subdir, _, files in os.walk(folder_path):
    # Check if there are CSV files in the current subfolder
    csv_files = [file for file in files if file.endswith(".csv")]

    if csv_files:
        # Iterate through each CSV file in the current subfolder
        for csv_file in csv_files:
            # Construct the full path to the CSV file
            csv_path = os.path.join(subdir, csv_file)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
            # Concatenate the DataFrame vertically
            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

# Save the concatenated data to CSV file
concatenated_data.to_csv(output_file, index=False)
print(f"Concatenated data saved to {folder_path}")
