"""This module calculates statistical features per surface segment"""

import pandas as pd
import numpy as np
import os
from scipy import integrate

data_file = os.path.join("data", "concatenated_data.csv")
df = pd.read_csv(data_file)

df.fillna(0, inplace=True)

# List of features to calculate
features = ["mean", "min", "max", "std", "iqr", "mad", "auc", "sauc"]

# Initialize an empty DataFrame to store the final features
final_features_df = pd.DataFrame()

# Initialize variables to track the current segment
current_class = None
segment_rows = []

print("Calculating statistical features...")

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    if current_class is None:
        # For the first row, initialize the current class and add it to the segment
        current_class = row["class"]
        segment_rows.append(row)
    elif current_class == row["class"]:
        # If the class is the same as the current segment, add it to the segment
        segment_rows.append(row)
    else:
        # If the class changes, calculate features for the current segment
        segment_df = pd.DataFrame(segment_rows)
        segment_features = {}

        # Include additional columns in the 'segment_features' dictionary
        additional_columns = [
            "Participant",
            "Sex",
            "Unable to walk on irregular surface",
            "Musculoskeletal or neurological disease",
            "Age",
            "Weight(kg)",
            "Height(cm)",
            "Leg Length(cm)",
        ]
        try:
            for feature in df.columns[
                0:24
            ]:  # This excludes the additional columns and 'class' column
                feature_values = segment_df[feature]
                segment_features[feature + "_mean"] = feature_values.mean()
                segment_features[feature + "_min"] = feature_values.min()
                segment_features[feature + "_max"] = feature_values.max()
                segment_features[feature + "_std"] = feature_values.std()
                segment_features[feature + "_iqr"] = np.percentile(
                    feature_values, 75
                ) - np.percentile(feature_values, 25)
                segment_features[feature + "_mad"] = np.median(
                    np.abs(feature_values - np.median(feature_values))
                )
                segment_features[feature + "_auc"] = integrate.simps(feature_values)
                segment_features[feature + "_sauc"] = integrate.simps(
                    np.abs(feature_values)
                )
            # Add the additional (anthropometrics) columns to the segment
            for col in additional_columns:
                segment_features[col] = segment_df[col].iloc[
                    0
                ]  # Assuming all values in the segment are the same

            # Add the class label to the columns
            segment_features["class"] = current_class

            # Append the data to the final DataFrame
            final_features_df = pd.concat(
                [final_features_df, pd.DataFrame([segment_features])], ignore_index=True
            )

            # Reset variables for the next segment
            current_class = row["class"]
            segment_rows = [row]
        except:
            # Show the participant where there is an error
            print(df.at[index, "Participant"])

# Calculate features for the last segment
segment_df = pd.DataFrame(segment_rows)
segment_features = {}

for feature in df.columns[0:24]:
    feature_values = segment_df[feature]
    segment_features[feature + "_mean"] = feature_values.mean()
    segment_features[feature + "_min"] = feature_values.min()
    segment_features[feature + "_max"] = feature_values.max()
    segment_features[feature + "_std"] = feature_values.std()
    segment_features[feature + "_iqr"] = np.percentile(
        feature_values, 75
    ) - np.percentile(feature_values, 25)
    segment_features[feature + "_mad"] = np.median(
        np.abs(feature_values - np.median(feature_values))
    )
    segment_features[feature + "_auc"] = integrate.simps(feature_values)
    segment_features[feature + "_sauc"] = integrate.simps(np.abs(feature_values))

for col in additional_columns:
    segment_features[col] = segment_df[col].iloc[0]

segment_features["class"] = current_class

final_features_df = pd.concat(
    [final_features_df, pd.DataFrame([segment_features])], ignore_index=True
)
# Save the final features DataFrame to a new CSV file
final_features_df.to_csv(os.path.join("data", "statistical_features.csv"), index=False)
print("Statistical features calculated successfully")
