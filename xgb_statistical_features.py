"""This module trains a feature-based XGBoost model
It uses feature reduction methods and a subject-wise splitting approach

Note: If classes do not start with 0, uncomment the lines 123 and 124
"""

import pandas as pd
import os
import random
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.feature_selection import VarianceThreshold


def reduce_features(train_set, features, variance_threshold=0.01, max_corr=0.80):
    """this is the feature reduction method

    Parameters:
    train_set (dataframe): train set used to reduce the features
    features (list): initial feature names
    variance_threshold (float): variance threshold used to identify quasi-constant features
    max_corr (float): threshold to keep non-correlated features

    Returns:
    reduced_features (list): reduced feature names
    """

    print("reducing features...")

    X_train = train_set[features]
    # Removing Constant features using variance threshold
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(X_train)
    # Get the names of the constant features
    constant_columns = [
        column
        for column in X_train.columns
        if column not in X_train.columns[constant_filter.get_support()]
    ]
    # Drop constant features
    for constant_column in constant_columns:
        print("dropping constant feature {}".format(constant_column))
    X_train.drop(labels=constant_columns, axis=1, inplace=True)

    # Drop quasi-constant features
    qconstant_filter = VarianceThreshold(threshold=variance_threshold)
    qconstant_filter.fit(X_train)
    # Get the names of the qconstant features
    qconstant_columns = [
        column
        for column in X_train.columns
        if column not in X_train.columns[qconstant_filter.get_support()]
    ]
    # Drop qconstant features
    for qconstant_column in qconstant_columns:
        print("dropping quasi-constant feature {}".format(qconstant_column))
    X_train.drop(labels=qconstant_columns, axis=1, inplace=True)

    # Drop duplicate features
    transpose = X_train.T
    unique_features = transpose.drop_duplicates(keep="first").T.columns
    duplicate_features = [x for x in X_train.columns if x not in unique_features]
    for duplicate_feature in duplicate_features:
        print("dropping duplicate feature {}".format(duplicate_feature))

    # Drop correlated features
    X_train = train_set[[*unique_features]]
    correlated_features = set()
    correlation_matrix = X_train.corr()
    # Add the columns with a correlation value of max_corr to the correlated_features set
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > max_corr:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    for correlated_feature in correlated_features:
        print(
            "dropping feature correlated greater than {} {}".format(
                max_corr, correlated_feature
            )
        )
    X_train.drop(labels=correlated_features, axis=1, inplace=True)
    reduced_features = X_train.columns
    return reduced_features


df = pd.read_csv(os.path.join("data", "statistical_features.csv"))

# Replace NaNs with the mean value of each column
df.fillna(df.mean(), inplace=True)

features = df.columns[:192].tolist()

features = reduce_features(df, features, variance_threshold=0.1, max_corr=0.8)

y = df["class"]
# Use LabelEncoder to convert string classes to numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

participants = df["Participant"].unique()

random.seed(2)
random.shuffle(participants)

# Assign a certain percentage of participants to the training set and the remaining to the test set
train_percentage = 0.8
num_train = int(train_percentage * len(participants))

train_participants = participants[:num_train]
test_participants = participants[num_train:]

# Use the participant list to filter the dataset into training and testing sets
train_set = df[df["Participant"].isin(train_participants)]
test_set = df[df["Participant"].isin(test_participants)]

# Split the data into features (X) and labels (y)
X_train, y_train = train_set[features], train_set["class"]
X_test, y_test = test_set[features], test_set["class"]

####### only if classes start from 1 instead of 0 #######
# y_train = y_train - 1
# y_test = y_test - 1

# Train XGBoost classifier
model = XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the accuracy score
print("Accuracy score =", accuracy_score(y_test, y_pred))
# Print prediction metrics
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
