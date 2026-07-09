import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "dataset/data_preprocessed_python"


def load_all_subjects():

    X = []
    y = []

    files = sorted(os.listdir(DATASET_PATH))

    for file in files:

        if file.endswith(".dat"):

            with open(os.path.join(DATASET_PATH, file), "rb") as f:

                subject = pickle.load(f, encoding="latin1")

            eeg = subject["data"][:, :32, :]     # First 32 EEG channels

            labels = subject["labels"]

            X.append(eeg)
            y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    return X, y


X, y = load_all_subjects()

print("EEG Shape :", X.shape)
print("Labels Shape :", y.shape)


# Flatten EEG

X = X.reshape(X.shape[0], -1)

print("Flatten Shape :", X.shape)


# Standardization

scaler = StandardScaler()

X = scaler.fit_transform(X)

print("Preprocessing Completed Successfully")