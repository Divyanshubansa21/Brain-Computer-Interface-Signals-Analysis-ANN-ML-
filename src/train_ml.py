import os
import pickle
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATASET_PATH = "dataset/data_preprocessed_python"

X = []
y = []

files = sorted(os.listdir(DATASET_PATH))

for file in files:

    if file.endswith(".dat"):

        with open(os.path.join(DATASET_PATH, file), "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        eeg = subject["data"][:, :32, :]

        labels = subject["labels"][:, 0]   # Valence

        # Binary Classification
        labels = (labels >= 5).astype(int)

        # Mean feature from each EEG channel
        features = np.mean(eeg, axis=2)

        X.append(features)
        y.append(labels)

X = np.vstack(X)
y = np.hstack(y)

print("Feature Shape :", X.shape)
print("Label Shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, pred))

print("\nClassification Report\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, pred))


# Save trained model

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel Saved Successfully!")