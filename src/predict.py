import joblib
import numpy as np

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

print("Model Loaded Successfully!")

# Dummy input (32 EEG features)
sample = np.random.rand(1, 32)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction: Positive Emotion")
else:
    print("Prediction: Negative Emotion")