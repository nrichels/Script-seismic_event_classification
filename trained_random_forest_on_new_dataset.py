import joblib
import pandas as pd

# Load model and encoder
clf = joblib.load("random_forest_model.pkl")
le = joblib.load("label_encoder.pkl")

# Load new data to classify (must have the same feature columns)
new_data = pd.read_csv("2025_features.csv")  # No 'class_name' column

# Predict
predictions = clf.predict(new_data)

# Decode predicted labels
predicted_labels = le.inverse_transform(predictions)
print(predicted_labels)

# Save predictions to csv
output = pd.DataFrame({
    "event_time": new_data["event_time"],
    "predicted_event": predicted_labels
})
output.to_csv("20205_predictions.csv", index=False)