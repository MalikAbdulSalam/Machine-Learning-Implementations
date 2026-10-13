import joblib

# Load trained model
model = joblib.load("regression_model_nouman.pkl")

# New data
new_data = [[100]]

# Make prediction
prediction = model.predict(new_data)

print("Predicted salary:", prediction[0])