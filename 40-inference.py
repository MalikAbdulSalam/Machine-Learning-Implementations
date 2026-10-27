import joblib

# Load trained model
model_customer = joblib.load("./customer_project_trained_model.pkl")


# New data
new_data = [[2, 3, 12669,9656,7561,214,2674,1338]]

# Make prediction
prediction = model_customer.predict(new_data)


print("cluster:", prediction[0])