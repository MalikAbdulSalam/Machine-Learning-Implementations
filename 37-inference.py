import joblib

# Load trained model
model_customer = joblib.load("./customer_cluster.pkl")


# New data
new_data = [[24, 26000]]

# Make prediction
prediction = model_customer.predict(new_data)


print("cluster:", prediction[0])