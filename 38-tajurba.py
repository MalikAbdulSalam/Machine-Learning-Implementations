import joblib

# Load trained model
model_customer = joblib.load("./customer_cluster.pkl")


# New data
new_data = [[24, 26000],[80 , 80000]]

# Make prediction
prediction = model_customer.predict(new_data)
print(type(prediction))

print("cluster:", prediction[0])
print("cluster:", prediction[1])

if prediction[0] == 0:
    print("this is older customer")
if prediction[0] == 1:
    print("this is younger customer")
