from sklearn.metrics import confusion_matrix

# Actual Labels
y_test = [1,0,1,0,1]

# Predicted Labels
y_pred = [1,0,1,1,0]

cm = confusion_matrix(y_test, y_pred)

print(cm)