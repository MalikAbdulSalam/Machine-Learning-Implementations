from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


    
# Actual Labels
y_test = [1,0,1,0,1]

# Predicted Labels
y_pred = [1,0,1,1,0]

# print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



#plot confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)
disp.plot(cmap="Blues")
plt.show()


# print all other reports
print(classification_report(y_test, y_pred))