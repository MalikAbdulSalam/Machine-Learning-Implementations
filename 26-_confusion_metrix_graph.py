from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Actual Labels
y_test = [1,0,1,0,1]

# Predicted Labels
y_pred = [1,0,1,1,0]

cm = confusion_matrix(y_test, y_pred)

print(cm)



disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.show()