import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

with open('dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
model = keras.models.load_model('model.h5')
points = dataset['arrays']
labels = dataset['labels']

X = points
y = labels


# Split the dataset into training and remaining data (which will be further split into validation and testing)
X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.4, random_state=42)

# Split the remaining data into validation and testing sets
X_test, X_val, y_test, y_val = train_test_split(X_remain, y_remain, test_size=0.4, random_state=42)



# Compute the confusion matrix
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mat = confusion_matrix(y_val, y_pred_classes)
print('Confusion Matrix:')
print(confusion_mat)
sn.heatmap(confusion_mat, cmap=sn.cubehelix_palette(as_cmap=True), fmt="g", annot=True)
plt.show()