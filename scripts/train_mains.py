import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import os
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
#from tensorflow.keras.callbacks import TensorBoard

# Constants
num_points = 63
num_classes = 8
#Import dataset
with open('hands_version/dataset.pickle', 'rb') as f:
    dataset = pickle.load(f)
values = dataset["arrays"]
labels = dataset["labels"]

X = np.asarray(values)
Y = np.asarray(labels)


#Split dataset with scikit-learn
from sklearn.model_selection import train_test_split


# Séparation en train set (70%) et test set (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Séparation du test set en validation set (50%) et test set (50%)
X_val, X_test, y_val, y_test = train_test_split(
    X_test,
    y_test,
    test_size=0.5,
    random_state=42
)

# Define the log directory for TensorBoard
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define the model architecture
model = keras.Sequential([
    tf.keras.layers.GaussianNoise(0.1, seed=None, input_shape=(num_points,)),
    #keras.layers.BatchNormalization(), -> Only for SGD
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model with the loss
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
#optimizer = keras.optimizers.SGD(learning_rate=0.01, weight_decay=0.0001)

model.compile(optimizer=optimizer,
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])



# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),callbacks=[tensorboard_callback])

# Evaluation of the model
loss, accuracy = model.evaluate(X_val, y_val)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Compute the confusion matrix
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mat = confusion_matrix(y_val, y_pred_classes)
print('Confusion Matrix:')
print(confusion_mat)
sn.heatmap(confusion_mat, cmap=sn.cubehelix_palette(as_cmap=True), fmt="g", annot=True)
plt.show()

# Save the model
model.save('hands_version/model.h5')