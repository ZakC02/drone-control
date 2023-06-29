# Réécriture de la chaîne de fonctionnement avec MediaPipe
import cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np
import time

num_points = 75
num_classes = 9
# Define the model architecture
model = keras.Sequential([
    keras.layers.GaussianNoise(0.1, input_shape=(num_points,)),
    #keras.layers.BatchNormalization(), -> Only for SGD
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])
model.load_weights("models/model_gestes.h5")



def analyze(landmarks):
    array = list(landmarks.landmark)
    array = array[:25]
    array_f = []
    #visibilities = [point.visibility for point in array]
    #visibilities = [1 if visibility > 0.7 else 0 for visibility in visibilities]
    #valid = sum(visibilities)
    #if valid < 22:
    #    print("invalid")
    #    return
    for point in array:
        x = point.x
        y = point.y
        z = point.z
        array_f.append(x)
        array_f.append(y)
        array_f.append(z)
    return np.asarray(array_f)

## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


LABELS = ["neutre", "decollage", "droite", "gauche", "rapprocher", "reculer", "atterir", "flip", "gear second"]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame for pose detection
        pose_results = pose.process(frame_rgb)
        # print(pose_results.pose_landmarks)
        # draw skeleton on the frame
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', frame)
        #computations
        pred = analyze(pose_results.pose_landmarks)
        pred2 = model.predict(np.expand_dims(pred,axis=0),verbose=False)
        print(LABELS[np.argmax(pred2)], pred2[0][np.argmax(pred2)] * 100)
        if pred2[0][np.argmax(pred2)] < 0.8:
            print("neutre")
    except:
        print("not found")
    
    if cv2.waitKey(1) == ord('q'):
        break
          
cap.release()
cv2.destroyAllWindows()