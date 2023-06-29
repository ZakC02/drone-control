# Réécriture de la chaîne de fonctionnement avec MediaPipe
import cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np

model = keras.models.load_model("hands_version/model.h5")

def analyze(landmarks):
    array = list(landmarks.landmark)
    array_f = []
    #visibilities = [point.visibility for point in array]
    #visibilities = [1 if visibility > 0.7 else 0 for visibility in visibilities]
    #valid = sum(visibilities)
    #if valid < 22:
    #    print("invalid")
    #    #return
    #else:
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
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)


LABELS = ["neutre", "fermé", "haut", "gauche", "droite", "bas", "rond", "autre"]

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
        hands_results = hands.process(frame_rgb)
        # print(pose_results.pose_landmarks)
        # draw skeleton on the frame
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', frame)
        #computations
        pred = analyze(hands_results.multi_hand_landmarks[0])
        pred2 = model.predict(np.expand_dims(pred,axis=0),verbose=False)
        print(LABELS[np.argmax(pred2)], pred2[0][np.argmax(pred2)] * 100)
    except:
        print("not found")
    
    if cv2.waitKey(1) == ord('q'):
        break
          
cap.release()
cv2.destroyAllWindows()