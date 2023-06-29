from djitellopy import tello
import cv2
from tensorflow import keras
import time
import mediapipe as mp
import numpy as np


model = keras.models.load_model("models/model_mains.h5")


def analyze(landmarks):
    array = list(landmarks.landmark)
    array_f = []
    # visibilities = [point.visibility for point in array]
    # visibilities = [1 if visibility > 0.7 else 0 for visibility in visibilities]
    # valid = sum(visibilities)
    # if valid < 22:
    #    print("invalid")
    #    #return
    # else:
    for point in array:
        x = point.x
        y = point.y
        z = point.z
        array_f.append(x)
        array_f.append(y)
        array_f.append(z)
    return np.asarray(array_f)


def sendToDrone(command):
    dist = 50
    if command == "neutre":
        drone.get_height()
    elif command == "haut":
        drone.move_up(dist//2)
    elif command == "bas" and drone.get_height() <= 20:
        drone.land()
    elif command == "bas" and drone.get_height() > 20:
        drone.move_down(dist//2)
    elif command == "droite":
        drone.move_left(dist)
    elif command == "gauche":
        drone.move_right(dist)
    elif command == "fermé":
        drone.move_back(dist)
    elif command == "rond":
        drone.move_forward(dist)
    elif command == "autre":
        drone.flip_right()

# initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, max_num_hands=1)


LABELS = ["neutre", "fermé", "haut",
          "gauche", "droite", "bas", "rond", "autre"]

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.takeoff()
drone.move_up(100)
drone.set_speed(30)
pipe = ["neutre" for i in range(7)]
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # read frame
    _, frame = cap.read()
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # process the frame for pose detection
        # convert rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands.process(frame)
        # print(pose_results.pose_landmarks)
        # draw skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, hands_results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', frame)
        # computations
        pred = analyze(hands_results.multi_hand_landmarks[0])
        pred2 = model.predict(np.expand_dims(
            pred, axis=0), verbose=False)
        gesture = LABELS[np.argmax(pred2)]
        print(gesture, pred2[0][np.argmax(pred2)] * 100)
        pipe.append(gesture)
        pipe.pop(0)
        if pipe.count(gesture) == len(pipe):
            sendToDrone(gesture)
        else:
            sendToDrone("neutre")
    except:
        print("not found")
        drone.get_height()
        pipe.append("neutre")
        pipe.pop(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
