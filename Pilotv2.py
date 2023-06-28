from djitellopy import tello
import cv2
from tensorflow import keras
import time
import mediapipe as mp
import numpy as np


model = keras.models.load_model("model.h5")


def analyze(landmarks):
    array = list(landmarks.landmark)
    array = array[:25]
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
    elif command == "decollage" and drone.get_height() <= 250:
        drone.move_up(dist//2)
    elif command == "atterir" and drone.get_height() <= 50:
        drone.land()
    elif command == "atterir" and drone.get_height() > 50:
        drone.move_down(dist//2)
    elif command == "droite":
        drone.move_left(dist)
    elif command == "gauche":
        drone.move_right(dist)
    elif command == "reculer":
        drone.move_back(dist)
    elif command == "rapprocher":
        drone.move_forward(dist)
    elif command == "flip":
        drone.flip_right()
    elif command == "gear second":
        #drone.flip_forward()
        #drone.move_down(20)
        #drone.move_up(20)
        drone.flip_back()
    
    

# initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


LABELS = ["neutre", "decollage", "droite", "gauche", "atterir",
          "reculer", "rapprocher", "flip", "gear second"]

drone = tello.Tello()
drone.connect()
print(drone.get_battery())
drone.streamon()
drone.takeoff()
drone.move_up(100)
drone.set_speed(20)
pipe = ["neutre" for i in range(10)]
while True:
    img = drone.get_frame_read().frame
    try:
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # process the frame for pose detection
        # convert rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(img)
        # print(pose_results.pose_landmarks)
        # draw skeleton on the frame
        mp_drawing.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # display the frame
        cv2.imshow('Output', img)
        # computations
        pred = analyze(pose_results.pose_landmarks)
        pred2 = model.predict(np.expand_dims(
            pred, axis=0), verbose=False)
        gesture = LABELS[np.argmax(pred2)]
        print(gesture, pred2[0][np.argmax(pred2)] * 100)
        if pred2[0][np.argmax(pred2)] < 0.8:
            gesture = "neutre"
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

    # time.sleep(1 / fps)
    #
    # cv2.imshow("Live Video Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
