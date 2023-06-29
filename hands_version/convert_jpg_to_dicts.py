import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
from tqdm import tqdm
input_path = "hands_version/dataset_pictures/"
dataset = {"arrays":[], "labels":[]}

def convert(landmarks,gesture):
    #print("convert")
    array = list(landmarks.landmark)
    array_f = []
    for point in array:
        x = point.x
        y = point.y
        z = point.z
        array_f.append(x)
        array_f.append(y)
        array_f.append(z)
    dataset["arrays"].append(np.asarray(array_f))
    dataset["labels"].append(gesture)


## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,max_num_hands=2)



for gesture in tqdm(range(0,8)):
    #print a loading bar
    #print("Loading gesture " + str(gesture) + "...")
    for frame_path in os.listdir(input_path + str(gesture)):
        try:
            frame = cv2.imread(input_path + str(gesture) + "/" + frame_path)
            # resize the frame for portrait video
            # frame = cv2.resize(frame, (350, 600))
            # convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process the frame for pose detection
            hands_results = hands.process(frame_rgb)
            #computations
            if hands_results.multi_hand_landmarks:
                convert(hands_results.multi_hand_landmarks[0],gesture)
        except:
            print("error")


with open('hands_version/dataset.pickle', 'wb') as f:
    pickle.dump(dataset, f)