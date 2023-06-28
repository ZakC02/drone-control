import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
from tqdm import tqdm
import random
input_path = "dataset_pictures/"
dataset = {"arrays":[], "labels":[]}

def convert(landmarks,gesture):
    #print("convert")
    array = list(landmarks.landmark)
    array = array[:25]
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
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


for gesture in tqdm(range(0,9)):
    if gesture == 5:
        continue
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
            pose_results = pose.process(frame_rgb)
            #computations
            convert(pose_results.pose_landmarks,gesture)
        except:
            print("error")
gesture = 4
for frame_path in os.listdir(input_path + str(gesture)):
    try:
        frame = cv2.imread(input_path + str(gesture) + "/" + frame_path)
        # resize the frame for portrait video
        # frame = cv2.resize(frame, (350, 600))
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # process the frame for pose detection
        frame_rgb = cv2.flip(frame_rgb, 1)
        pose_results = pose.process(frame_rgb)
        #computations
        convert(pose_results.pose_landmarks,5)
    except:
        print("error")

with open('dataset2.pickle', 'wb') as f:
    pickle.dump(dataset, f)