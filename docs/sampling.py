import cv2
import mediapipe as mp
import os
import pickle
import numpy as np
from tqdm import tqdm
from random import randint
input_path = "dataset_pictures/"

## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

for num in range(10):
    folder = randint(0,6)
    img = randint(0,900)
    frame = cv2.imread(input_path + str(folder) + "/" + str(folder)+'-'+str(img)+".jpg")
    # convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite('examples_pictures/' + str(img) + 'default.jpg',frame)
    # process the frame for pose detection
    pose_results = pose.process(frame_rgb)
    # print(pose_results.pose_landmarks)
    # draw skeleton on the frame
    mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite('examples_pictures/' + str(img)+'.jpg',frame)