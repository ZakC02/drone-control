import socket,time, cv2
import mediapipe as mp
from tensorflow import keras
import numpy as np
 

tello_ip = '192.168.10.1'
tello_port = 8889
tello_address = (tello_ip, tello_port)
host ='192.168.10.2'

model = keras.models.load_model("model.h5")

def analyze(landmarks):
    array = list(landmarks.landmark)
    array = array[:25]
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
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


LABELS = ["neutre", "decollage", "droite", "gauche", "atterir", "reculer", "rapprocher", "flip", "gear second", "fortnite"]


#mypc_address = (host, port)

local_ip=''
socket = socket.socket (socket.AF_INET, socket.SOCK_DGRAM)
socket.bind ((local_ip,8889))
socket.sendto ('command'.encode (' utf-8 '), tello_address)
socket.sendto ('streamon'.encode (' utf-8 '), tello_address)
#socket.sendto ('battery?'.encode (' utf-8 '), tello_address)

print ("Start streaming")

capture = cv2.VideoCapture ('udp://0.0.0.0:11111',cv2.CAP_FFMPEG)
if not capture.isOpened():
    capture.open('udp://0.0.0.0:11111')

def sendToDrone(command):
    labels = {'neutre' : '', 'decollage' : 'takeoff', 'droite' : 'right 50', 'gauche' : 'left 50', 'atterir' : 'land', 'reculer' : 'back 50', 'rapprocher' : 'forward 50', 'flip' : 'flip r', 'gear second' : 'speed 20','fortnite' : 'flip b'}
    socket.sendto(labels[command].encode ('utf-8'), tello_address)

time.sleep(5)

command = "decollage"
sendToDrone(command)

frame_rate = 10
prev = 0
while True:
    #Récupérer la sortie vidéo
    time_elapsed = time.time() - prev
    ret, frame = capture.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()
        #print(ret)
        if(ret):
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
                command = LABELS[np.argmax(pred2)]
                sendToDrone(command)
            except:
                print("not found")
            #cv2.imshow('frame', frame)  
    if cv2.waitKey (1)&0xFF == ord ('q'):
        sendToDrone("atterir")
        break

    

capture.release ()
cv2.destroyAllWindows ()
socket.sendto ('streamoff'.encode (' utf-8 '), tello_address)
