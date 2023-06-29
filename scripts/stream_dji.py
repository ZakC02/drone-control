from djitellopy import tello
import cv2

drone = tello.Tello()
drone.connect()

drone.streamon()

while True:
    img = drone.get_frame_read().frame
    cv2.imshow("Live Video Feed", img)
    cv2.waitKey(1)