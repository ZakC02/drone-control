from djitellopy import tello

drone = tello.Tello()
drone.connect()
print(drone.get_battery())