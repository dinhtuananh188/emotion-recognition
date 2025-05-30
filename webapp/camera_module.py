import cv2
import requests
import numpy as np

#camera_url = "rtsp://admin:Fitdnu12@192.168.167.163:554/stream1"

def initialize_camera():
    #camera = cv2.VideoCapture(camera_url)
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise Exception("Error: Could not open webcam.")
    return camera

def capture_frame(camera):
    success, frame = camera.read()
    if not success:
        raise Exception("Error: Could not read frame from camera.")
    return frame

