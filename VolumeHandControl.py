import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam_width = 1280
cam_height = 720

cap = cv2.VideoCapture(0)

cap.set(3, cam_width)
cap.set(4, cam_height)

current_time = 0
previous_time = 0

detector = htm.handDetector(detectionConfidence = 0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]

volume_bar = 400
volume_percentage = 0

# volume.SetMasterVolumeLevel(-5.0, None)

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)
    
    landmark_list = detector.findPosition(img, draw = False)
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[4][1], landmark_list[4][2]
        x2, y2 = landmark_list[8][1], landmark_list[8][2]
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
        
        cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        
        if length > 160:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        # Hand range: 30 - 160
        # Volume range: -74.0 - 0.0
        
        vol = np.interp(length, [30, 160], [min_volume, max_volume])
        volume_bar = np.interp(length, [30, 160], [400, 150])
        volume_percentage = np.interp(length, [30, 160], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
    
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volume_percentage)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    cv2.putText(img, f'fps: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)