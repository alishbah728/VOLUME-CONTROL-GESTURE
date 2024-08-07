import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelComplexity = 1, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
    
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
    
        return img
    
    def findPosition(self, img, handNumber = 0, draw = True):
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            
            for id, landmark in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                landmark_list.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        
        return landmark_list

def main():
    cap = cv2.VideoCapture(0)
    
    prev_time = 0
    curr_time = 0
    
    detector = handDetector()
    
    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)
        
        # landmark_list = detector.findPosition(img)
        # if len(landmark_list) != 0:
        #     print(landmark_list[4])
    
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, f'fps: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()