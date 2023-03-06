import math
import cv2
import time
import mediapipe as mp

class poseDetector():
    #initializing
    def __init__(self, mode=False, model_complexity=1, upBody=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        """
        This function takes an image process it saves the landmarks in the var self.results and if landmarks are
        detected then draws lines between them and then returns this new image
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Changing the color
        self.results = self.pose.process(imgRGB) #processing the image and saving the landmarks in `self.results`
        if self.results.pose_landmarks: #if lm present
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS) #Draws the connections between landmarks
        return img

    def findPosition(self, img, draw = True):
        """
        takes the processed image as input and saves the pixcek values of landnarks in the list and draws circles on
        them if draw = true
        """
        self.lmList = [] #Initializing the lmList
        if self.results.pose_landmarks: #if landmarks are present then save the value of thier pixcels in lmList
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw: #if `draw == True` then draws the circles on those landmarks
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()

