# importing the required modules
import cv2
import mediapipe as mp
import time


mpDraw = mp.solutions.drawing_utils #Creating an object of utils
mpPose = mp.solutions.pose #Creating a pose object
pose = mpPose.Pose() #Intializing the pose object
cap = cv2.VideoCapture(0) #Capturing frames using the videoCapture method
pTime = 0

while True:
    success , img = cap.read() #Reading the frames
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converting them into rgb
    results = pose.process((imgRGB)) #Processes the image and return the landmark of the most prominent person detected
    if results.pose_landmarks: # if landmarks are present
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # Draws line between landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED) #Drawing circles on each landmarks
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
