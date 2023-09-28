import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# hand tracking module from mediapipe # compulsory
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions. drawing_utils

#Draw FPS
pTime = 0
cTime = 0


while True:
    success , img = cap.read()

    #send rgb image
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Treack hand marks
    if results.multi_hand_landmarks:
         for handLns in results.multi_hand_landmarks:
             for id , ln in enumerate(handLns.landmark):
                 h , w ,c = img.shape
                 cx , cy = int(ln.x*w) , int(ln.y*h)
                 print(id , cx , cy)
                 if id == 0:
                     cv2.circle(img ,(cx,cy) , 25  , (255,0,255) , cv2.FILLED)

         mpDraw.draw_landmarks(img , handLns,mpHands.HAND_CONNECTIONS)

    #to show FPS
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img , str(int(fps)) , (10,70) , cv2.FONT_HERSHEY_SIMPLEX , 3 , (255,0,255) , 3)

    cv2.imshow("Image" , img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break