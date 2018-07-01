import numpy as np
import cv2

path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(path)

cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    cv2.imshow("Web-Camera",frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(gray,scaleFactor = 1.05, minNeighbors = 10, minSize = (40,40))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Face-Detection",frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
