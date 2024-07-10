import cv2
from deepface import DeepFace

#open webcam
facecascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")
#read face  
while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame)#analyse face
    #draw rectangle
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font=cv2.FONT_HERSHEY_SIMPLEX#font type
    #write these things
    cv2.putText(frame,result[0]['dominant_gender'],(0,100),font,2,(0,0,255),3,cv2.LINE_4);

    
    cv2.putText(frame,str(result[0]['age']),(0,200),font,2,(255,0,0),3,cv2.LINE_4);
    cv2.imshow('Demo video',frame)
    
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()