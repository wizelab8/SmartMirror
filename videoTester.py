import os
import cv2
import numpy as np
import faceRecognition as fr
import imutils

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.createLBPHFaceRecognizer()
face_recognizer.load('trainingData.yml')#Load saved training data

name = {0 : "elon",1 : "jeevak"}


cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    test_img=imutils.resize(test_img,width=300)
    faces_detected,gray_img=fr.faceDetection(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    resized_img = cv2.resize(test_img, (200, 200))
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        font = cv2.FONT_HERSHEY_SIMPLEX
        resized_img = cv2.resize(test_img, (200, 200))
        
        if confidence >60:#If confidence less than 37 then don't print predicted face text on screen
           fr.put_text(resized_img,predicted_name,x,y)


    
    cv2.imshow('face recognition tutorial ',resized_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):#wait until 'q' key is pressed
          break


cap.release()
cv2.destroyAllWindows

