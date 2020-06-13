

from Tkinter import *
import locale
import threading
import time
import requests
import json
import traceback
import feedparser

from PIL import Image, ImageTk
from contextlib import contextmanager

import cv2 #For Image processing 
import numpy as np #For converting Images to Numerical array 
import os #To handle directories 
from PIL import Image #Pillow lib for handling images
import faceRecognition as fr 
import imutils

LOCALE_LOCK = threading.Lock()

      
class Namein(Frame):
    def __init__(self ,parent ,*args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.config(bg='black')     
        self.title ='Hello'# 'News' is more internationally generic
        self.namein = Label(self, text=self.title, font=('Helvetica', medium_text_size), fg="white", bg="black")              
        self.namein.pack(side=TOP,anchor=S)
        self.get_names()
        
    def get_names(self):        
        predicted_name="null"
        name = {0 : "elon",1 : "jeevak"}
        while(predicted_name=="null"):
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
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    resized_img = cv2.resize(test_img, (200, 200))
                    if confidence>60:#If confidence less than 37 then don't print predicted face text on screen
                       predicted_name=name[label]
                       #Label(self, text=predicted_name, font=('Helvetica', medium_text_size), fg="white", bg="black")
                       self.title =predicted_name# 'News' is more internationally generic
                       self.namein = Label(self, text=self.title, font=('Helvetica', medium_text_size), fg="white", bg="black")
        self.after(5000, self.get_names)


class FullscreenWindow:

    def __init__(self):
        self.tk = Tk()
        self.tk.configure(background='black')
        self.topFrame = Frame(self.tk, background = 'black')
        self.bottomFrame = Frame(self.tk, background = 'black')
        self.topFrame.pack(side = TOP, fill=BOTH, expand = YES)
        self.bottomFrame.pack(side = BOTTOM, fill=BOTH, expand = YES)
        self.state = False
        self.tk.bind("<Return>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)
        # clock
        self.clock = Clock(self.topFrame)
        self.clock.pack(side=RIGHT, anchor=N, padx=100, pady=60)
        # weather
        self.weather = Weather(self.topFrame)
        self.weather.pack(side=LEFT, anchor=N, padx=100, pady=60)
        # news
        #self.news = News(self.bottomFrame)
        #self.news.pack(side=LEFT, anchor=S, padx=100, pady=60)
        # name
        self.namein = Namein(self.bottomFrame)
        self.namein.pack(side=RIGHT ,anchor=S, padx=100, pady=60)
        # calender - removing for now
        # self.calender = Calendar(self.bottomFrame)
        # self.calender.pack(side = RIGHT, anchor=S, padx=100, pady=60)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.tk.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.tk.attributes("-fullscreen", False)
        return "break"



if __name__ == '__main__':
    
    labels = ["elon", "jeevak"] 
    predicted_name = "null"

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_recognizer = cv2.createLBPHFaceRecognizer()
    face_recognizer.load('trainingData.yml')

    cap = cv2.VideoCapture(0)#Get vidoe feed from the Camera

    
    w = FullscreenWindow()
    w.tk.mainloop()
