from keras.models import load_model
import cv2
import numpy as np
import time

vs = cv2.VideoCapture(0)
time.sleep(0.5)


frame = vs.read()
rows = vs.get(3)
cols = vs.get(4) # float
print(rows)
print(cols)
x,y=int(rows/2),int(cols/2)
w,h=100,100
count=85


model = load_model('thumb_model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    ret, frame = vs.read()
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)     
    img=frame[y:y+h,x:x+w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = np.reshape(img,[1,100,100,1])

    classes = model.predict_classes(img)
    print(classes)
    if classes==0:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,255), 4)
    
   
    cv2.imshow('image', frame)
    
    
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
