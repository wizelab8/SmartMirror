import cv2
import os
import time

vs = cv2.VideoCapture(0)
time.sleep(0.5)

frame = vs.read()
rows = vs.get(3)
cols = vs.get(4) # float
print(rows)
print(cols)
x,y=int(rows/2),int(cols/2)

#set dimensions of input box 
w,h=100,100

#set "train" or "validation" 
TEST="train/"
VALIDATION="validation/"

#"Thumbs_up" ot "none"
THUMBS_UP="Thumbsup/"
NONE="none/"

DIRECTORY="dataThumbs/"+TEST+THUMBS_UP

count=1

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    ret, frame = vs.read()
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)     

    cv2.imshow('image', frame)

    
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    if k== ord('c'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(DIRECTORY + str(count) + '.' + ".png", gray[y:y+h,x:x+w])

    elif count >= 50: # Take 50 samples and stop video
         break

camera.release()
cv2.destroyAllWindows()
