# USAGE
# python faster_facial_landmarks.py --shape-predictor shape_predictor_5_face_landmarks.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from math import hypot


# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

#--------
nose_image = cv2.imread("3.png")
frame = vs.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)
#---------

# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # check to see if a face was detected, and if so, draw the total
        # number of faces on the frame
        if len(rects) > 0:
                text = "{} face(s) found".format(len(rects))
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        # loop over the face detections
        for rect in rects:
                # compute the bounding box of the face and draw it on the
                # frame
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                        (0, 255, 0), 1)

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                #shape = face_utils.shape_to_np(shape)

                #-------------
                left_eye=(shape.part(0).x, shape.part(0).y)
                right_eye=(shape.part(2).x, shape.part(2).y)
                center_nose=((shape.part(1).x+shape.part(3).x)/2, (shape.part(1).y+shape.part(3).y)/2)
                
                nose_width = int(hypot(left_eye[0] - right_eye[0],left_eye[1] - right_eye[1]) * 1.7)
                nose_height = int(nose_width * 0.33)
                
                top_left = (int(center_nose[0] - nose_width/2),
                              int(center_nose[1] - nose_height/2))
                bottom_right = (int(center_nose[0] + nose_width/2),
                               int(center_nose[1] + nose_height/2))

                # Adding the new nose
                nose_pig = cv2.resize(nose_image,(nose_width, nose_height))
                nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
                _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

                nose_area = frame[top_left[1]: top_left[1] + nose_height,
                            top_left[0]: top_left[0] + nose_width]
                nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
                final_nose = cv2.add(nose_area_no_nose, nose_pig)

                frame[top_left[1]: top_left[1] + nose_height,
                            top_left[0]: top_left[0] + nose_width] = final_nose
                #------------

                cv2.imshow("Nose area", nose_area)
                cv2.imshow("Nose pig", nose_pig)
                cv2.imshow("final nose", final_nose)

                shape = face_utils.shape_to_np(shape)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
                for (i, (x, y)) in enumerate(shape):
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                        cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
