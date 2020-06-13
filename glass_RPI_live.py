from FileImports import *
TRAINING_OBJECT=1


faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

coord=None
flag=1
colorLower = (24, 100, 100)
colorUpper = (44, 255, 255)
WINDOW_WIDTH=600
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = imutils.resize(frame, width=WINDOW_WIDTH)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    crop=frame.copy()
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        coord=(x,y,w,h)

    # Display the resulting frame
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('c') or len(faces)==1 and y-h/5>0:
        crop_img = crop[y-h//5: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
        cv2.imwrite("face.jpg", crop_img)
        
        break
        


model = load_model('models/facial_feature_model.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('face.jpg')
img = cv2.resize(img,(200,200))
img = np.reshape(img,[1,200,200,3])

classes = model.predict(img)

print(classes)
dict_glasses={}
glass_no=0
for i in classes[0]:
	dict_glasses[str(glass_no)]=i
	glass_no+=1
print("dict",dict_glasses)
glass_no=0
dict_glasses={k: v for k, v in sorted(dict_glasses.items(), key=lambda item: item[1],reverse=True)}
list_glasses=list(dict_glasses.keys())
print(list_glasses)
print("GLASS Number: "+str(list_glasses[glass_no]))

rows = video_capture.get(4)
cols = video_capture.get(3) # float
print(rows)
print(cols)

w,h=75,75
y,x=int(rows/2),int(cols-2*w)
print("x:",x," y:",y)
thumb_model = load_model('models/thumb_model.h5')

thumb_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


#glass_no=3
# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")

#vs = VideoStream(usePiCamera=True).start() # Raspberry Pi


#--------
GLASS_DIR="clothing_accessories/glass_frames/"
print(GLASS_DIR+str(list_glasses[glass_no])+".png")
nose_image = cv2.imread(GLASS_DIR+str(list_glasses[glass_no])+".png")
nose_image.setflags(write=1)
ret, frame  = video_capture.read()
rows, cols, _ = frame.shape
print(frame.shape)
nose_mask = np.zeros((rows, cols), np.uint8)
face_img = np.zeros((rows, cols), np.uint8);
#---------

# loop over the frames from the video stream
while True:	
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	ret, frame  = video_capture.read()
	frame_dup=frame.copy()
	
	frame = imutils.resize(frame, width=WINDOW_WIDTH)
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
			face_img=frame_dup[bY:bY+int(1.7*bH),bX:bX+int(1.5*bW)]
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
			try:
				nose_area = frame[top_left[1]: top_left[1] + nose_height,
							top_left[0]: top_left[0] + nose_width]
				nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
				final_nose = cv2.add(nose_area_no_nose, nose_pig)
				frame[top_left[1]:top_left[1]+nose_height, top_left[0]:top_left[0]+nose_width] = final_nose
			except:
				continue
	

	cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)     
	thumb_img=frame[y:y+w,x:x+h]
	cv2.imshow("thumb",thumb_img)
	thumb_img = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2GRAY)
	thumb_img = cv2.resize(thumb_img,(100,100))
	thumb_img = np.reshape(thumb_img,[1,100,100,1])
	classes = thumb_model.predict_classes(thumb_img)
	if classes==0:
        	cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,255), 4)
        	now = datetime.now()
        	current_time = now.strftime("%H:%M:%S")
        	cv2.imwrite("data/train/"+str(list_glasses[glass_no])+"/"+str(current_time)+".png", face_img)
        	flag+=1
        	glass_no=(flag%len(list_glasses))
        	nose_image = cv2.imread(GLASS_DIR+str(list_glasses[glass_no])+".png")
        	nose_image.setflags(write=1)
        	rows, cols, _ = frame.shape
        	nose_mask = np.zeros((rows, cols), np.uint8)
        	time.sleep(3)
		
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, colorLower, colorUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	 
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((xx, yy), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		 
		# only proceed if the radius meets a minimum size
		if radius > 10 and radius <30:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(xx), int(yy)), int(radius),(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			flag+=1
			glass_no=(flag%len(list_glasses))
			nose_image = cv2.imread(GLASS_DIR+str(list_glasses[glass_no])+".png")
			nose_image.setflags(write=1)
			rows, cols, _ = frame.shape
			nose_mask = np.zeros((rows, cols), np.uint8)
			time.sleep(2)
	
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.stop()
