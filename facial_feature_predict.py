from keras.models import load_model
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)
model = load_model('models/facial_feature_model.h5')

model.compile(loss='categorical_crossentropy',
	      optimizer='rmsprop',
	      metrics=['accuracy'])
rows = video_capture.get(3)
cols = video_capture.get(4) # float
print(rows)
print(cols)
x,y=int(rows/2),int(cols/2)
w,h=200,200

while True:
	ret, frame = video_capture.read()
	img=frame[y:y+h,x:x+w]
	cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)     
	img = cv2.resize(img,(200,200))
	img = np.reshape(img,[1,200,200,3])
	classes = model.predict(img)
	print(classes)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
			break

video_capture.release()
cv2.destroyAllWindows()
