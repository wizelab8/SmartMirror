import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(37, GPIO.IN)         #Read output from PIR motion sensor
while True:
	i=GPIO.input(37)
	if i==0:                 #When output from motion sensor is LOW
		print("No intruders")
	else:
		print("Intruder")
	time.sleep(0.5)
