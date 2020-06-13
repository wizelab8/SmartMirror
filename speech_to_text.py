import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source:
	print("Say Something")
	sudio=r.listen(source)
	print("Time over")
	
try:
	print("Text: "+r.recognize_google(audio))
except:
	pass
