import subprocess
p1=subprocess.Popen(["python3","f1.py"])
p2=subprocess.Popen(["python3","f2.py"])


def p1open():
	p1=subprocess.Popen(["python3","f1.py"])

def p2open():
	p2=subprocess.Popen(["python3","f2.py"])

while True:
	i=input("Enter process no: ")
	if i==1:
		p2.terminate()
		p2.kill()
		p1open()
	elif i==2:
		p1.terminate()
		p1.kill()
		p2open()
	else:
		continue
			
