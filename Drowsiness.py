Import cv2
importos
fromkeras.models import load_model
importnumpy as np
frompygame import mixer
import time
mixer.init()
sound = mixer.Sound(&#39;alarm.wav&#39;)
face = cv2.CascadeClassifier(&#39;haar cascade
files\haarcascade_frontalface_alt.xml&#39;)
leye = cv2.CascadeClassifier(&#39;haar cascade
files\haarcascade_lefteye_2splits.xml&#39;)
reye = cv2.CascadeClassifier(&#39;haar cascade
files\haarcascade_righteye_2splits.xml&#39;)
lbl=[&#39;Close&#39;,&#39;Open&#39;]
model = load_model(&#39;models/cnncat2.h5&#39;)
path = os.getcwd()
cap = cv2.VideoCapture(0)

31

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
while(True):
ret, frame = cap.read()
height,width = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces =
face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)
)
left_eye = leye.detectMultiScale(gray)
right_eye = reye.detectMultiScale(gray)
cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) ,
thickness=cv2.FILLED )
for (x,y,w,h) in faces:
cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

32

for (x,y,w,h) in right_eye:
r_eye=frame[y:y+h,x:x+w]
count=count+1
r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
r_eye = cv2.resize(r_eye,(24,24))
r_eye= r_eye/255
r_eye= r_eye.reshape(24,24,-1)
r_eye = np.expand_dims(r_eye,axis=0)
rpred = model.predict_classes(r_eye)
if(rpred[0]==1):
lbl=&#39;Open&#39;
if(rpred[0]==0):
lbl=&#39;Closed&#39;
Break
for (x,y,w,h) in left_eye:
l_eye=frame[y:y+h,x:x+w]
count=count+1
l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)

33
l_eye = cv2.resize(l_eye,(24,24))
l_eye= l_eye/255
l_eye=l_eye.reshape(24,24,-1)
l_eye = np.expand_dims(l_eye,axis=0)
lpred = model.predict_classes(l_eye)
if(lpred[0]==1):
lbl=&#39;Open&#39;
if(lpred[0]==0):
lbl=&#39;Closed&#39;
break
if(rpred[0]==0 and lpred[0]==0):
score=score+1
cv2.putText(frame,&quot;Closed&quot;,(10,height-20), font,
1,(255,255,255),1,cv2.LINE_AA)
# if(rpred[0]==1 or lpred[0]==1):
else:
score=score-1
cv2.putText(frame,&quot;Open&quot;,(10,height-20), font,
1,(255,255,255),1,cv2.LINE_AA)

34

if(score&lt;0):
score=0
cv2.putText(frame,&#39;Score:&#39;+str(score),(100,height-20), font,
1,(255,255,255),1,cv2.LINE_AA)
if(score&gt;15):
#person is feeling sleepy so we beep the alarm
cv2.imwrite(os.path.join(path,&#39;image.jpg&#39;),frame)
try:
sound.play()
except: # isplaying = False
pass
if(thicc&lt;16):
thicc= thicc+2

else:
thicc=thicc-2
if(thicc&lt;2):
thicc=2

35

cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
cv2.imshow(&#39;frame&#39;,frame)
if cv2.waitKey(1) &amp; 0xFF == ord(&#39;q&#39;):
break
cap.release()
cv2.destroyAllWindows()