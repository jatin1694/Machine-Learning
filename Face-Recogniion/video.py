#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2, time

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

frame_count = 1

while True: 
    frame_count = frame_count + 1 
    check, frame = video.read() 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.05,
                                 minNeighbors=5)
    
    #Draws recatngles around all the faces
    for x,y,w,h in faces: 
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    print(frame)
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    #Video window quits when 'q' key is pressed
    if key == ord('q'):
        break
print(frame_count)
video.release()
cv2.destroyAllWindows()


# In[ ]:




