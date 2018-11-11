import numpy as np

import cv2

face_cascade = cv2.CascadeClassifier('F:/Program Files/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('F:/Program Files/opencv/sources/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)<br>while 1:

    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]

        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex,ey,ew,eh) in eyes:

            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    print "found " +str(len(faces)) +" face(s)"

    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:

        break

cap.release()

cv2.destroyAllWindows()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    if ([i for i in faces]):     #testa se tem rosto detectado:                                #testa se string está vazia
        face_center_x = faces[0,0]+faces[0,2]/2
        face_center_y = faces[0,1]+faces[0,3]/2

        err_x = 30*(face_center_x - frame_w/2)/(frame_w/2)
        err_y = 30*(face_center_y - frame_h/2)/(frame_h/2)
        #print("X: ",face_center_x," ","Y: ",face_center_y)
        ser.write((str(err_x) + "x!").encode())        #otimizacao: não enviar string, mas inteiro direto
        ser.write((str(err_y) + "y!").encode())        #otimizacao: não enviar string, mas inteiro direto
        print("X: ",err_x," ","Y: ",err_y)
    else:
        ser.write("o!".encode())        
                     
# When everything done, release the capture
ser.close()
cap.release()
cv2.destroyAllWindows()