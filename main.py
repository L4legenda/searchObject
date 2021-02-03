import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = cv2.imread('test.jpg')
    # image = frame

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    faces_detected = "Лиц обнаружено: " + format(len(faces))
    print(faces_detected)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # Display the resulting frame
    # cv2.imshow('frame',image)
    cv2.imwrite("test2.jpg", image)
    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()