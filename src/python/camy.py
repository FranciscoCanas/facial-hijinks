import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def detect(img, cascade):
    for scale in [float(i)/10 for i in range(11, 25)]:
        for neighbors in range(2,5):
            rects = cascade.detectMultiScale(img, scaleFactor=scale, minNeighbors=neighbors,
                                             minSize=(40, 40), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

            print 'scale: %s, neighbors: %s, len rects: %d' % (scale, neighbors, len(rects))



def find_face_from_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, face_cascade)
    return rects, gray


if __name__ == "__main__":
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        #rects, gray = find_face_from_img(frame)
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=2,
                                             minSize=(40, 40), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # print len(faces)
        for (x,y,w,h) in faces:
            cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            print len(eyes)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



        # Display the resulting frame
    # When everything done, release the capture
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)

    cap.release()
    cv2.destroyAllWindows()

