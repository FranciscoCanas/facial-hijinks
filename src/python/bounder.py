import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def find_bounds(img_file):

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        print '{}-{}'.format(img_file, (x,y,w,h))

    #cv2.imshow('img',img)
    cv2.imwrite('boxed.' + img_file, img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(path):
    files = [f for f in listdir(path) if isfile(join(path,f)) ]
    for file in files:
        if '.jpg' in file or '.png' in file:
            find_bounds(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='.', help='Folder full of .jpg or .png files.')
    args = parser.parse_args()
    main(args.dir)