import argparse
from os import listdir
from os.path import isfile, join
import string
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
profile_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')


def has_faces(obj):
    a = np.ones((2, 2))
    t = type(a)
    if type(obj) != t:
        return False
    if obj.size < 1:
        return False
    return True


def find_bounds(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    front_faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    profile_faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

    if has_faces(front_faces) and has_faces(profile_faces):
        faces = np.vstack((front_faces, profile_faces))
    else:
        if has_faces(front_faces):
            faces = front_faces
        else:
            faces = profile_faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        print '{}-{}'.format(img_path, (x, y, w, h))

    #cv2.imshow('img',img)
    if has_faces(faces):
        img_name, ext = string.rsplit(img_path, '.', 1)
        cv2.imwrite(img_name + '.boxed.' + ext, img)
        with open(img_name + '.vj', 'w') as f:
            f.write('{}\n'.format(len(faces)))
            for (x, y, hw, s) in box_matrix_translate(faces):
                f.write('{} {} {} {}\n'.format(x, y, hw, s))

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def box_matrix_translate(faces):
    t_faces = np.ones(shape=faces.shape)
    hw = faces[:,2] / 2
    hh = faces[:,3] / 2
    t_faces[:,0] = faces[:,0] + hw
    t_faces[:,1] = faces[:,1] + hh
    t_faces[:,2] = hw
    return t_faces


def main(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    if not files:
        print 'Nothing to be done.'
    for file in files:
        if '.jpg' in file or '.png' in file or '.jpeg' in file:
            find_bounds(path + '/' + file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Add face bounding boxes to a folder full of jpg or png files: saves the modified images as boxed.*')
    parser.add_argument('--dir', default='.', help='Folder full of .jpg or .png files.')
    args = parser.parse_args()
    main(args.dir)