import glob
import cv2
import numpy as np
import time


def view(path, img_name, X, preds, preds_per_frame, changes):
    num_frames = len(preds_per_frame.items())
    i = 1
    img_path = path + img_name

    while(True):
        if i < 1: i = 1
        if i >= num_frames - 1: i = num_frames - 1
        dets = preds_per_frame[i]
        i += show_frame(i, dets, img_path, X, preds, changes)


def show_frame(frame, detections, path, X, preds, changes):
    print 'frame: {}'.format(frame)
    img_path = path % frame
    img = cv2.imread(img_path)

    for d in detections:
        pred = preds[d]
        x = int(X[d, 0])
        y = int(X[d, 1])
        hw = int(X[d, 2])
        cv2.rectangle(img, (x-hw, y-hw), (x + hw, y + hw), (255, 0, 0), 2)
        cv2.putText(img, str(pred), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0))

    if frame in changes:
        cv2.putText(img, "Shot Change", (5, 25), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0))

    # Uncomment this to show them in GUI:
    cv2.imshow('Frame {}'.format(frame), img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
    elif key == ord('k'):
        i = 1
    elif key == ord('h'):
        i = -1
    elif key == ord('u'):
        i = -10
    elif key == ord('j'):
        i = 10
    else:
        i = 1
    cv2.destroyAllWindows()
    return i




def video_view(path, X, preds, preds_per_frame):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = 10.0
    print 'FPS: {}'.format(fps)
    print 'Frames: {}'.format(frames)
    frame_count = -1
    frame_num = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        elapsed = time.time() - start
        frame_num = np.floor(elapsed * 5.0)
        cv2.putText(frame, str(frame_num), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255 ,0))

        faces = preds_per_frame[frame_num]
        for i in faces:
            pred = preds[i]
            x = int(X[i, 0])
            y = int(X[i, 1])
            hw = int(X[i, 2])
            cv2.rectangle(frame, (x-hw, y-hw), (x + hw, y + hw), (0, 255, 0), 2)
            cv2.putText(frame, str(pred), (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))

        cv2.imshow('The Show', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
        #else:
            #key = cv2.waitKey( 1000 / int(fps) );


if __name__ == "__main__":
    scene = 3
    path = '/home/fran/school/csc412/proj/frames/scene_{}/'.format(scene)
    img_name = 'image-%04d.jpg'
    M = np.loadtxt('/home/fran/school/csc412/proj/frames/scene_{}/M.m'.format(scene))
    y_file = ('/home/fran/school/csc412/proj/frames/scene_{}/labels.txt'.format(scene))
    shot_file = path + 'scene_{}frame_ids.txt'.format(scene)
    with open(y_file,'r') as ys:
        y = ys.readlines()

    with open(shot_file, 'r') as ss:
        shot_changes = ss.readlines()

    shot_changes = [int(change) + 1 for change in shot_changes]

    num_frames = len(glob.glob(path + '/*.jpg'))
    frames = M[:, 1].astype(int)
    X = M[:, 2:]

    preds_per_frame = {}
    for i in range(0, num_frames + 1):
        preds_per_frame[i] = []

    for i, pred in enumerate(y):
        frame = frames[i]
        preds_per_frame[frame].append(i)

    view(path, img_name, X, y, preds_per_frame, shot_changes)




