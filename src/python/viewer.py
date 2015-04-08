import cv2
import time


def view(path, img_name, X, preds, preds_per_frame):
    for frame, l in preds_per_frame.items():
        print 'frame: {}'.format(frame)
        img_path = path + img_name.format(frame)
        img = cv2.imread(img_path)
        if not l:
            continue
        for i in l:
            pred = preds[i]
            x = int(X[i, 0])
            y = int(X[i, 1])
            hw = int(X[i, 2])
            cv2.rectangle(img, (x-hw, y-hw), (x + hw, y + hw), (255, 0, 0), 2)
            cv2.putText(img, str(pred), (x, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0))

        # Uncomment this to show them in GUI:
        cv2.imshow('Frame {}'.format(frame), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite(path + '/tagged/' + img_name % frame, img)


def video_view(path, X, preds, preds_per_frame):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    print 'FPS: {}'.format(fps)
    print 'Frames: {}'.format(frames)
    frame_count = -1
    frame_num = -1
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        elapsed = time.time() - start

        if (frame_count % 10) == 0:
            frame_num += 1
        print frame_count
        print frame_num

        faces = preds_per_frame[frame_num]
        for i in faces:
            pred = preds[i]
            x = int(X[i, 0])
            y = int(X[i, 1])
            hw = int(X[i, 2])
        if x and y and hw:
            cv2.rectangle(frame, (x-hw, y-hw), (x + hw, y + hw), (255, 0, 0), 2)
            cv2.putText(frame, str(pred), (x, y), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0))

        cv2.imshow('The Show', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)


if __name__ == "__main__":
    video_view(1, 2, 3, 4)



