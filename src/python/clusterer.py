from collections import defaultdict
import cv2
import numpy as np
from sklearn.mixture import GMM

M = np.loadtxt('./M')
frames = M[:, 1].astype(int)
X = M[:, 2:]
scene = int(M[0, 0])

path = '/home/fran/school/csc412/proj/Images/scene_{}/'.format(scene)
img_name = 'image-%03d.jpeg'

classifier = GMM(n_components=5, covariance_type='diag', init_params='wc', n_iter=20)
print 'Training'
classifier.fit(X)
print 'Fit Params:'
print classifier.get_params()
preds = classifier.predict(X)

preds_per_frame = defaultdict(list)

for i, pred in enumerate(preds):
    frame = frames[i] + 1
    preds_per_frame[frame].append(i)


for frame, l in preds_per_frame.items():
    img_path = path + img_name % frame
    img = cv2.imread(img_path)
    if not l:
        continue

    for i in l:
        pred = preds[i]
        x = int(X[i, 0])
        y = int(X[i, 1])
        hw = int(X[i, 2])
        cv2.rectangle(img, (x-hw, y-hw), (x + hw, y + hw), (255, 0, 0), 2)
        cv2.putText(img, str(pred), (x,y), cv2.FONT_HERSHEY_PLAIN, 3.0, (255, 0, 0))

    if frame > 35:
        print "what"
    # Uncomment this to show them in GUI:
    # cv2.imshow('Frame {}'.format(frame), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(path + '/tagged/' + img_name % frame, img)


