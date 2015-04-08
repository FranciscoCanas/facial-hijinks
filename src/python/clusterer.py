from collections import defaultdict
import cv2
import numpy as np
from sklearn.mixture import GMM
from viewer import view, video_view

M = np.loadtxt('/home/fran/school/csc412/proj/frames/scene_3/M')
frames = M[:, 1].astype(int)
X = M[:, 2:]
scene = int(M[0, 0])

vid_path = '/home/fran/school/csc412/proj/frames/scene_3/scene3.mp4'
path = '/home/fran/school/csc412/proj/frames/scene_{}/'.format(scene)

img_name = 'image-%04d.jpg'

classifier = GMM(n_components=4, covariance_type='diag', init_params='wc', n_iter=100)
print 'Training'
classifier.fit(X)
print 'Fit Params:'
print classifier.get_params()
y = classifier.predict(X)

preds_per_frame = {}
for i in range(0, 275):
    preds_per_frame[i] = []

for i, pred in enumerate(y):
    frame = frames[i]
    preds_per_frame[frame].append(i)

view(path, img_name, X, y, preds_per_frame)
#video_view(vid_path, X, y, preds_per_frame)