from collections import defaultdict
import glob
import cv2
import numpy as np
from sklearn.mixture import GMM
from viewer import view, video_view
from sklearn.cluster import KMeans

M = np.loadtxt('/home/fran/school/csc412/proj/frames/scene_4/M.m')
frames = M[:, 1].astype(int)
X = M[:, 2:]
scene = int(M[0, 0])


vid_path = '/home/fran/school/csc412/proj/frames/scene_{}/scene{}.mp4'.format(scene, scene)
path = '/home/fran/school/csc412/proj/frames/scene_{}/'.format(scene)
num_frames = len(glob.glob(path + '/*.jpg'))
img_name = 'image-%04d.jpg'

classifier = KMeans(n_clusters=4, max_iter=100, precompute_distances=True)
print 'Training'
classifier.fit(X)
print 'Fit Params:'
print classifier.get_params()
centers = classifier.cluster_centers_
classifier = GMM(n_components=4, covariance_type='diag', init_params='wc', n_iter=100)
classifier.means_ = centers
classifier.fit(X)
y = classifier.predict(X)
preds_per_frame = {}
for i in range(0, num_frames + 1):
    preds_per_frame[i] = []

for i, pred in enumerate(y):
    frame = frames[i]
    preds_per_frame[frame].append(i)

view(path, img_name, X, y, preds_per_frame)
#video_view(vid_path, X, y, preds_per_frame)