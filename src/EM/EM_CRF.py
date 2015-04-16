from sklearn.cluster import KMeans
from load_data import *
from random import randint
import time

np.seterr(all='warn')

eps = 1e-32


def initialize(K, S, cast_counts, scene_cast, X, shot, init_mu=None, init_y=None):

    DESCS_size = X.shape[1] - 23 # -3 for x,y,half_width, -18 for PTS, -2 for scene_id, frame_id

    if init_mu == None:
        mu = (np.random.random((K, DESCS_size))) # cluster means
    else:
        mu = init_mu

    # log_mu = np.log(mu)

    Y = np.zeros((X.shape[0], K))

    C = np.random.random((3, S))
    L = np.zeros((X.shape[0], K)) + np.inf

    # assignment of names (clusters) to detections
    # (respects the scene weak label constraint)
    if init_y == None:
        for i in range(X.shape[0]):
            scene_id = int(X[i, 0])
            num_names = cast_counts[scene_id]
            rand_name_id = randint(0, num_names-1)

            person_id = scene_cast[scene_id][rand_name_id]
            Y[i, person_id] = 1
    else:
        Y[:, init_y] = 1


    for i in range(X.shape[0]):
        person_id = np.where(Y[i, :] == 1)[0][0]
        L[i, person_id] = phi_all(i, person_id, Y, X, shot, mu, C[0, :], C[1, :], C[2, :])

    return mu, Y, C, L




def M_step_C2_C3(Y, X, shot):

    S = shot.shape[0]

    _sum_C2 = np.zeros((S))
    _sum_C3 = np.zeros((S))

    for s in range(S):
        X_c = X[X[:,0] == s]
        if X_c.shape[0] == 0:
            continue

        F = X_c[-1, 1]
        for f in range(int(F) - 1):

            if shot[s, f]:
                continue

            dets_cur = X_c[X_c[:, 1] == f]
            dets_next = X_c[X_c[:, 1] == (f+1)]

            for det1 in dets_cur:

                ind1 = np.where(np.all(X == det1, axis=1))[0][0]
                label1 = np.where(Y[ind1, :] == 1)[0][0]

                if label1 == 0:
                    continue

                for det2 in dets_next:

                    ind2 = np.where(np.all(X == det2, axis=1))[0][0]
                    label2 = np.where(Y[ind2, :] == 1)[0][0]

                    if label2 == 0:
                        continue

                    _sum_C2[s] += dist(det1[2:23], det2[2:23]) * (label1 != 0) * (label2 != 0) * (label1 != label2)
                    _sum_C3[s] += dist(det1[2:23], det2[2:23]) * (label1 != 0) * (label2 != 0) * (label1 == label2)

    return _sum_C2, _sum_C3



def phi_all(i, k, Y, X, shot, mu, C1, C2, C3):

    s = X[i,0]
    f = X[i,1]

    if not(k in scene_cast[int(s)]) and (k != 0):
        return np.inf

    _sum = 0

    filter = (X[:, 1] == f) & (X[:, 0] == s)
    X_cf_inds = np.where(filter == True)[0]

    filter = (X[:, 1] == f-1) & (X[:, 0] == s)
    try:
        X_pf_inds = np.where(filter == True)[0]
    except:
        X_pf_inds = []


    filter = (X[:, 1] == f+1) & (X[:, 0] == s)
    try:
        X_nf_inds = np.where(filter == True)[0]
    except:
        X_nf_inds = []


    for det_cf in X_cf_inds:

        if det_cf == i:
            continue

        label2 = np.where(Y[det_cf, :] == 1)[0][0]

        if k == label2 and k != 0:
            return np.inf

    if (shot[s, f-1] == 0):
         _sum += time_pairwise(i, k, X_pf_inds, Y, X, C2, C3)

    if (shot[s, f] == 0):
        _sum += time_pairwise(i, k, X_nf_inds, Y, X, C2, C3)


    #phi_1
    _sum += dist(mu[k, :], X[i, 23:]) + C1[s] * Y[i, 0] + (C1[s]**2)/2
    _sum += (C2[s]**2)/2 + (C3[s]**2)/2
    return _sum


def time_pairwise(i, k, X_o, Y, X, C2, C3):

    s = X[i, 0]
    f = X[i, 1]

    _sum = 0
    for x in X_o:
        label2 = np.where(Y[x, :] == 1)[0][0]

        _sum += -C2[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k != label2) + \
            C3[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k == label2)

    return _sum


def E_step(Y, X, C, shot, mu, scene_cast, L):

    K = mu.shape[0]

    for y in range(Y.shape[0]):
        Y[y, :] = np.zeros((K))
        for k in range(K):
            L[y, k] = phi_all(y, k, Y, X, shot, mu, C[0, :], C[1, :], C[2, :])

        m = np.argmin(L[y, :])
        try:
            if m.shape[0] > 0:
                print "WE HAVE A PROBLEM"
        except:
            pass
        Y[y, m] = 1

    return Y


def M_step(Y, X, mu, C, shot):

    S = shot.shape[0]
    K = mu.shape[0]

    for k in range(K):
        mu[k, :] = np.sum(Y[:, k].reshape(-1, 1) * X[:, 23:], axis=0) / (np.sum(Y[:, k]) + eps)

    for s in range(S):
        C[0, s] = -np.sum(Y[X[:, 0] == s, 0])

    C2, C3 = M_step_C2_C3(Y, X, shot)
    C[1, :] = C2
    C[2, :] = -C3

    print "C1 " + str(C[0,:]) + '\n'
    print "C2 " + str(C[1,:]) + '\n'
    print "C3 " + str(C[2,:]) + '\n'


def sum_loss(L, Y):
    _sum = 0
    for y in range(Y.shape[0]):
        label = np.where(Y[y, :] == 1)[0][0]
        _sum += L[y, label]
    return _sum

def EM(Y, X, C, shot, mu, scene_cast, L, outdir, name_dict, maxiters=20, kmeans=True):
    print "STARTING EM. OR GRADIENT DESCENT. OR ICM. OR WHATEVER THIS IS :D "

    l = sum_loss(L, Y)
    print "initial loss: " + str(l)

    for iter in range(maxiters):

        print "iteration " + str(iter)

        E_step(Y, X, C, shot, mu, scene_cast, L)
        M_step(Y, X, mu, C, shot)

        l = sum_loss(L, Y)
        print l

        save_model(C, mu, Y, iter, outdir, name_dict, kmeans)
        label_hist(Y, 2)

    return Y

def save_model(C, mu, Y, iter, out_dir, name_dict, kmeans):

    if kmeans:
        outfile = out_dir+'/kmeans_init_model_iter_'+str(iter)
    else:
        outfile = out_dir+'/model_iter_'+str(iter)

    labels = []
    for y in range(Y.shape[0]):
        label = np.where(Y[y, :] == 1)[0][0]

        for (k, v) in name_dict.items():
            if label == v:
                name = k
                break

        labels.append(name)

    np.savez(outfile, C, mu, Y, labels)


def dist(A, B):
    # mu*mu.T + X*X.T   # (K+1, dets)
    return np.sum((A - B)**2)


def label_hist(Y, s):

    dets_num = Y.shape[0]
    character_num = Y.shape[1]

    counts = np.zeros((character_num))
    for i in range(dets_num):
        if (X[i, 0] != s):
            continue
        label = np.where(Y[i, :] == 1)[0][0]
        counts[label] += 1

    for c in range(character_num):
        for (k, v) in name_dict.items():
            if c == v:
                name = k
                break

        print name + ': ' + str(int(counts[c])) + '\n'


if __name__ == '__main__':

    use_kmeans = True

    prepend = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/'
    # prepend = '/u/eleni/412-project/the_mentalist_1x19/'

    dir_cast = prepend+'cast/'
    dir_feature_matrix = prepend+'feature_matrix_files/'
    frames_dir = prepend+'scenes/scene_frames/'
    dir_shot_change = prepend+'shot_changes/'

    out_path = '/Users/elenitriantafillou/model_output/'
    # out_path = '/u/eleni/model_output/'

    scene_cast, cast_counts, name_dict, S = load_data(dir_cast)

    S = 15

    X = construct_feature_matrix(dir_feature_matrix)
    max_frames = np.max(X[:, 1]) + 1
    shot_change = construct_shot_change(dir_shot_change, S, max_frames)

    N = len(name_dict.keys())

    # K-means initialization: uncomment for true k-means awesomeness in your faces
    if use_kmeans:
        classifier = KMeans(n_clusters=N, max_iter=100, precompute_distances=True)
        print 'K-Means Training'
        classifier.fit(X[:, 23:])
        y = classifier.predict(X[:, 23:])
        centers = classifier.cluster_centers_
        mu, Y, C, L = initialize(N, S, cast_counts, scene_cast, X, shot_change, centers, y) # Uncomment to use k-means init

    else:
        mu, Y, C, L = initialize(N, S, cast_counts, scene_cast, X, shot_change)

    Y = EM(Y, X, C, shot_change, mu, scene_cast, L, out_path, name_dict, kmeans=use_kmeans)

    for s in range(2, S):
        if use_kmeans:
            scene_out = out_path+'scene'+str(s)+'/kmeans_init_labels.txt'
        else:
            scene_out = out_path+'scene'+str(s)+'/labels.txt'
        f = open(scene_out, 'w')
        for y in range(Y.shape[0]):
            if X[y, 0] != s:
                continue
            label = np.where(Y[y, :] == 1)[0][0]

            for (k, v) in name_dict.items():
                if label == v:
                    name = k
                    break

            f.write(name+'\n')
