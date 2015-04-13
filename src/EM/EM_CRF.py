from sklearn.cluster import KMeans
from load_data import *
from random import randint
import time

eps = 1e-32


def initialize(N, S, cast_counts, scene_cast, X, shot, init_mu=None, init_y=None):

    K = N + 1

    DESCS_size = X.shape[1] - 23 # -3 for x,y,half_width, -18 for PTS, -2 for scene_id, frame_id

    if not init_mu:
        mu = (np.random.random((K, DESCS_size))) # cluster means
    else:
        mu = init_mu

    # log_mu = np.log(mu)

    Y = np.zeros((X.shape[0], K))

    C = np.random.random((3, S))
    L = np.zeros((X.shape[0], K)) + np.inf

    # assignment of names (clusters) to detections
    # (respects the scene weak label constraint)
    if not init_y:
        for i in range(X.shape[0]):
            scene_id = int(X[i, 0])
            num_names = cast_counts[scene_id]
            rand_name_id = randint(0, num_names-1)

            person_id = scene_cast[scene_id][rand_name_id]
            Y[i, person_id] = 1
    else:
        Y[:,init_y] = 1

    for i in range(X.shape[0]):
        L[i, person_id] = phi_all(i, person_id, Y, X, shot, mu, C[0, :], C[1, :], C[2, :])

    return mu, Y, C, L



def phi_1(Y, X, C1, mu):

    # print "phi_1 start " + str(time.time())

    X_descs = X[:,23:]

    K = mu.shape[0]
    dets_size = X_descs.shape[0]

    _sum = 0
    d = np.zeros((dets_size, K))
    for x in range(dets_size):

        _sum = C1[X[x,0]] * Y[x,0]

        for k in range(K):
            d[x,k] = (dist(mu[k,:], X_descs[x,:])) * Y[x,k]

    _sum += np.sum(d[x,k])

    # print "phi_1 end " + str(time.time())
    return _sum


def phi_2(Y, X, S):

    print "phi_2 start " + str(time.time())

    for s in range(S):
        X_c = X[X[:,0] == s]
        if X_c.shape[0] == 0:
            continue

        F = X_c[-1, 1]
        for f in range(int(F)):
            X_cf = X_c[X_c[:,1] == f]
            if X_cf.shape[0] == 0:
                continue

            for det1 in X_cf:
                for det2 in X_cf:
                    ind1 = np.where(X == det1)[0][0]
                    ind2 = np.where(X == det2)[0][0]

                    label1 = np.where(Y[ind1, :] == 1)[0][0]
                    label2 = np.where(Y[ind2, :] == 1)[0][0]

                    if label1 == label2 and Y[ind1, 0] == 1:
                        return np.inf


    print "phi_2 start " + str(time.time())
    return 0


def phi_3(Y, X, shot, C2, C3):

    print "phi_3 start " + str(time.time())

    S = C2.shape[0]

    _sum = 0
    for s in range(S):
        X_c = X[X[:,0] == s]
        if X_c.shape[0] == 0:
            continue

        F = X_c[-1, 1]
        for f in range(int(F) - 1):

            if shot[s,f]:
                continue

            dets_cur = X_c[X_c[:,1] == f]
            dets_next = X_c[X_c[:,1] == (f+1)]

            for det1 in dets_cur:
                for det2 in dets_next:

                    ind1 = np.where(X == det1)[0][0]
                    ind2 = np.where(X == det2)[0][0]

                    label1 = np.where(Y[ind1, :] == 1)[0][0]
                    label2 = np.where(Y[ind2, :] == 1)[0][0]

                    _sum += -C2[s] * dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 != label2) + \
                    C3[s] * dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 == label2)

    print "phi_3 end " + str(time.time())
    return _sum



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

            if shot[s,f]:
                continue

            dets_cur = X_c[X_c[:,1] == f]
            dets_next = X_c[X_c[:,1] == (f+1)]

            for det1 in dets_cur:
                for det2 in dets_next:

                    ind1 = np.where(X == det1)[0][0]
                    ind2 = np.where(X == det2)[0][0]

                    label1 = np.where(Y[ind1, :] == 1)[0][0]
                    label2 = np.where(Y[ind2, :] == 1)[0][0]

                    _sum_C2[s] -= dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 != label2)
                    _sum_C3[s] += dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 == label2)

    return _sum_C2, _sum_C3

def old_phi_234(Y, X, shot, C2, C3):

    S = C2.shape[0]

    _sum = 0
    for s in range(S):
        X_c = X[X[:,0] == s]
        if X_c.shape[0] == 0:
            continue

        # F = X_c[-1, 1]
        for d in range(X_c.shape[0]):

            label = np.where(Y[d, :] == 1)[0][0]
            if not(label in scene_cast[int(s)]):
                return np.inf


            f = X_c[d,1]

            if shot[s,f]:
                continue

            X_cf = X_c[X_c[d:,1] == f]
            if X_cf.shape[0] == 0:
                continue

            det1 = X_c[d,:]
            ind1 = np.where(X == det1)[0][0]
            label1 = np.where(Y[ind1, :] == 1)[0][0]

            #phi_2
            for det2 in X_cf:
                ind2 = np.where(X == det2)[0][0]
                label2 = np.where(Y[ind2, :] == 1)[0][0]

                if label1 == label2 and Y[ind1, 0] == 1:
                    return np.inf

            #phi_3
            dets_next = X_c[X_c[:,1] == (f+1)]
            for det2 in dets_next:
                ind2 = np.where(X == det2)[0][0]
                label2 = np.where(Y[ind2, :] == 1)[0][0]

                _sum += -C2[s] * dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 != label2) + \
                C3[s] * dist(det1[2:22], det2[2:22]) * (label1 != 0) * (label2 != 0) * (label1 == label2)


            _sum += phi_1(Y,X,C[0,:], mu)

    print "loss end " + str(time.time())
    return _sum



def phi_all(i, k, Y, X, shot, mu, C1, C2, C3):

    s = X[i,0]
    f = X[i,1]

    if not(k in scene_cast[int(s)]):
        return np.inf

    _sum = 0

    filter = (X[:,1] == f) & (X[:,0] == s)
    # X_cf_inds = np.where(X == filter)[0][0]
    X_cf_inds = np.where(filter == True)[0]

    filter = (X[:,1] == f-1) & (X[:,0] == s)
    try:
        X_pf_inds = np.where(filter == True)[0]
    except:
        X_pf_inds = []


    filter = (X[:,1] == f+1) & (X[:,0] == s)
    try:
        X_nf_inds = np.where(filter == True)[0]
    except:
        X_nf_inds = []


    for det_cf in X_cf_inds:

        if det_cf == i:
            continue

        label2 = np.where(Y[det_cf, :] == 1)[0][0]

        if k == label2 and Y[i, 0] == 0:
            return np.inf

    if shot[s,f-1]:
         _sum += time_pairwise(i, k, X_pf_inds, Y, X, C2, C3)

    if shot[s,f]:
        _sum += time_pairwise(i, k, X_nf_inds, Y, X, C2, C3)


    #phi_1
    _sum += dist(mu[k, :], X[i, 23:]) + C1[s] * Y[i, 0]

    return _sum


def time_pairwise(i, k, X_o, Y, X, C2,C3):

    s = X[i,0]
    f = X[i,1]

    _sum = 0
    for x in X_o:
        label2 = np.where(Y[x, :] == 1)[0][0]

        _sum += -C2[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k != label2) + \
            C3[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k == label2)

    return _sum


def phi_4(Y, X, scene_cast):

    print "phi_4 start " + str(time.time())

    dets_size = Y.shape[0]
    for x in range(dets_size):
        s = X[x, 0]
        label = np.where(Y[x, :] == 1)[0][0]

        if not(label in scene_cast[int(s)]):
            return np.inf

    print "phi_4 end " + str(time.time())
    return 0


def loss_function(Y, X, C, shot, mu, scene_cast):
    S = C.shape[1]
    # return phi_1(Y, X, C[0,:], mu) + phi_2(Y,X,S) + phi_3(Y, X, shot, C[1,:], C[2,:]) + phi_4(Y, X, scene_cast)

    pass
    #return phi_234(Y, X, shot, C[1,:], C[2,:])



def E_step(Y, X, C, shot, mu, scene_cast, L):

    K = mu.shape[0]

    for y in range(Y.shape[0]):
        Y[y,:] = np.zeros((K))
        for k in range(K):
            L[y,k] = phi_all(y,k,Y,X,shot,mu,C[0,:],C[1,:],C[2,:])

        m = np.argmin(L[y,:])
        try:
            if m.shape[0] > 0:
                print "WE HAVE A PROBLEM"
        except:
            pass
        Y[y,m] = 1


        # loss = np.zeros((K))
        # for k in range(K):
        #     Y[y,:] = np.zeros((K))
        #     Y[y, k] = 1
        #     loss[k] = loss_function(Y, X, C, shot, mu, scene_cast)
        #
        # ind = np.argmin(loss)
        # Y[y,:] = np.zeros((K))
        # Y[y, ind] = 1

    return Y


# def E_step(mu, X_descs, Y):
#
#     K = mu.shape[0]
#     dets_size = X_descs.shape[0]
#
#     Y = np.zeros((dets_size, K))
#
#     d = np.zeros((dets_size, K))
#     for x in range(dets_size):
#         for k in range(K):
#             d[x,k] = dist(mu[k,:], X_descs[x,:])
#
#         ind = np.argmin(d[x,:])
#         Y[x, ind] = 1
#
#     return Y


def M_step(Y,X,mu,C,shot):

    S = shot.shape[0]
    K = mu.shape[0]

    for k in range(K):
        mu[k,:] = np.sum(Y[:, k].reshape(-1,1) * X[:, 23:], axis=0) / (np.sum(Y[:,k]) + eps)

    for s in range(S):
        C[0, s] -= np.sum(Y[X[0] == s, 0])

    dC2, dC3 = M_step_C2_C3(Y, X, shot)
    C[1,:] -= dC2
    C[2,:] -= dC3


def sum_loss(L,Y):
    _sum = 0
    for y in range(Y.shape[0]):
        label = np.where(Y[y, :] == 1)[0][0]
        _sum += L[y,label]
    return _sum

def EM(Y, X, C, shot, mu, scene_cast, L, maxiters=20):
    print "STARTING EM. OR GRADIENT DESCENT. OR ICM. OR WHATEVER THIS IS :D "

    for iter in range(maxiters):

        print "iteration " + str(iter)

        E_step(Y,X,C,shot,mu,scene_cast,L)
        M_step(Y,X,mu,C,shot)

        l = sum_loss(L,Y)
        print l

        #save_model(C, mu, )

    return Y


def dist(A, B):
    # mu*mu.T + X*X.T   # (K+1, dets)
    return np.sum((A - B)**2)



def name_predictions(Y, name_dict):
    names = []
    for y in Y:
        ind = np.nonzero(y)[0][0]
        names.append(name_dict[ind])
    return names


if __name__ == '__main__':

    prepend = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/'

    dir_cast = prepend+'cast/'
    dir_feature_matrix = prepend+'feature_matrix_files/'
    frames_dir = prepend+'scenes/scene_frames/'
    dir_shot_change = prepend+'shot_changes/'

    out_path = '/Users/elenitriantafillou/model_output/'

    scene_cast, cast_counts, name_dict, S = load_data(dir_cast)
    X = construct_feature_matrix(dir_feature_matrix, S)
    max_frames = np.max(X[:, 1]) + 1
    shot_change = construct_shot_change(dir_shot_change, S, max_frames)

    N = len(name_dict.keys())

    # K-means initialization: uncomment for true k-means awesomeness in your faces
    # classifier = KMeans(n_clusters=N, max_iter=100, precompute_distances=True)
    # print 'K-Means Training'
    # classifier.fit(X[:, 23:])
    # y = classifier.predict(X)
    # centers = classifier.cluster_centers_
    # mu, Y, C, L = initialize(N, S, cast_counts, scene_cast, X, shot_change, centers, y) # Uncomment to use k-means init

    mu, Y, C, L = initialize(N, S, cast_counts, scene_cast, X, shot_change)

    Y = EM(Y, X, C, shot_change, mu, scene_cast, L)

    for s in range(2, S):
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
