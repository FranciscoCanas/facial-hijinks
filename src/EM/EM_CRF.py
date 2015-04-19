from sklearn.cluster import KMeans
from load_data import *
from random import randint
from sklearn import metrics

import time

np.seterr(all='warn')

eps = 1e-32


def initialize(K, S, cast_counts, scene_cast, X, shot, init_mu=None, init_y=None):

    DESCS_size = X.shape[1] - 23 # -3 for x,y,half_width, -18 for PTS, -2 for scene_id, frame_id

    if init_mu == None:
        mu = (np.random.random((K, DESCS_size))) # cluster means
    else:
        mu = init_mu

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
        for d in range(Y.shape[0]):
            Y[d, init_y[d]] = 1

    for i in range(X.shape[0]):
        person_id = np.where(Y[i, :] == 1)[0][0]
        L[i, person_id] = phi_all(i, person_id, Y, X, shot, mu, C[0, :], C[1, :], C[2, :])

    return mu, Y, C, L


def M_step_C2_C3(Y, X, shot):

    S = shot.shape[0]

    _sum_C2 = np.zeros((S))
    _sum_C3 = np.zeros((S))

    for s in range(S):
        X_c = X[X[:, 0] == s]
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

                    _sum_C2[s] += dist(det1[2:23], det2[2:23]) * (label1 != label2)
                    _sum_C3[s] += dist(det1[2:23], det2[2:23]) * (label1 == label2)

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

    _sum = 0
    for x in X_o:
        label2 = np.where(Y[x, :] == 1)[0][0]

        _sum += -C2[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k != label2) + \
            C3[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != 0) * (label2 != 0) * (k == label2)
        #
        # _sum += -C2[s] * dist(X[i, 2:23], X[x, 2:23]) * (k != label2) + \
        #     C3[s] * dist(X[i, 2:23], X[x, 2:23]) * (k == label2)

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

def EM(Y_tr, X_tr, X_test, C, shot, mu, scene_cast, L_tr, outdir, name_dict, t3, t5, t6, test_scenes, maxiters=50, kmeans=True):
    print "STARTING EM. OR GRADIENT DESCENT. OR ICM. OR WHATEVER THIS IS :D "

    accs_tr = np.zeros((maxiters+1))
    acc_tr = evaluate_train(X_tr, Y_tr, t3, t5, t6, test_scenes)
    accs_tr[0] = acc_tr #kmeans acc
    print "kmeans acc " + str(acc_tr)

    accs_test = np.zeros((maxiters+1))
    if not (test_scenes == []):
        Y_test = predict(X_test, shot_change, mu, scene_cast, C, test_scenes)
        acc_test = get_test_accuracy(Y_test, t3, t5, t6, test_scenes)
        accs_test[0] = acc_test

    sil_score = np.zeros((maxiters+1))
    labels_tr = np.where(Y_tr == 1)[1]
    score = metrics.silhouette_score(X_tr, labels_tr)
    sil_score[0] = score

    l = sum_loss(L_tr, Y_tr)
    print "initial loss: " + str(l)

    for iter in range(maxiters):

        print "iteration " + str(iter)

        E_step(Y_tr, X_tr, C, shot, mu, scene_cast, L_tr)
        M_step(Y_tr, X_tr, mu, C, shot)

        l = sum_loss(L_tr, Y_tr)
        print l

        save_model(C, mu, Y_tr, iter, outdir, name_dict, kmeans)

        # evaluate train set using ground truths:
        acc_tr = evaluate_train(X_tr, Y_tr, t3, t5, t6, test_scenes)
        accs_tr[iter+1] = acc_tr
        print accs_tr

        # evaluate test using ground truths:
        if not (test_scenes == []):
            Y_test = predict(X_test, shot_change, mu, scene_cast, C, test_scenes)
            acc_test = get_test_accuracy(Y_test, t3, t5, t6, test_scenes)
            accs_test[iter+1] = acc_test
            print accs_test

        labels_tr = np.where(Y_tr == 1)[1]
        sil_score[iter+1] = metrics.silhouette_score(X_tr, labels_tr)
        print sil_score

    return Y_tr, accs_tr, accs_test, sil_score

def save_model(C, mu, Y, iter, out_dir, name_dict, kmeans):

    if kmeans:
        outfile = out_dir+'/better_kmeans_init_iter_'+str(iter)
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
    return np.sum((A - B)**2)


def evaluate_train(X, Y, t3, t5, t6, test_scenes):

    num_dets_3 = X[X[:,0] == 3, :].shape[0]
    num_dets_5 = X[X[:,0] == 5, :].shape[0]
    num_dets_6 = X[X[:,0] == 6, :].shape[0]

    label_3 = np.where(Y[X[:, 0] == 3, :] == 1)[1]
    label_5 = np.where(Y[X[:, 0] == 5, :] == 1)[1]
    label_6 = np.where(Y[X[:, 0] == 6, :] == 1)[1]

    got_rights_3 = np.where(label_3 == t3)[0].shape[0]
    got_rights_5 = np.where(label_5 == t5)[0].shape[0]
    got_rights_6 = np.where(label_6 == t6)[0].shape[0]

    if 3 in test_scenes:
        num_dets = num_dets_5 + num_dets_6
        got_rights = got_rights_5 + got_rights_6
    elif 5 in test_scenes:
        num_dets = num_dets_3 + num_dets_6
        got_rights = got_rights_3 + got_rights_6
    elif 6 in test_scenes:
        num_dets = num_dets_5 + num_dets_3
        got_rights = got_rights_5 + got_rights_3
    elif test_scenes == []:
        num_dets = num_dets_5 + num_dets_3 + num_dets_6
        got_rights = got_rights_5 + got_rights_3 + got_rights_6

    acc = got_rights / float(num_dets)
    return acc


# assuming that the test set is only one scene...
def get_test_accuracy(Y_test, t3, t5, t6, test_scenes):

    if 3 in test_scenes:
        t = t3
    elif 5 in test_scenes:
        t = t5
    else:
        t = t6

    preds = np.where(Y_test == 1)[1]
    got_rights = np.where(preds == t)[0].shape[0]

    test_acc = got_rights / float(X_test.shape[0])
    return test_acc


# predict a label for each detection in X_test
def predict(X_test, shot, mu, scene_cast, C, test_scenes):
    N = mu.shape[0]
    L_test = np.zeros((X_test.shape[0], N)) + np.inf

    # initialize Y_test
    Y_test = np.zeros((X_test.shape[0], N))
    for i in range(Y_test.shape[0]):
        scene_id = int(X_test[i, 0])
        num_names = cast_counts[scene_id]
        rand_name_id = randint(0, num_names-1)
        person_id = scene_cast[scene_id][rand_name_id]
        Y_test[i, person_id] = 1

    # initialize the C for the test scenes
    C_ave = np.mean(C, axis=1)
    for test_scene in test_scenes:
        C[:, test_scene] = C_ave

    Y_test = E_step(Y_test, X_test, C, shot, mu, scene_cast, L_test)
    return Y_test



if __name__ == '__main__':

    use_kmeans = True

    prepend = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/'
    # prepend = '/u/eleni/412-project/the_mentalist_1x19/'

    dir_cast = prepend+'cast/'
    dir_feature_matrix = prepend+'feature_matrix_files/'
    # dir_feature_matrix = prepend+'feature_matrix_files/zero_thresh/'
    frames_dir = prepend+'scenes/scene_frames/'
    dir_shot_change = prepend+'shot_changes/'
    targets_dir = prepend+'labels/'

    out_path = '/Users/elenitriantafillou/model_output/'
    # out_path = '/u/eleni/model_output/'

    plot_path = out_path+'plot_files/'

    scene_cast, cast_counts, name_dict, S = load_data(dir_cast)
    t3, t5, t6 = load_labels(targets_dir)
    S = 40

    test_scenes = []
    X_train, X_test = construct_feature_matrix_train_test(dir_feature_matrix, test_scenes)


    max_frames_tr = np.max(X_train[:, 1]) + 1
    try:
        max_frames_t = np.max(X_test[:, 1]) + 1
    except:
        max_frames_t = 0
    max_frames = max(max_frames_tr, max_frames_t)

    shot_change = construct_shot_change(dir_shot_change, S, max_frames)
    N = len(name_dict.keys())

    # K-means initialization: uncomment for true k-means awesomeness in your faces
    if use_kmeans:
        classifier = KMeans(n_clusters=N, max_iter=100, precompute_distances=True)
        print 'K-Means Training'
        classifier.fit(X_train[:, 23:])
        y = classifier.predict(X_train[:, 23:])
        centers = classifier.cluster_centers_
        mu, Y_train, C, L_train = initialize(N, S, cast_counts, scene_cast, X_train, shot_change, centers, y) # Uncomment to use k-means init

    else:
        mu, Y_train, C, L_train = initialize(N, S, cast_counts, scene_cast, X_train, shot_change)


    Y_train, accs_train, accs_test = EM(Y_train, X_train, X_test, C, shot_change, mu, scene_cast, L_train, out_path, name_dict, t3, t5, t6, test_scenes, kmeans=use_kmeans)

    acc_tr_fname = plot_path + 'train_acc.txt'
    f = open(acc_tr_fname, 'w')
    for acc in accs_train:
        f.write(str(acc)+'\n')
    f.close()

    acc_t_fname = plot_path + 'test_acc.txt'
    f = open(acc_t_fname, 'w')
    for acc in accs_test:
        f.write(str(acc)+'\n')
    f.close()


    # test set
    # Y_test = predict(X_test, shot_change, mu, scene_cast, C, test_scenes)
    # test_acc = get_test_accuracy(Y_test, t3, t5, t6, test_scenes)
    # print "test acc for scene 5: " + str(test_acc)

    # for s in range(2, S):
    #     if use_kmeans:
    #         scene_out = out_path+'scene'+str(s)+'/kmeans_init_labels.txt'
    #     else:
    #         scene_out = out_path+'scene'+str(s)+'/labels.txt'
    #     f = open(scene_out, 'w')
    #     for y in range(Y.shape[0]):
    #         if X[y, 0] != s:
    #             continue
    #         label = np.where(Y[y, :] == 1)[0][0]
    #
    #         for (k, v) in name_dict.items():
    #             if label == v:
    #                 name = k
    #                 break
    #
    #         f.write(name+'\n')
