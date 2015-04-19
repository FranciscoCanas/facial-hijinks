import numpy as np
import os
__author__ = 'elenitriantafillou'



def construct_feature_matrix_train_test(in_dir, test_scenes):

    # assumes that all scenes belong to training
    # except for those mentioned in list test_scenes

    temp1, temp2, files = os.walk(in_dir).next()
    num_scenes = len(files) - 1

    for i in range(2, num_scenes):

        fname = in_dir+'M'+str(i)+'.m'
        if not(os.path.isfile(fname)):
            continue

        if i in test_scenes:
            M_test = np.loadtxt(fname)
            if M_test.shape[0] == 0:
                continue

            try:
                X_test = np.vstack((X_test, M_test))
            except:
                X_test = M_test

        else:
            M_train = np.loadtxt(fname)
            if M_train.shape[0] == 0:
                continue

            try:
                X_train = np.vstack((X_train, M_train))
            except:
                X_train = M_train

    if test_scenes == []:
        X_test = 0

    return X_train, X_test



def construct_feature_matrix(in_dir):

    # M = np.loadtxt(in_dir+'M1.txt')
    # print M.shape # (261, 1961)
    # X[:,:] = np.loadtxt(in_dir+'M3.m')

    temp1, temp2, files = os.walk(in_dir).next()
    num_scenes = len(files) - 1

    X = np.loadtxt(in_dir+'M2.m')
    for i in range(3, num_scenes):

        fname = in_dir+'M'+str(i)+'.m'
        if not(os.path.isfile(fname)):
            continue

        M = np.loadtxt(fname)
        if M.shape[0] == 0:
            continue

        X = np.vstack((X, M))

    X[:, 1] -= 1
    return X


def construct_shot_change(dir_shot_changes, S, max_frames):

    shot = np.zeros((S,max_frames))
    for scene_id in range(S):
        fname = dir_shot_changes+'scene_'+str(scene_id)+'frame_ids.txt'
        if not(os.path.isfile(fname)):
            continue

        s = (np.loadtxt(fname)).reshape(1,-1)
        s = s.astype(int)

        shot[scene_id, s] = 1

    return shot

def load_data(dir_cast):

    path_all_cast = dir_cast + 'episode_cast.txt'
    f = open(path_all_cast, 'r')

    # episode_cast = []
    name_dict = {}

    name_dict['N.A.P'] = 0
    n = 1
    for line in f:
        # episode_cast.append(line[:-1])
        name_dict[line[:-1]] = n
        n += 1

    f.close()
    N = n

    # get number of scenes for episode
    temp1, temp2, files = os.walk(dir_cast).next()
    num_scenes = len(files) - 2

    scene_cast = []
    count_scene_cast = []
    scene_id = 0
    while scene_id < num_scenes:
        f = open(dir_cast+'present_cast_'+str(scene_id)+'.txt', 'r')
        present_cast=[]
        count = 0
        for line in f:
            name = line[:-1]
            present_cast.append(name_dict[name])
            count += 1
        scene_cast.append(present_cast)
        count_scene_cast.append(count)
        scene_id += 1

    return scene_cast, count_scene_cast, name_dict, num_scenes


def load_labels(targets_dir):
    t3 = np.loadtxt(targets_dir + 'scene3_labels.txt')
    t5 = np.loadtxt(targets_dir + 'scene5_labels.txt')
    t6 = np.loadtxt(targets_dir + 'scene6_labels.txt')

    return t3, t5, t6

if __name__ == '__main__':
    prepend = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/'

    dir_cast = prepend+'cast/'
    feature_dir = prepend+'feature_matrix_files/'
#
    scene_cast, count_scene_cast, name_dict, num_scenes = load_data(dir_cast)
#     construct_feature_matrix(feature_dir)
    targets_dir = prepend+'labels/'
    t3, t5, t6 = load_labels(targets_dir)

    test_scenes = [5]
    X_train, X_test = construct_feature_matrix_train_test(feature_dir, test_scenes)
    print name_dict