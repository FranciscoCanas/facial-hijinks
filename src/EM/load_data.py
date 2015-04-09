import numpy as np
import os
__author__ = 'elenitriantafillou'



def construct_feature_matrix(in_dir, S):

    # M = np.loadtxt(in_dir+'M_scene3.txt')
    # print M.shape # (261, 1961)
    # X[:,:] = np.loadtxt(in_dir+'M_scene3.txt')

    # X = np.loadtxt(in_dir+'M_scene1.txt')
    # for scene_id in range(2,S):
    #     M = np.loadtxt(in_dir+'M_scene'+str(i)+'.txt')
    #     X = np.vstack((X, M))

    X = np.loadtxt(in_dir+'M_scene3.txt')
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

    print shot.shape
    return shot

def load_data(dir_cast):

    path_all_cast = dir_cast + 'episode_cast.txt'
    f = open(path_all_cast, 'r')

    # episode_cast = []
    name_dict = {}
    n = 0

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


def get_max_frames(frames_dir, S):

    frames_num = []

    for scene_id in range(1,S):
        path = frames_dir+'scene_'+str(scene_id)
        temp1, temp2, files = os.walk(path).next()
        num = len(files) - 1
        frames_num.append(num)

    return max(frames_num)


if __name__ == '__main__':

    dir_cast = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/cast/'
    dir_feature_matrix = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/feature_matrix_files/'
    frames_dir = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/scenes/scene_frames/'
    dir_shot_change = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/shot_changes/'

    #the former is a list and the latter a list of lists
    scene_cast, cast_counts, name_dict, S = load_data(dir_cast)
    # print len(cast_counts)

    max_frames = get_max_frames(frames_dir,S)
    print max_frames


    X = construct_feature_matrix(dir_feature_matrix, S)
    print X[:,1]

    shot_change = construct_shot_change(dir_shot_change, S, max_frames)
    # print shot_change