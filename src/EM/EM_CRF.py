from load_data import *
from random import randint


def initialize(K, N, S, cast_counts, scene_cast, X):

    mu = (np.random.random((K+1))) # cluster means
    # log_mu = np.log(mu)

    Y = np.zeros((X.shape[0],N+1))

    # assignment of names (clusters) to detections
    # (respects the scene weak label constraint)
    for i in range(X.shape[0]):
        scene_id = int(X[i,0])
        num_names = cast_counts[scene_id]
        rand_name_id = randint(0,num_names-1)

        person_id = scene_cast[scene_id][rand_name_id]
        Y[i,person_id] = 1

    C = np.random.random((3, S))

    return mu, Y, C



def name_predictions(Y, name_dict):
    names = []
    for y in Y:
        ind = np.nonzero(y)[0][0]
        names.append(name_dict[ind])
    return names


if __name__ == '__main__':
    dir_cast = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/cast/'
    dir_feature_matrix = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/feature_matrix_files/'
    frames_dir = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/scenes/scene_frames/'
    dir_shot_change = '/Users/elenitriantafillou/research_ML/the_mentalist_1x19/shot_changes/'

    scene_cast, cast_counts, name_dict, S = load_data(dir_cast)
    max_frames = get_max_frames(frames_dir,S)
    print max_frames
    # num_feature_cols = 1961

    N = len(name_dict.keys())

    X = construct_feature_matrix(dir_feature_matrix, S)
    # print X[:,1]

    shot_change = construct_shot_change(dir_shot_change, S, max_frames)

    initialize(N+1, N, S, cast_counts, scene_cast, X)