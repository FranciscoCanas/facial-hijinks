addpath('./detect/');
addpath('./extract/')
root_images_folder_path = '/u/eleni/doppia/data/frames/scene_';
root_results_folder_path = '~/face_detection_results/frames/scene_';
base_path = results_folder_path

images_folder_path = [root_images_folder_path int2str(scene) '/']
results_folder_path = [root_images_folder_path int2str(scene) '/']
fprintf(['Detecting Scene ' int2str(scene)])
run_detect
fprintf(['Extracting Scene ' int2str(scene)])
run_extract

