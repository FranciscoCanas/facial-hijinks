addpath('./detection/');
addpath('./extraction/')
root_images_folder_path = '/u/eleni/doppia/data/frames/scene_';
root_results_folder_path = '~/face_detection_results/frames/scene_';


images_folder_path = [root_images_folder_path int2str(scene) '/'];
results_folder_path = [root_results_folder_path int2str(scene) '/'];
base_path = results_folder_path;
fprintf(['Detecting Scene ' int2str(scene)])
cd('./detection/')
run_detect
fprintf(['Extracting Scene ' int2str(scene)])
cd('../extraction/')
run_extract

