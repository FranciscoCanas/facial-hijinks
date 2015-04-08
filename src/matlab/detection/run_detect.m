% Add the voc-dpm repo to path:
addpath('./voc-dpm');
startup
% This script will detect faces in a set of images
% The minimum face detection size is 36 pixels,
% the maximum size is the full image.

model_path = '/u/eleni/doppia/data/trained_models/face_detection/dpm_baseline.mat';
face_model = load(model_path);

% lower detection threshold generates more detections
%detection_threshold = -0.5; 
detection_threshold = 0; 

% 0.3 or 0.2 are adequate for face detection.
nms_threshold = 0.3;

image_names = dir(fullfile(images_folder_path, '*.jpg'));

count = 0;

for i=1:numel(image_names)
    image_name = image_names(i).name;
    image_name;
    image_path = fullfile(images_folder_path, image_name);
    image = imread(image_path);

    % ds: (x1, y1, x2, y2, model, score) 
    [ds, bs] = process_face(image, face_model.model, detection_threshold, nms_threshold);
    ds_size = size(ds);
    if ds_size(1) > 0
        dsp = ds(:,[1,3,2,4]);
    end
    %compute union intersections:
    %intersections = rectint(dsp, dspm);
	
    bimage_path = fullfile(results_folder_path, ['bounded/', image_name]);
    results_path = fullfile(results_folder_path,[image_name, '.vj']);

    showsboxes_face(image, ds, bimage_path);
    file = fopen(results_path, 'w');
    fprintf(' DETS in %s: %d\n', image_name, ds_size(1));
    fprintf(file, '%d\n', ds_size(1));
    if ds_size(1) > 0
    	for i=1:ds_size(1)
                count = count + 1;
    	        fprintf(file, '%d %d %d %d\n', round(dsp(i,:)));
    	end
    end
    disp(['Created ', results_path]);
end

disp('All images processed');