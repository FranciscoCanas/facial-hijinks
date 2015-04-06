init;
base_path='~/face_detection_results/frames/scene_4/'
img = 'image-035.jpg';
scene = 4;

fprintf('Extracting Features...\n');

image_names = dir(fullfile(base_path, '*.jpg'));

F = 1961; % Current number of features in use.
M = zeros(1,F);
frame = 1;

for i=1:numel(image_names)
    image_name = image_names(i).name;
    image_name;
    image_path = fullfile(base_path, image_name);
    ds_path = fullfile([base_path image_name '.vj']);
    T=dlmread(ds_path);
	

	[DETS,PTS,DESCS]=extfacedescs(opts, image_path, false);

	sz = size(DETS);
	N = sz(2);

	fprintf(' DETS: %d x %d\n',size(DETS,1),size(DETS,2));
	fprintf('  PTS: %d x %d x %d\n',size(PTS,1),size(PTS,2),size(PTS,3));
	fprintf('DESCS: %d x %d\n',size(DESCS,1),size(DESCS,2));

	SCENE = zeros(N,1) + scene;
	FRAME = zeros(N,1) + frame;

	if N > 0
		DETS=DETS';
		DESCS=DESCS';
		PTSX=reshape(PTS(1,:,:),[9,N])';
		PTSY=reshape(PTS(2,:,:),[9,N])';
		Cur_M = [SCENE FRAME DETS PTSX PTSY DESCS];
		M = [M ; Cur_M];
	end
	frame = frame + 1;
end

M = M(2:end,:);
fprintf('Size of Final Feature Matrix: %d x %d', size(M,1), size(M,2));
fprintf('Saving to M');
save('M', 'M', '-ascii');