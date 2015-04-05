addpath('./CLASS');
init;
path='~/face_detection_results/frames/scene_4/image-035.jpeg';
fprintf('running from cached face detections...\n');

[DETS,PTS,DESCS]=extfacedescs(opts,path,true);

fprintf(' DETS: %d x %d\n',size(DETS,1),size(DETS,2));
fprintf('  PTS: %d x %d x %d\n',size(PTS,1),size(PTS,2),size(PTS,3));
fprintf('DESCS: %d x %d\n',size(DESCS,1),size(DESCS,2));

if strcmp(computer,'PCWIN')

    pause;
    
    fprintf('running using face detector binary...\n');

    I=imread('047640.jpg');
    
    [DETS,PTS,DESCS]=extfacedescs(opts,I,true);
    
	fprintf(' DETS: %d x %d\n',size(DETS,1),size(DETS,2));
	fprintf('  PTS: %d x %d x %d\n',size(PTS,1),size(PTS,2),size(PTS,3));
	fprintf('DESCS: %d x %d\n',size(DESCS,1),size(DESCS,2));
end
