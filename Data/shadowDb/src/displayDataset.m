%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% displayDataset.m
%  Demonstration file for the shadow database. Use this as a starting
%  point, and edit to your liking!
%
% Copyright 2006-2010 Jean-Francois Lalonde
% Carnegie Mellon University
% Consult the LICENSE.txt file for licensing information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% replace this with your own path
datasetBasePath = '/nfs/hn26/jlalonde/results/shadowDetection/onlineDb';

% needs these files to load XML files
addpath('./load_xml');

imgPath = fullfile(datasetBasePath, 'img');
xmlPath = fullfile(datasetBasePath, 'xml');

files = dir(fullfile(imgPath, '*.jpg'));
fh = figure;

fprintf('Found %d images\n', length(files));

for f=1:length(files)
    % load image and xml file
    img = imread(fullfile(imgPath, files(f).name));
    shadowInfo = load_xml(fullfile(xmlPath, strrep(files(f).name, '.jpg', '.xml')));
    
    % load (x,y) coordinates
    xCoords = arrayfun(@(p) str2double(p.x), shadowInfo.shadowCoords.pt);
    yCoords = arrayfun(@(p) str2double(p.y), shadowInfo.shadowCoords.pt);
    
    % display boundaries
    figure(fh); hold off; imshow(img); hold on;
    plot(xCoords, yCoords, '.r', 'MarkerSize', 5);
    drawnow;
end
