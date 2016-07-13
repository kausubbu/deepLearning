function patches = sampleIMAGES(patchsize, numpatches)
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns.
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data
%  from IMAGES.
%
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1

for i=1:numpatches
    
    % Sample a random image
    imageID = randi(size(IMAGES,3));
    
    % Sample random indices
    limitX = size(IMAGES, 1);
    limitY = size(IMAGES, 2);
    patchX = randi([1 limitX-patchsize], 1);
    patchY = randi([1 limitY-patchsize], 1);
    
    % Get the sample patch
    sampledPatch = IMAGES(patchX:patchX+patchsize-1, patchY:patchY+patchsize-1, imageID);
    
    % Update the patches variable
    patches(:,i) = sampledPatch(:);
    
end

%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end