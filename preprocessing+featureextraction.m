%% Glaucoma Feature Extraction and Saving to CSV

% Select image file interactively
[file, path] = uigetfile({'*.png;*.jpg;*.jpeg', 'Image Files (*.png, *.jpg, *.jpeg)'}, 'Select an Image', 'MultiSelect', 'on');

if isequal(file, 0)
    error('No image selected. Exiting.');
end

if ischar(file)  
    file = {file}; % Convert single file to cell array for consistency
end

allFeatures = [];  

for i = 1:length(file)
    % Read Image
    img = imread(fullfile(path, file{i}));

    % Convert to grayscale if needed
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end

    % Apply Filters
    filteredImg = imgaussfilt(grayImg, 1); % Gaussian filter
    filteredImg = medfilt2(filteredImg, [3 3]); % Median filter
    enhancedImg = adapthisteq(filteredImg); % Adaptive Histogram Equalization

    % ROI Extraction: Segment Optic Disc & Cup
    [discMask, cupMask] = segmentOpticDiscCup(enhancedImg);

    % Feature Extraction
    structural = extractEyeStructuralFeatures(discMask, cupMask);
    statistical = extractStatisticalFeatures(enhancedImg);
    texture = extractTexturalFeatures(enhancedImg);

    % Combine Features
    featureVector = [structural, statistical, texture];
    allFeatures = [allFeatures; featureVector];
end

% Define Feature Names
featureNames = {'CupToDiscRatio', 'OpticDiscArea', 'OpticCupArea', 'RimThickness',
                'MeanIntensity', 'StdIntensity', 'Skewness', 'Kurtosis',
                'Contrast', 'Correlation', 'Energy', 'Homogeneity'};

% Ask user where to save the file
[saveFile, savePath] = uiputfile('*.csv', 'Save Feature Data As');
if isequal(saveFile, 0)
    disp('No file selected. Exiting without saving.');
    return;
end

% Save features to the selected file
featureTable = array2table(allFeatures, 'VariableNames', featureNames);
writetable(featureTable, fullfile(savePath, saveFile));

disp(['Feature extraction complete. Data saved to ', fullfile(savePath, saveFile)]);

%% Supporting Functions

function [discMask, cupMask] = segmentOpticDiscCup(img)
    % Dummy segmentation using thresholding (Replace with actual segmentation)
    level = graythresh(img);
    discMask = imbinarize(img, level);
    
    % Assume the cup is a subset of the disc
    cupMask = imerode(discMask, strel('disk', 10)); % Reduce size to simulate optic cup
end

function structural = extractEyeStructuralFeatures(discMask, cupMask)
    % Compute properties
    discStats = regionprops(discMask, 'Area', 'BoundingBox');
    cupStats = regionprops(cupMask, 'Area', 'BoundingBox');

    if isempty(discStats) || isempty(cupStats)
        structural = [0, 0, 0, 0]; % Handle case when segmentation fails
        return;
    end

    % Extract Areas
    opticDiscArea = discStats(1).Area;
    opticCupArea = cupStats(1).Area;
    
    % Compute Cup-to-Disc Ratio (CDR)
    cupToDiscRatio = opticCupArea / opticDiscArea;
     
    % Compute Rim Thickness as difference in bounding box sizes
    discRadius = mean(discStats(1).BoundingBox(3:4)) / 2;
    cupRadius = mean(cupStats(1).BoundingBox(3:4)) / 2;
    rimThickness = discRadius - cupRadius;

    structural = [cupToDiscRatio, opticDiscArea, opticCupArea, rimThickness];
end

function stats = extractStatisticalFeatures(img)
    % Compute intensity-based statistical features
    stats = [mean(img(:)), std(double(img(:))), skewness(double(img(:))), kurtosis(double(img(:)))];
end

function texture = extractTexturalFeatures(img)
    glcm = graycomatrix(img);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    texture = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
end
