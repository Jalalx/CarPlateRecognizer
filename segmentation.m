%%
% This program reads a car plate image and seperate all characters within
% into a cell array of images.
% Programmers: Jalal Amini Robaty, Shayan Asadpour
%%

% imagePath: Specify car plate image path
% suppressFigures: if set TRUE, figures will not be shown, otherwise shown.
% frames: A cell array contains all characters inside car plate.
function [frames] = segmentation(imagePath, suppressFigures)    
    
    % Creates 8 element of cells.
    frames = cell(1, 8);
    
    if(~suppressFigures)
        figure('Name', 'Image');
    end
    
    % read and resize image.
    I = imread(imagePath);
    I = imresize(I, [140 710]);
    
    if(~suppressFigures)
        subplot(2, 1, 1);
        imshow(I);
    end
    
    
    I = rgb2gray(I);
    
    I = im2bw(I, graythresh(I));
    
    I = ~I;
    I = imclearborder(I);
    
    I = bwareaopen(I, 60);
    if(~suppressFigures)
        subplot(2, 1, 2);
        imshow(I);
    end
    
    % Find all connected components in image and set max in L_max. this
    % fuction groups all components into groups of 1 to L_max.
    L = bwlabel(I, 4);
    L_max = max(max(L)); 

    if(~suppressFigures)
        figure('Name', 'Segmentation Result');    
    end
    
    plotIndex = 1;
    for i = 1 : L_max 
        
        % Now we should find top, left, width and height of each compnent.
        [r, c] = find(L == i);         

        min_y = min(r); 
        min_x = min(c); 

        min_w = max(r) - min(r); 
        min_h = max(c) - min(c); 

        % Crop component from image and put it in A.
        A = imcrop(I, [min_x min_y min_h min_w]);

        [r ,c] = size(A);
        if(r > 40 && r < 120 && c > 15 && c < 130)
            if(~suppressFigures)
                subplot(2, 5, plotIndex);
                imshow(A);
            end
            
            frames{plotIndex} = A;
            plotIndex = plotIndex + 1;
        end
    end
end