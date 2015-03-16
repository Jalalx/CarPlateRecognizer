%%
% This file loads all car plate images and call segmentation function to
% sengment all them into seperated files
% Developed by Jalal Amini Robaty, Shayan Asadpour

%%
% If directory does not exists, creates a new one.
if(~exist('frames', 'dir'))
    mkdir('frames');
end
files = dir('A');
filesCount = size(files);

for i = 1 : filesCount(:, 1)
    
    try
        % Segments all car plate images into seperated frames.
        frames = segmentation(['A\' num2str(i) '.jpg'], true);
        framesCount = size(frames);


        for j = 1 : framesCount
            % If frame was not empty write it to file.
            if(~isempty(frames{j}))
                imwrite(frames{j}, ...
                ['frames\' num2str(i) '-' num2str(j) '.jpg'], 'bmp');
            end
        end    
    catch ex
        ex
    end
end