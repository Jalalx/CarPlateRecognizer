%%
%   Character Recognition Project version 1.0.0
%   NEURAL NETWORK Multi Layer Perceptron 
%   COPYRIGHT (c) 2010 - 2011 
%   Programmed by: Jalal Amini Robaty, Shayan Asadpour 
%	Master: Mrs. Abnavi
%%
function [id] = mlp(char, epoch, suppressFigures)
 
    %epoch = 1;
    learningRate = 0.075;     
    momentum = 0.8;
    
    
    
    
    charTypeCount = 23;
    characters = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23;
                 '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 't', 'te', 's', 'gh', 'n', 'h', 'j', 'v', 'y', 'l', 'm'};
    
    imageRows = 20;
    imageCols = 20;
    imageSize = imageRows * imageCols;
    
    
    
    charTypeLearnCount = 3;
    totalInputCount = charTypeCount * charTypeLearnCount;
    hiddenLayerNeuronCount = 280;
    
    S = zeros(imageSize, totalInputCount);
    
    
    outputWeights = zeros(charTypeCount, hiddenLayerNeuronCount) ;
    hiddenWeights = rand(hiddenLayerNeuronCount, imageSize) / 10  - 0.05 ;
    
  
    hiddenLayerBias = ones(hiddenLayerNeuronCount, 1);
    outputLayerBias = ones(charTypeCount, 1);
    testOutputs = zeros(totalInputCount, 1);
    trainOutputs = zeros(totalInputCount, 1);
    
    
    outputMatrix = zeros(charTypeCount, totalInputCount);
    
    % Load samples
    for j = 1 : charTypeLearnCount % = 3
        for i = 1 : charTypeCount % = 21
            path = ['20x20/' characters{2, i} '-' int2str(j) '.jpg'];   
            image = imread(path);
            image = imresize(image, [20 20]);
            image = imresize(image, [imageRows imageCols]);
            image = ~image;
            col = reshape(image, imageSize, 1);  
            S(:, ((j - 1) *  charTypeCount) + i) = col;
        end
    end
    
        
    % Initialize Target matrix
    T = zeros(charTypeCount, charTypeCount);
    for i = 1 : charTypeCount
        T(i, i) = 1;
    end

    [row col] = size(S); %#ok<NASGU>
    

    while epoch > 0
        counter = 0 ; 
        % Training Phase
        for i = 1 : totalInputCount
            Iy = hiddenWeights * S(:, i) + hiddenLayerBias;
            % Sigmoid function for hidden layer
            Y = tansig(Iy);
            
    
            Io = outputWeights * Y + outputLayerBias;
            O =  purelin(Io); 
            [c, f] = max (O);
            O(f, 1) = 1;
            for g = 1 : charTypeCount
                if g ~= f
					if O(g, 1) <= c
						O(g, 1) = 0;
					end
                end
            end
			
            
            x = fix(mod(i, charTypeCount));
            if(x == 0)
                x = charTypeCount;
            end
            
            thetao = T(:, x) - O ;
            thetay = outputWeights' * thetao;
            
       
            
            
            
            
            outputLayerBias = outputLayerBias + learningRate * thetao;
            hiddenLayerBias = hiddenLayerBias + learningRate * thetay;
            
            hiddenOldWeight = outputWeights;
            outputWeights = outputWeights + learningRate * thetao * Y'; 
            hiddenweightChange = outputWeights - hiddenOldWeight;
            outputWeights = outputWeights + (hiddenweightChange * momentum);
            
            outputOldWeight = hiddenWeights;
            hiddenWeights = hiddenWeights + learningRate * thetay * S(:, i)';
            outputWeightChange = hiddenWeights - outputOldWeight;
            hiddenWeights = hiddenWeights + (outputWeightChange * momentum);
            
            
            outputMatrix(:, i) = O;
            trainOutputs(i, 1) = find(O, 1);            
        end
        
        if (isequal(T, outputMatrix))
            epoch = 0;
        end
        
        
        
        %Test Phase
        for i = 1 : totalInputCount
            
            Iy = hiddenWeights * S(:, i) + hiddenLayerBias;
            Y  = tansig(Iy);

            Io = outputWeights * Y + outputLayerBias;
            O =  purelin(Io); 
            
            [c, f] = max (O);
            O(f, 1)=1;
            for g = 1 : charTypeCount
                if g ~= f
                    if O(g, 1) <= c
                        O(g, 1) = 0;
                    end
                end
            end
             
            testOutputs(i, 1) = find(O, 1);
        end
        
        epoch = epoch - 1;
        epoch
    end
    
    if(~suppressFigures)
        %% Display training result    
        for i = 1 : totalInputCount
            x = fix(mod(i, charTypeCount));
            if(x == 0)
                x = charTypeCount;
            end



            if(x == 1)
                figure('Name', 'Training Result');
            end
            subplot(4, 6, x);
            image = reshape(S(:, i), sqrt(imageSize), sqrt(imageSize));
            imshow(image, []);



            subplot(4, 6, x);
            id =  trainOutputs(i, 1);
            titleId = characters{2, id };
            title(titleId);
        end

        %% Display test result
        for i = 1 : totalInputCount
           x = fix(mod(i, charTypeCount));
            if(x == 0)
                x = charTypeCount;
            end

            if(x == 1)
                figure('Name', 'Test Result');
            end

            subplot(4, 6, x);
            image = reshape(S(:, i), sqrt(imageSize), sqrt(imageSize));
            imshow(image, []);

            subplot(4, 6, x);
            id =  testOutputs(i, 1);

            titleId = characters{2, id };
            title(titleId);
        end
    end
    
    %% Generalization Phase
    % Load samples
    Iy = hiddenWeights * char + hiddenLayerBias;
    Y  = tansig(Iy);

    Io = outputWeights * Y + outputLayerBias;
    O =  purelin(Io); 




    [c, f] = max(O);
    O(f, 1) = 1;
    for g = 1 : charTypeCount
        if g ~= f
            if O(g, 1) <= c
                O(g, 1) = 0;
            end

        end
    end              

    id = find(O, 1);

    
    % Display generalization result
    figure('Name', 'Generalization Result');  
    subplot(1, 1, 1);
    image = reshape(char, sqrt(imageSize), sqrt(imageSize));
    imshow(image, []);        

    titleId = characters{2, id};
    title(titleId);

%     percentage = counter / charTypeCount * 100.0;                      
%     set(gcf, 'name',['Generalization Result : '...
%     num2str(percentage) '% Correct'], 'numbertitle', 'off');

    
    pause;


end


 
