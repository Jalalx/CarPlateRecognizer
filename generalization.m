function generalization(hiddenWeights, hiddenLayerBias, outputWeights, ...
        outputLayerBias, characters, S)
    %% Generalization Phase
        
    % Load samples    
    charTypeCount = 23;    
    generalizationOutputs = zeros(charTypeCount, 1);
    imageSize = 400;
    
    counter = 0;
    
    
    for i = 1 : 8
        if(~isempty(S{1, i}))
            sampleChar = S{1, i};
            Iy = hiddenWeights * sampleChar + hiddenLayerBias;
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
            if (isequal(i, f))
                 counter = counter + 1.0 ;                
            end
            generalizationOutputs(i, 1) = find(O, 1);
        end
    end
       
    
    % Display generalization result
    figure('Name', 'Generalization Result');
    for i = 1 : 8        
        if(~isempty(S{1, i}))
            sampleChar = S{1, i};

            subplot(2, 4, i);
            image = reshape(sampleChar, sqrt(imageSize), sqrt(imageSize));
            imshow(image, []);        

            id = generalizationOutputs(i, 1);

            titleId = characters{2, id};
            title(titleId);
        else
            subplot(2, 4, i);
            imshow(ones(20, 20));
            title('Error');
        end
    end

    
    
%     percentage = counter / charTypeCount * 100.0;                      
%     set(gcf, 'name',['Generalization Result : '...
%     num2str(percentage) '% Correct'], 'numbertitle', 'off');

    
  


end

