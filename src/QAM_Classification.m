function [Theta1Re, Theta2Re, Theta1Im, Theta2Im, errorRe, errorIm,acR, acI, accuracy] = QAM_Classification(input_layer_size, hidden_layer_size, num_labels, lambda, options, IQmap, SrxT, StxT, SrxV, StxV)
    %%QAM_CLASSIFICATION Implements the creation, training and validation
    % of 2 Neural Network (for the Real and Imaginary values), given the 
    % hyperparameters and number of iterations.
    % The return values are the Thetas, Errors and Accuracy of the NN.
    
    %% Part 1: Initializing NN Pameters
    fprintf('\nInitializing Neural Network Parameters ...\n')
    
    % Extract x and y
    xRe = real(SrxT');
    xIm = imag(SrxT');
    yRe = real(StxT);
    yIm = imag(StxT);

    % Map y to index type values
    uyr = unique(real(IQmap));
    uyi = unique(imag(IQmap));
    
    
    for i=1:length(yRe)
        for j=1:length(uyr)
            if round(yRe(i),4)==round(uyr(j),4)
                yRe(i) = j;
                break
            end
        end
        for j=1:length(uyi)
            if round(yIm(i),4)==round(uyi(j),4)
                yIm(i) = j;
                break
            end
        end
    end
    

    %lambda=0;
    %input_layer_size=4;
    % Features to be considered
    if input_layer_size == 2
        featuresReT = [ones(length(xRe(:,1)),1) xRe];
        featuresImT = [ones(length(xIm(:,1)),1) xIm];
    end
    if input_layer_size == 3
        featuresReT = [ones(length(xRe(2:end,1)),1) xRe(2:end,1) xRe(1:end-1,1)];
        featuresImT = [ones(length(xIm(2:end,1)),1) xIm(2:end,1) xIm(1:end-1,1)];
    end
    if input_layer_size == 4
        featuresReT = [ones(length(xRe(3:end,1)),1) xRe(3:end,1) xRe(2:end-1,1) xRe(1:end-2,1)];
        featuresImT = [ones(length(xIm(3:end,1)),1) xIm(3:end,1) xIm(2:end-1,1) xIm(1:end-2,1)];
    end
    if input_layer_size == 5
        featuresReT = [ones(length(xRe(4:end,1)),1) xRe(4:end,1) xRe(3:end-1,1) xRe(2:end-2,1) xRe(1:end-3,1)];
        featuresImT = [ones(length(xIm(4:end,1)),1) xIm(4:end,1) xIm(3:end-1,1) xIm(2:end-2,1) xIm(1:end-3,1)];
    end

    % 2 layer ANN to classify digits
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    
    
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
    
    %load 'res_4_10_0.mat'
    
    %for the last one
    %initial_nn_params = [Theta1Re(:); Theta2Re(:)];

    %% Part 2: Training the NNs
    
    % Real
    fprintf('\nTraining Neural Network for Real Values ... \n')
    costFunctionRe = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, featuresReT, yRe, lambda);
    [nn_params, cost] = fmincg(costFunctionRe, initial_nn_params, options);
    Theta1ReF = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2ReF = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
    %figure(2), subplot 121, plot(Theta1,'b.'), title('Theta1'), subplot 122, plot(Theta2,'b.'), title('Theta2')
    
    
    
    
    
    
    %lambda=0.3;
    %input_layer_size=5;
    % Features to be considered
    if input_layer_size == 2
        featuresReT = [ones(length(xRe(:,1)),1) xRe];
        featuresImT = [ones(length(xIm(:,1)),1) xIm];
    end
    if input_layer_size == 3
        featuresReT = [ones(length(xRe(2:end,1)),1) xRe(2:end,1) xRe(1:end-1,1)];
        featuresImT = [ones(length(xIm(2:end,1)),1) xIm(2:end,1) xIm(1:end-1,1)];
    end
    if input_layer_size == 4
        featuresReT = [ones(length(xRe(3:end,1)),1) xRe(3:end,1) xRe(2:end-1,1) xRe(1:end-2,1)];
        featuresImT = [ones(length(xIm(3:end,1)),1) xIm(3:end,1) xIm(2:end-1,1) xIm(1:end-2,1)];
    end
    if input_layer_size == 5
        featuresReT = [ones(length(xRe(4:end,1)),1) xRe(4:end,1) xRe(3:end-1,1) xRe(2:end-2,1) xRe(1:end-3,1)];
        featuresImT = [ones(length(xIm(4:end,1)),1) xIm(4:end,1) xIm(3:end-1,1) xIm(2:end-2,1) xIm(1:end-3,1)];
    end
    
    %load 'res_5_10_03.mat'
    
    %for the last one
    %initial_nn_params = [Theta1Im(:); Theta2Im(:)];
    initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
    
    
    
    % Imaginary
    fprintf('Training Neural Network for Imaginary Values ... \n')
    costFunctionIm = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, featuresImT, yIm, lambda);
    [nn_params, cost] = fmincg(costFunctionIm, initial_nn_params, options);
    Theta1ImF = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
    Theta2ImF = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
    %figure(3), subplot 121, plot(Theta1,'b.'), title('Theta1'), subplot 122, plot(Theta2,'b.'), title('Theta2')

    Theta1Im=Theta1ImF;
    Theta2Im=Theta2ImF;
    Theta1Re=Theta1ReF;
    Theta2Re=Theta2ReF;

    
    %load res_final.mat
    %% Part 3: Decode and Evaluate Results (Errors)
    fprintf('\nCalculating errors and accuracy ... \n')
    
    xRe = real(SrxV');
    yRe = real(StxV);
    xIm = imag(SrxV');
    yIm = imag(StxV);
    
    %input_layer_size=4;
    if input_layer_size == 2
        featuresReV = [ones(length(xRe(:,1)),1) xRe];
    end
    if input_layer_size == 3
        featuresReV = [ones(length(xRe(2:end,1)),1) xRe(2:end,1) xRe(1:end-1,1)];
    end
    if input_layer_size == 4
        featuresReV = [ones(length(xRe(3:end,1)),1) xRe(3:end,1) xRe(2:end-1,1) xRe(1:end-2,1)];
    end
    if input_layer_size == 5
        featuresReV = [ones(length(xRe(4:end,1)),1) xRe(4:end,1) xRe(3:end-1,1) xRe(2:end-2,1) xRe(1:end-3,1)];
    end

    %for the final one

    
    predRe = predict(Theta1Re, Theta2Re, featuresReV);
    
    %input_layer_size=5;
    if input_layer_size == 2
        featuresImV = [ones(length(xIm(:,1)),1) xIm];
    end
    if input_layer_size == 3
        featuresImV = [ones(length(xIm(2:end,1)),1) xIm(2:end,1) xIm(1:end-1,1)];
    end
    if input_layer_size == 4
        featuresImV = [ones(length(xIm(3:end,1)),1) xIm(3:end,1) xIm(2:end-1,1) xIm(1:end-2,1)];
    end
    if input_layer_size == 5
        featuresImV = [ones(length(xIm(4:end,1)),1) xIm(4:end,1) xIm(3:end-1,1) xIm(2:end-2,1) xIm(1:end-3,1)];
    end
    
    
    predIm = predict(Theta1Im, Theta2Im, featuresImV);
    
    % decoding results
    for i=1:length(predRe)
        predRe(i) = uyr(predRe(i));
    end
    for i=1:length(predIm)
        predIm(i) = uyi(predIm(i));
    end
    
    
    if input_layer_size == 2
        m = length(xRe);
    end
    if input_layer_size == 3
        m = length(xRe(1:end-1));
        yRe=yRe(1:end-1);
        yIm=yIm(1:end-1);
    end
    if input_layer_size == 4
        m = length(xRe(1:end-2));
        yRe=yRe(1:end-2);
        yIm=yIm(1:end-2);

    end
    if input_layer_size == 5
        m = length(xRe(1:end-3));
        yRe=yRe(1:end-3);
        yIm=yIm(1:end-3);
    end
    
    
    predRe=round(predRe,4);
    yRe=round(yRe,4);
    predIm=round(predIm,4);
    yIm=round(yIm,4);
    

    %errorRe = (predRe-yRe)*(predRe-yRe)';
    errorRe=0;
    errorIm=0;
    %errorIm = (predIm-yIm)*(predIm-yIm)';
    acR = mean(double(predRe==yRe)) * 100
    acI = mean(double(predIm==yIm)) * 100
    accuracy = mean(double(complex(predRe,predIm)==StxV(1:end-input_layer_size+2))) * 100;
    
    
    %a=predRe(1:end-1);
    %b=StxV(1:end-3);
    %c=yRe(1:end-2);
    %d=yIm(1:end-3);
    %save xy.mat a predIm c d b
    
    
    %hold on;plot([1:100],((a(1:100)==c(1:100))*3),'xr');plot([1:100],((predIm(1:100)==d(1:100))*2),'xb'); plot([1:100],aux,'xk');axis([0 101 0 4]); line([1:100;1:100], [ones(1,100);ones(1,100)*3],'LineStyle',':', 'Color','k');legend('Real Prediction', 'Imaginary Prediction', 'Total Prediction')
end