%% Machine Learning Project: 64-QAM Classification
% Authors:	Filipe Pires (85122) and Jo�o Alegria (85048)
% Date:     November 2018

%% Initialization
clear all; close all; clc

%% Setup the NN parameters
input_layer_size  =  [2 3 4 5];      % feature mappings of the main feature
hidden_layer_size = [10]; %[3 8 15 24];	% hidden units
num_labels        = 8;              % 8 labels, each NN (Re and Im) will have 8 labels (8x8=64)

lambda = [0]; %[0 0.01 0.3];
options = optimset('MaxIter', 5); % 100 iterations

%% Part 1: Loading and Visualizing Data
fprintf('Loading and Visualizing Data ...\n')
load('data.mat')
%IQmap = round(IQmap,4); % Stx = sent QAM data
Stx = round(Stx,4);     % Srx = noisy signals at the receiver
Srx = round(Srx,4);     % IQmap = expected results

constellationDiagram(Srx,Stx,0.1) % visualizing 10% of all the data

divT = 0.01; % training data = 60% of the entire data available
divV = 0.01; % validation data = 20% of the entire data available
SrxT = Srx(1,1:ceil(divT*length(Srx)));
StxT = Stx(1,1:ceil(divT*length(Stx)));
SrxV = Srx(1,ceil(divT*length(Srx)):ceil((divT+divV)*length(Srx)));
StxV = Stx(1,ceil(divT*length(Stx)):ceil((divT+divV)*length(Stx)));

%% Part 2: Finding the Best Combination of Hyperparameters for each NN

for i=1:length(input_layer_size)
    for h=1:length(hidden_layer_size)
        for l=1:length(lambda)
            [Theta1Re, Theta2Re, Theta1Im, Theta2Im, errorRe, errorIm, acR, acI, accuracy] = QAM_Classification(input_layer_size(i), hidden_layer_size(h), num_labels, lambda(l), options, IQmap, SrxT, StxT, SrxV, StxV);
            fprintf('\nValidation Set - Accuracy: %.3f%%; Error (Re): %.3f; Error (Im): %.3f.\n', accuracy, errorRe, errorIm);
            filename = strcat('res_', int2str(input_layer_size(i)), '_', int2str(hidden_layer_size(h)), '_', int2str(l), '.mat');
            %save(filename, 'Theta1Re', 'Theta2Re', 'Theta1Im', 'Theta2Im', 'errorRe', 'errorIm', 'acR', 'acI', 'accuracy');
        end
    end
end
