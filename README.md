# 64-QAM Classification
Optical Communications: 64-QAM Classification with Neural Networks

## Description 

This project provides an overview of a Machine Learning (ML) approach for dealing with the interferences present in the optical communication channels.
Focusing on the Quadrature Amplitude Modulation (QAM) scheme, the strategy is to consider a signal stream of symbols as a classification problem and to create an Autonomous Neural Network (ANN) capable of predicting the class of each symbol.
Also, it gives an insight into possible further research strategies aiming to improve the success rate and accuracy of the predictions.

The validation score was 0.89, which was below the expectations, since any classifier with an accuracy under 99% is not welcomed in the world of optical communications.
We gather that the assumption that other symbols on the signal stream could influence the class of the symbol being classified might be partially incorrect, or at least their weights may be of little effect.
It is quite possible that another source of information is influencing more the current symbol and that it should be considered by the ANNs for a better predicition.

## Repository Structure 

/docs - contains the written paper on the conducted analysis and the respective presentation

/results - contains outputs produced by the program

/src - contains the source code written in Matlab

## Additional Resources

![constellation](https://github.com/FilipePires98/64-QAM-Classification/blob/main/docs/paper/figure2.jpg)

Constellation diagram for rectangular 64-QAM transmitted (red crosses) and received (blue dots) symbols, using 10% of the collected data.

![architecture](https://github.com/FilipePires98/64-QAM-Classification/blob/main/docs/paper/figure4.jpg)

Visual representation of the ANNs.

## Authors

The authors of this repository are Filipe Pires and João Alegria, and the project was developed for the Machine Learning Course of the licenciate's degree in Informatics Engineering of the University of Aveiro.

For further information, please read our [paper](https://github.com/FilipePires98/64-QAM-Classification/blob/main/docs/paper/Paper.pdf) or contact us at filipesnetopires@ua.pt or joao.p@ua.pt.



