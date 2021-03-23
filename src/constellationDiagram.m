%% Constellation Diagram (Display Data Function)
function [] = constellationDiagram(Srx, Stx, division)
%CONSTELLATIONDIAGRAM represents the signal modulated by the QAM scheme.
%
%   CONSTELLATIONDIAGRAM(Srx, Stx, division) displays the signal as a 
%   two-dimensional xy-plane scatter diagram in the complex plane. 
%   The number of constellation points in a diagram gives the size of the 
%   "alphabet" of symbols that can be transmitted by each sample, and so 
%   determines the number of bits transmitted per sample.
%
%   The red Xs represent the sent QAM data, whereas the blue dots
%   represent the received data.
%
    %use only 'division' percent of the original data (e.g. 0.05)
    Srx = Srx(1,1:ceil(division*length(Srx)));
    Stx = Stx(1,1:ceil(division*length(Stx)));

    figure(1), plot(Srx,'b.'), hold on, plot(Stx, 'xr', 'MarkerSize', 10)
    axis([-1 1 -1 1])
    title('64QAM'), legend('received QAM data', 'sent QAM data')
end