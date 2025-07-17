function [ priorDistribution ] = computePhiPrior( phi )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    priorDistribution = (exp(phi))/((exp(phi) + 1 )^2);

end