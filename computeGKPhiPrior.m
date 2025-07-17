function [ priorDistribution ] = computeGKPhiPrior( proposed_phi )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    a = 10;
    b = 5;
    c = 10;
    d = 5;
    
    priorDistribution = a * b * c * d *  prod((exp(proposed_phi))./((exp(proposed_phi) + 1 ).^2));

end

