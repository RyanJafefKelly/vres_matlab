function [ proposalDensity ] = computeProposalDensity( proposedTheta, thetas, tuningParam )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    if ndims(proposedTheta > 1)
        proposalDensity = mean(mvnpdf(proposedTheta, thetas, tuningParam));
    else 
        proposalDensity = mean(normpdf(proposedTheta, thetas, tuningParam));
    end

end

