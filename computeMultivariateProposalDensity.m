function [ proposalDensity ] = computeMultivariateProposalDensity( proposedTheta, tuningParam, thetas )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% mvnpdf

proposalDensity = mean(mvnpdf(proposedTheta, thetas, tuningParam));

% sum = 0;
% for i = 1:N_a
%     if abs(proposedTheta - thetas(i)) < 10*tuningParam  %reduce computation time (does not compute normpdf when it will give a value close to 0)
%         sum = sum + normpdf(proposedTheta, thetas(i), 2*tuningParam);
%     end
% end
% proposalDensity = (1/N_a) * sum;


end

