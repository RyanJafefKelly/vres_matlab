function [ output ] = computeGKDiscrepancy( grad, weight_matrix, epsilon )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    dist_prop = sqrt(grad*weight_matrix*grad'); 
    if(abs(dist_prop) <= epsilon) 
        output = 1;
    else 
        output = 0;
    end

end

