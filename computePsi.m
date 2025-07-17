function [ output ] = computePsi( Y, x, epsilon )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    ro = Y - x;  
    if(abs(ro) < epsilon) 
        output = 1;
    else 
        output = 0;
    end

end

