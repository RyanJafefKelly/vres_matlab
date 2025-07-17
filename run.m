    

% obtain observed summary statistic (only requires a single run). 
% Observed statistic should be a vector of zeros if fitting of mixture
% model worked well
load('gandk_data');
numComp=3;
obj = gmdistribution.fit(x,numComp,'Options',statset('MaxIter', 100000,'TolFun',1e-10));
theta_d = [obj.PComponents(1:(numComp-1)) obj.mu' reshape(obj.Sigma,numComp,1)'];  
weight_matrix = compute_obs_inf(theta_d,x,obj,numComp);
weight_matrix = inv(weight_matrix);
% save the above weight_matrix and theta_d, obj in .mat file and pass it into the ABC
% algorithm

save('gandk_output.mat', 'weight_matrix', 'theta_d', 'obj');

% an example simulation
n = length(x);  % number of data points
theta_prop = [3 1 2 0.5];
y_s = simulate_gk(n,theta_prop);
grad = compute_grad(theta_d,y_s,obj,numComp);
dist_prop = grad*weight_matrix*grad';

