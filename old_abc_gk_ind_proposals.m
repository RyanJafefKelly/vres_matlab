% trueTheta = 0.5;
% numTrialsBin = 100;
N = 1000; %number of particles (must be even currently)
% Y = binornd(numTrialsBin, trueTheta); 
% Y = mnrnd(10, [0.5, 0.5]);
 
load('gandk_output.mat');
 
numParams = 4;
numComp = 3;
 
alpha = 0.5;
N_a = round(N*alpha); %Must be integer 
thetas = zeros(numParams,N);
phis = zeros(numParams, N);
distances = zeros(1, N);
epsilonTarget = 250;
epsilonInitial = 1000;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 0; %set for first loop 
% theta_prop = [3 1 2 0.5];
priorValues = [10, 5, 10, 5];
numSimulationsArray = zeros(0,0);
 
 
%(initialisation)
%Perform the rejection sampling algorithm with e1. Produces a set of
%particles 
i = 1;
attempts = 0;
while (i <= N) 
     
    theta_prop = zeros(1, numParams);
    for j = 1:numParams
        theta_prop(j) = rand*priorValues(j);
    end
    %Simulate x 
    % simulate auxillary model Binomial...
    y_s = simulate_gk(N,theta_prop);
    grad = compute_grad(theta_d,y_s,obj,numComp);
    dist_prop = grad*weight_matrix*grad';
    attempts = attempts + 1;
     
    if abs(dist_prop) < epsilonInitial
        thetas(1:numParams, i) = theta_prop;
        distances(1, i) = dist_prop;
        i = i + 1;
    end
end
 
 
 
%Sort the particles set by ro and set et and e max
[distances, indices] =  sort(distances);
     
epsilonThreshold = distances(N - N_a);
% if epsilonThreshold < epsilonTarget
%     epsilonMax = epsilonThreshold;
% end
epsilonMax = distances(N);
counter = 0;
acceptanceRate = 1;
while((epsilonTarget <= epsilonMax) && (0.01 < acceptanceRate))
    numSimulations = 0;
    accepted = 0;
    [distances, indices] =  sort(distances);
    thetas = thetas(1:numParams,indices);
 
    %randsample
     
    epsilonThreshold = distances(N - N_a);
    epsilonMax = distances(N);
     
%     oldThetas = thetas;
     
    counter = counter + 1;
 
    %for duplicated particles
     
    thetas(:, (N_a + 1):N) = datasample(thetas(:, 1:N_a), N - N_a, ndims(thetas));
     
%     phis = log((thetas)/(1 - thetas));
%     phis = arrayfun(@(x) log((x)/(1 - x)), thetas);
     
         
    for i = 1:numParams
        phis(i,:) = arrayfun(@(x) log((x)./(priorValues(i) - x)), thetas(i,:));
    end
 
    %covariance matrix
     
    sigma = cov(phis');
     
    for j = (N - N_a + 1):N
        proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
        proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);
        originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
        if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
            proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
            %compute psi (within tolerance)
            y_s = simulate_gk(N, proposed_theta);
            numSimulations = numSimulations + 1;
            grad = compute_grad(theta_d,y_s, obj, numComp);            
            psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
            if psi == 1
                accepted = accepted + 1;
                thetas(:,j) = proposed_theta;
                distances(j)= grad*weight_matrix*grad';  
            end
        end
    end
     
    %very bad temperory fix
%     if accepted == 0
%         accepted = 1;
%     end
 
    acceptanceRate = accepted/(N - N_a) ;
    R_t = round((log(c))/(log(1 - acceptanceRate)));
%   acceptanceRate = 0.1;
     
 
        for k = 1:R_t-1
            proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
            proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);
            originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
            if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
                proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
                 
                y_s = simulate_gk(N, proposed_theta);
                numSimulations = numSimulations + 1;
                grad = compute_grad(theta_d,y_s, obj, numComp);
                 
                %compute psi (within tolerance)
                psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
                if psi == 1
                    thetas(:,j) = proposed_theta;
                    distances(j) = grad*weight_matrix*grad'; 
                end
            end
        end
        numSimulationsArray = [numSimulationsArray, numSimulations];
    end
     
for i = 1:numParams
    figure;
    ksdensity(thetas(i,:));
end