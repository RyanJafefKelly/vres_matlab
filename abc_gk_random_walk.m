% trueTheta = 0.5;
% numTrialsBin = 100;
N = 800; %number of particles (must be even currently)
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
epsilonTarget = 20;
epsilonInitial = 1000;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 0; %set for first loop 
priorValues = [10, 5, 10, 5];
numSimulationsArray = zeros(0,0);
proposedThetasAll = cell(0);
proposedThetasIterations= cell(0);



%(initialisation)
%Perform the rejection sampling algorithm with e1. Produces a set of
%particles 
i = 1;
attempts = 0;
numSimulations = 0;
firstSims = 0;
while (i <= N) 
    
    theta_prop = zeros(1, numParams);
    for j = 1:numParams
        theta_prop(j) = rand*priorValues(j);
    end
    %Simulate x 
    % simulate auxillary model Binomial...
    y_s = simulate_gk(N,theta_prop);
%     numSimulations = numSimulations + 1;
    firstSims = firstSims + 1;
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
while((lowest_acceptance_rate < acceptanceRate))

    accepted = 0;
    [distances, indices] =  sort(distances);
    thetas = thetas(1:numParams,indices);

    %randsample
    
    proposed_thetas = zeros(numParams+1, 1);
    proposedThetasIterations = cell(0);
    
    epsilonThreshold = distances(N - N_a);
    epsilonMax = distances(N);
    
%     oldThetas = thetas;
    
    counter = counter + 1;

    %for duplicated particles
    
    thetas(:, (N_a + 1):N) = datasample(thetas(:, 1:N_a), N - N_a, 2);
            
    for i = 1:numParams
        phis(i,:) = arrayfun(@(x) log((x)./(priorValues(i) - x)), thetas(i,:));
    end

    %covariance matrix
    
    sigma = cov(phis');
    
    for j = (N - N_a + 1):N
        proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
        proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
%         proposedThetas(:,(j - (N - N_a))) = transpose(proposed_theta);
%         proposedThetas(5, (j - (N - N_a))) = 0;
        if rand < (computeGKPhiPrior(proposed_phi)/computeGKPhiPrior(transpose(phis(:,j))))
            %compute psi (within tolerance)
            y_s = simulate_gk(N, proposed_theta);
            numSimulations = numSimulations + 1;
            grad = compute_grad(theta_d,y_s, obj, numComp);
            firstSims = firstSims + 1;
            
            psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
            if psi == 1
                phis(:,j) = proposed_phi';
                accepted = accepted + 1;
                thetas(:,j) = proposed_theta;
                distances(j)= sqrt(grad*weight_matrix*grad');  
%                 proposedThetas(5, (j - (N - N_a))) = 1;
            end
        end
%         proposedThetasIterations{1} = proposedThetas;
    end
    
    %very bad temperory fix
%     if accepted == 0
%         accepted = 1;
%     end

    acceptanceRate = accepted/(N - N_a) ;
    acceptanceRate
    if acceptanceRate <= lowest_acceptance_rate
        break;
    end
    R_t = round((log(c))/(log(1 - acceptanceRate)));
%   acceptanceRate = 0.1;
    
    for j = (N - N_a + 1):N
        for k = 1:R_t-1
            proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
%             proposedThetas(1:numParams,(j - (N - N_a))) = proposed_theta;
%             proposedThetas(5, (j - (N - N_a))) = 0;
%             proposalDensity = computeProposalDensity(proposed_phi, N_a, tuning_param, phis );
%             originalProposalDensity = computeProposalDensity(phis(j), N_a, tuning_param, phis);

%             if rand < ((computePhiPrior(proposed_phi) * originalProposalDensity)/(computePhiPrior(phis(j)) * proposalDensity))
            if rand < (computeGKPhiPrior(proposed_phi)/computeGKPhiPrior(transpose(phis(:,j))))
                proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
                
                y_s = simulate_gk(N, proposed_theta);
                numSimulations = numSimulations + 1;
                grad = compute_grad(theta_d,y_s, obj, numComp);
                
                %compute psi (within tolerance)
                psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
                if psi == 1
                    phis(:,j) = proposed_phi';
                    thetas(:,j) = proposed_theta;
                    distances(j) = sqrt(grad*weight_matrix*grad'); 
%                     proposedThetas(5, (j - (N - N_a))) = 1;
                end
            end
%             proposedThetasIterations{k} = proposedThetas;
        end
    end
        numSimulationsArray = [numSimulationsArray, numSimulations];
%         proposedThetasAll{counter} = proposedThetasIterations;
    epsilonMax
    end
    

% figure;
% plot(thetas);
for i = 1:numParams
    figure;
    ksdensity(thetas(i,:));
end

