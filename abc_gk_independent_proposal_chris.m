% trueTheta = 0.5;
% numTrialsBin = 100;
N = 800; %number of particles
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
scores = zeros(8,N); %currently magic number 8 (aux model params)
epsilonTarget = 1;
epsilonInitial = 150;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 0; %set for first loop

priorValues = [10, 5, 10, 5];

thetasAll = zeros(0);
proposalsAll = zeros(0);
thetasAllAccepted = zeros(0);
scoresAll = zeros(0);
R_t_all = zeros(0);
simulation_size = 1000;
lowest_acceptance_rate = 0.005;

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
    y_s = simulate_gk(simulation_size,theta_prop);
    grad = compute_grad(theta_d,y_s,obj,numComp);
    dist_prop = grad*weight_matrix*grad';
    attempts = attempts + 1;
    if abs(dist_prop) < epsilonInitial
        thetas(:, i) = theta_prop;
        distances(1, i) = dist_prop;
        scores(:,i) = grad;
        i = i + 1;
    end
end


for i = 1:numParams
    phis(i,:) = arrayfun(@(x) log((x)./(priorValues(i) - x)), thetas(i,:));
end

%Sort the particles set by ro and set et and e max
[distances, indices] =  sort(distances);

epsilonThreshold = distances(N - N_a);

epsilonMax = distances(N);
counter = 0;
acceptanceRate = 1;
while((lowest_acceptance_rate < acceptanceRate))%||(epsilonMax > 0.9)) %(epsilonTarget <= epsilonMax) &&
    
    thetasIteration = zeros(0);
    proposalsIteration = zeros(0);
    scoresIteration = zeros(0);
    
    
    %     numSimulations = 0;
    accepted = 0;
    [distances, indices] =  sort(distances);
    thetas = thetas(:,indices);
    
    epsilonThreshold = distances(N - N_a);
    epsilonMax = distances(N);
    
    counter = counter + 1;
    
    %for duplicated particles
    
    [thetas(:, (N_a + 1):N), datasample_indices] = datasample(thetas(:, 1:N_a), N - N_a, ndims(thetas));
    distances((N_a+1):N) = distances(datasample_indices);   
    
    for i = 1:numParams
        phis(i,:) = arrayfun(@(x) log((x)./(priorValues(i) - x)), thetas(i,:));
    end
    
    phis_ind = phis(:,1:N_a);
    
    %covariance matrix
    
    sigma = 2*cov(phis');
    
    for j = (N - N_a + 1):N
        %proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma); %phis(:,j)' + mvnrnd(phis(:,j)', sigma);
        proposed_phi = mvnrnd(phis_ind(:,randi(N_a))' , sigma);
        proposalDensity = computeProposalDensity(proposed_phi, phis_ind', sigma);

        originalProposalDensity = computeProposalDensity(phis(:,j)', phis_ind', sigma);
        proposed_theta = priorValues'./(1 + exp(-proposed_phi'));

        y_s = simulate_gk(simulation_size, proposed_theta);
        grad = compute_grad(theta_d,y_s, obj, numComp);

        thetasIteration = [thetasIteration, proposed_theta];
        proposalsIteration = [proposalsIteration, proposalDensity];
        scoresIteration = [scoresIteration, grad'];
        if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))

            psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
            if psi == 1
                accepted = accepted + 1;
                thetas(:,j) = proposed_theta;
                phis(:,j) = proposed_phi;
                distances(j)= sqrt(grad*weight_matrix*grad');
            end
        end
    end
    
    
    acceptanceRate = accepted/(N - N_a) ;
    acceptanceRate
    if acceptanceRate < lowest_acceptance_rate
        break;
    end
    
    R_t = round((log(c))/(log(1 - acceptanceRate)));
    
    for j = (N - N_a + 1):N
        for k = 1:R_t-1
            %proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
            proposed_phi = mvnrnd(phis_ind(:,randi(N_a))' , sigma);
            proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);

            originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
            proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));

            y_s = simulate_gk(simulation_size, proposed_theta);
            grad = compute_grad(theta_d,y_s, obj, numComp);

            
            thetasIteration = [thetasIteration, proposed_theta];
            proposalsIteration = [proposalsIteration, proposalDensity];
            scoresIteration = [scoresIteration, grad'];
            if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
      
                %compute psi (within tolerance)
                psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
                if psi == 1
                    thetas(:,j) = proposed_theta;
                    phis(:,j) = proposed_phi;
                    distances(j) = sqrt(grad*weight_matrix*grad');

                end
            end
        end
    end
    
    thetasAll = [thetasAll, thetasIteration];
    proposalsAll = [proposalsAll, proposalsIteration];
    scoresAll = [scoresAll, scoresIteration];
    
    R_t_all = [R_t_all, R_t];
    thetasAllAccepted = [thetasAllAccepted, thetas];
    R_t_all
    epsilonThreshold
    epsilonMax
    acceptanceRate
end

posteriorThetas = thetas;

%test

for i = 1:numParams
    figure;
    ksdensity(thetas(i,:));
end



