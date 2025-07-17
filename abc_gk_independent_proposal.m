% trueTheta = 0.5;
% numTrialsBin = 100;
N = 600; %number of particles
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
% theta_prop = [3 1 2 0.5];
priorValues = [10, 5, 10, 5];
% numSimulationsArray = zeros(0,0);
%thetasAll = zeros(4,1000000);
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
% if epsilonThreshold < epsilonTarget
%     epsilonMax = epsilonThreshold;
% end
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
    thetas = thetas(1:numParams,indices);
    %     phis = phis(1:numParams, indices);
    
    %randsample
    
    epsilonThreshold = distances(N - N_a);
    epsilonMax = distances(N);
    
    %     oldThetas = thetas;
    
    counter = counter + 1;
    
    %for duplicated particles
    
    [thetas(:, (N_a + 1):N), datasample_indices] = datasample(thetas(:, 1:N_a), N - N_a, ndims(thetas));
    distances((N_a+1):N) = distances(datasample_indices);
    
    %     phis(:, (N_a + 1):N) = datasample(phis(:, 1:N_a), N - N_a, ndims(phis));
    
    %     phis = log((thetas)/(1 - thetas));
    %     phis = arrayfun(@(x) log((x)/(1 - x)), thetas);
    
    
    for i = 1:numParams
        phis(i,:) = arrayfun(@(x) log((x)./(priorValues(i) - x)), thetas(i,:));
    end
    
    %covariance matrix
    
    sigma = 2*cov(phis');
    
    for j = (N - N_a + 1):N
        proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma); %phis(:,j)' + mvnrnd(phis(:,j)', sigma);
        proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);

        originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
        proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));

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
    %     acceptanceRate
    R_t = round((log(c))/(log(1 - acceptanceRate)));
    %   acceptanceRate = 0.1;
    
    for j = (N - N_a + 1):N
        for k = 1:R_t-1
            proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
            proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);

            originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
            proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));

            y_s = simulate_gk(simulation_size, proposed_theta);
            grad = compute_grad(theta_d,y_s, obj, numComp);
            
            thetasIteration = [thetasIteration, proposed_theta];
            proposalsIteration = [proposalsIteration, proposalDensity];
            scoresIteration = [scoresIteration, grad'];
            if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
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
    
    %      numSimulationsArray = [numSimulationsArray, numSimulations];
    %thetasAll(:,((counter-1)*N)+1:((counter)*(N))) = thetas;
    %         thetasAll = [thetasAll, thetas];
    %         scoresAll = [scoresAll, scores];
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


%recycling
%randomly select theta from approximated posterior
% theta_r = datasample(thetas',1);

%
% theta_r = transpose(posteriorThetas(:,randi(length(posteriorThetas))));
% y_r = simulate_gk(simulation_size, proposed_theta);
% grad_theta_r = compute_grad(theta_d,y_r, obj, numComp);
% discrepancies = zeros(1, length(thetasAll));
% for i = 1:length(thetasAll)
%     %discrepancies(i) = norm(grad_theta_r - scoresAll(:,i));
%     discrepancies(i) = (grad_theta_r' - scoresAll(:,i))'*weight_matrix*(grad_theta_r' - scoresAll(:,i));
% end
%
% %keep small portion of thetas with lowest discrepancy
% proportion_keep = 0.01;
% [discrepancies, indices] = sort(discrepancies);
% thetasAll = thetasAll(:, indices);
% reducedThetas = thetasAll(:,1:round(proportion_keep * length(thetasAll)));
% discrepancies = discrepancies(1:round(proportion_keep * length(thetasAll)));
% indices = indices(1:round(proportion_keep * length(thetasAll)));
%
% tic;
% % loop_numbers = zeros(0);
% weights = zeros(1,length(reducedThetas));
% normalised_weights = zeros(1, length(reducedThetas));
% weight_set = zeros(1,length(reducedThetas));
% for i = 1:length(indices)
%     k = 1;
%     while (sum(R_t_all(1:k)) * (N - N_a)) < indices(i) %determine what theta set
%         k = k + 1;
%     end
% %     loop_numbers(i) = k;
%     posteriorDensity = computeProposalDensity( reducedThetas(:,i)', posteriorThetas', 2*cov(posteriorThetas'));
%     thetasDensity = thetasAllAccepted(:, ((N - N_a)*(k-1))+1:((N - N_a) * k));
%     qDensity = computeProposalDensity(reducedThetas(:,i)', thetasDensity', 2*cov(thetasDensity'));
%     weights(i) = posteriorDensity/qDensity;
%     weight_set(i) = k;
% end
%
% [weight_set, weight_indices] = sort(weight_set);
% weights = weights(weight_indices);
%
% %count number from each distribution
% set_size = zeros(1, max(weight_set));
% ESS_t = zeros(1, max(weight_set));
% for i = 1:max(weight_set)
%     sum_previous_iteration = sum(set_size);
%     set_size(i) = sum(i == weight_set);
%     if set_size(i) ~= 0
%         normalised_weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)) = weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i))./sum(weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)));
%         ESS_t(i) = 1/(sum( normalised_weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)).^2));
%     end
% end
%
% normalised_ESS_t = ESS_t/sum(ESS_t);
%
% final_weights = zeros(1, length(weights));
% counter = 1;
% for i = 1:max(weight_set)
%     for k = 1:set_size(i)
%         final_weights(counter) = normalised_weights(counter) * normalised_ESS_t(i);
%         counter = counter + 1;
%     end
% end
%
%
%
%
% toc;
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
%
