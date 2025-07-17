N = 400; %number of particles 

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
% thetasAll = zeros(0);
% thetasAllAccepted = zeros(0);
% scoresAll = zeros(0);
R_t_all = zeros(0);
simulation_size = 1000;

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
while((0.005 < acceptanceRate)) %(epsilonTarget <= epsilonMax) && 
    numSimulations = 0;
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
%         proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
%         thetasAll = [thetasAll, proposed_theta];
%         y_s = simulate_gk(simulation_size, proposed_theta);
%         grad = compute_grad(theta_d,y_s, obj, numComp);  
%         scoresAll = [scoresAll, grad'];
        if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
            proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
            %compute psi (within tolerance)
            y_s = simulate_gk(N, proposed_theta);
%             numSimulations = numSimulations + 1;
            grad = compute_grad(theta_d,y_s, obj, numComp);            
            psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
            if psi == 1
                accepted = accepted + 1;
                thetas(:,j) = proposed_theta;
%                 scores(:,j) = grad;
%                 phis(:,j) = proposed_phi;
                distances(j)= sqrt(grad*weight_matrix*grad');  
            end
        end
    end
    
    %very bad temperory fix
%     if accepted == 0
%         accepted = 1;
%     end

    %temp fix (stop R_t inf)
    if accepted == 0
        accepted = 1;
    end

    acceptanceRate = accepted/(N - N_a) ;
    R_t = round((log(c))/(log(1 - acceptanceRate)));
%   acceptanceRate = 0.1;
    
     for j = (N - N_a + 1):N
        for k = 1:R_t-1
            proposed_phi = transpose(phis(1:numParams,j)) + mvnrnd(zeros(1,numParams), sigma);
            proposalDensity = computeProposalDensity(proposed_phi, phis', sigma);
            originalProposalDensity = computeProposalDensity(phis(:,j)', phis', sigma);
%             proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
%             thetasAll = [thetasAll, proposed_theta];
%             y_s = simulate_gk(simulation_size, proposed_theta);
%             grad = compute_grad(theta_d,y_s, obj, numComp);  
%             scoresAll = [scoresAll, grad'];
            if rand < ((computeGKPhiPrior(proposed_phi)*originalProposalDensity)/(proposalDensity*computeGKPhiPrior(transpose(phis(:,j)))))
                proposed_theta = priorValues'./(1 + exp(-transpose(proposed_phi)));
                
                y_s = simulate_gk(N, proposed_theta);
%                 numSimulations = numSimulations + 1;
                grad = compute_grad(theta_d,y_s, obj, numComp);
                
                %compute psi (within tolerance)
                psi = computeGKDiscrepancy(grad,weight_matrix, epsilonThreshold);
                if psi == 1
                    thetas(:,j) = proposed_theta;
%                     phis(:,j) = proposed_phi;
                    distances(j) = sqrt(grad*weight_matrix*grad'); 
%                     scores(:,j) = grad;
                end
            end
        end
     end
%      numSimulationsArray = [numSimulationsArray, numSimulations];
        %thetasAll(:,((counter-1)*N)+1:((counter)*(N))) = thetas;
%         thetasAll = [thetasAll, thetas];
%         scoresAll = [scoresAll, scores];
%      R_t_all = [R_t_all, R_t]; 
%      thetasAllAccepted = [thetasAllAccepted, thetas];
     R_t_all
     epsilonThreshold
     epsilonMax
     acceptanceRate
end
    
posteriorThetas = thetas;

%test    
symbols = ['a','b','g','k'];
for i = 1:numParams
    figure;
    ksdensity(thetas(i,:));
    xlabel(symbols(i));
    ylabel('density');
    title(['Approximate posterior for ' symbols(i)]); 
end
    
















