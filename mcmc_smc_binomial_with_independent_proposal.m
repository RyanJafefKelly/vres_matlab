trueTheta = 0.5;
numTrialsBin = 100;
N = 15000; %number of particles (must be even currently)
% Y = binornd(numTrialsBin, trueTheta); 
% Y = mnrnd(10, [0.5, 0.5]);
load('bin_data.mat');

num_params = 1;

alpha = 0.5;
N_a = round(N*alpha); %Must be integer 
thetas = zeros(num_params,N);
thetasAfterOneIteration = zeros(1, N);
phis = zeros(1, N);
rhos = zeros(1, N);
epsilonTarget = 2;
epsilonInitial = 10;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 0; %set for first loop 
thetasAll = zeros(1, 1000000);
scoresAll = zeros(1, 1000000);
proposalDensityAll = zeros(1, 1000000);
current_theta = 0;
lowest_acceptance_rate = 0.005;


%(initialisation)
%Perform the rejection sampling algorithm with e1. Produces a set of
%particles 
i = 1;
while (i <= N) 
    %Draw theta from prior
    theta = rand;
    %Simulate x 
    % simulate auxillary model Binomial...
    x = binornd(numTrialsBin, theta); %expected value binomial
    
    %Discrepancy < epsilon
    if( abs((Y-x)) < epsilonInitial) 
        thetas(i) = theta;
        rhos(i) = abs(Y - x);
        i = i+1;
    end
end

%Sort the particles set by ro and set et and e max
[rhos, indices] =  sort(rhos);
    
epsilonThreshold = rhos(N - N_a);
% if epsilonThreshold < epsilonTarget
%     epsilonMax = epsilonThreshold;
% end
epsilonMax = rhos(N);
counter = 0;
acceptanceRate = 1;
while(acceptanceRate >= lowest_acceptance_rate)%(epsilonTarget <= epsilonMax)
    accepted = 0;
    [rhos, indices] =  sort(rhos);
    thetas = thetas(indices);
    
    %randsample
    
    epsilonThreshold = rhos(N - N_a);
    epsilonMax = rhos(N);
    
    oldThetas = thetas;
    
    counter = counter + 1;
    %Compute the tuning parameters of the MCMC kernel using theta particle set
    %cov
%     tuning_param = var(phis);
    
%     if all(phis(:) == 0)
%         tuning_param = var(thetas(1:N-N_a));
%         tuning_param = sqrt(2*tuning_param);
%     else 
%         tuning_param = var(phis(1:N-N_a));
%         tuning_param = sqrt(2*tuning_param);
%     end

    %for duplicated particles
    
    [thetas((N - N_a + 1):N), indices_datasample] = datasample(thetas(1:N_a), N - N_a);
    rhos(N_a+1:N) = rhos(indices_datasample);

    
%     phis = log((thetas)/(1 - thetas));
    phis = arrayfun(@(x) log((x)/(1 - x)), thetas);
    phis_ind = phis(:,1:N_a);
    
    tuning_param = var(phis(1:N-N_a));
    
    
       
    for j = (N - N_a + 1):N
        
        
        proposed_x = randi([1 N_a]);
        % mvnrnd
%         proposed_phi = normrnd(phis(proposed_x), tuning_param);
        proposed_phi = mvnrnd(phis_ind(randi(N_a))' , tuning_param);


        proposalDensity = computeProposalDensity(proposed_phi, phis', tuning_param );
        originalProposalDensity = computeProposalDensity(phis(j), phis', tuning_param);
        proposed_theta = 1/(1 + exp(-proposed_phi));
        x =  binornd(numTrialsBin, proposed_theta);
        
%         thetasAll = [thetasAll, proposed_theta];
        current_theta = current_theta + 1;
        thetasAll(current_theta) = proposed_theta;
        
%         scoresAll = [scoresAll, x];
%         proposalDensityAll = [proposalDensityAll, proposalDensity];
        
        if rand < ((computePhiPrior(proposed_phi) * originalProposalDensity)/(computePhiPrior(phis(j)) * proposalDensity))
%             proposed_theta = 1/(1 + exp(-proposed_phi));
%             x =  binornd(numTrialsBin, proposed_theta);

            %compute psi (within tolerance)
            psi = computePsi(Y,x, epsilonThreshold);
            if psi == 1
                accepted = accepted + 1;
                thetas(j) = proposed_theta;
                rhos(j) = abs(Y-x);
            end
        end
    end
            
    %Compute Rt based on overall MCMC acceptance rate of previous iteration 
    %compute acceptance rate
%     notAccepted = 0;
%     for i = (N-N_a + 1):N
%         if oldThetas(i) == thetasAfterOneIteration(i)
%             notAccepted = notAccepted + 1;
%         end 
%     end 
%     acceptanceRate = accepted/(N - N_a) ;
%     acceptanceRate = 0.1;
%     R_t = round((log(c))/(log(1 - acceptanceRate)));


    acceptanceRate = accepted/(N - N_a) ;
    acceptanceRate
    if acceptanceRate < lowest_acceptance_rate
        break;
    end
%   acceptanceRate = 0.1;
    R_t = round((log(c))/(log(1 - acceptanceRate)));
    for j = (N - N_a + 1):N
        for k = 1:R_t-1
%             proposed_phi = phis(j) + normrnd(0, tuning_param);
            proposed_phi = mvnrnd(phis_ind(randi(N_a))' , tuning_param);
            
            proposalDensity = computeProposalDensity(proposed_phi, phis', tuning_param );
            originalProposalDensity = computeProposalDensity(phis(j), phis', tuning_param);

            proposed_theta = 1/(1 + exp(-proposed_phi));
            x =  binornd(numTrialsBin, proposed_theta);
            
            current_theta = current_theta + 1;
            thetasAll(current_theta) = proposed_theta;
%             thetasAll = [thetasAll, proposed_theta];
%             scoresAll = [scoresAll, x];
%             proposalDensityAll = [proposalDensityAll, proposalDensity];
            
            if rand < ((computePhiPrior(proposed_phi) * originalProposalDensity)/(computePhiPrior(phis(j)) * proposalDensity))

%                 proposed_theta = 1/(1 + exp(-proposed_phi));
%                 x =  binornd(numTrialsBin, proposed_theta);
            
                %compute psi (within tolerance)
                psi = computePsi(Y,x, epsilonThreshold);
                if psi == 1
                    thetas(j) = proposed_theta;
                    rhos(j) = abs(Y-x);
                end
            end
        end
    end
    
end

    figure;
    plot(thetas);
    figure;
    ksdensity(thetas);

