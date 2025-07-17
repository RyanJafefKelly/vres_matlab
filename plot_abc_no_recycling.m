num_thetas = 5;
numParams = 4;
estimated_thetas = zeros(numParams, num_thetas);
selected_posterior_thetas = zeros(numParams, num_thetas);
simulation_size = 1000;
numComp = 3;
% selected_posterior_thetas_all = zeros(0);
% estimated_thetas_all = zeros(0);

N = 750;

alpha = 0.5;
N_a = round(N*alpha); %Must be integer 
thetas = zeros(numParams,N);
phis = zeros(numParams, N);
distances = zeros(1, N);
scores = zeros(8,N); %currently magic number 8 (aux model params)
epsilonTarget = 1;
epsilonInitial = 500;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 0; %set for first loop 
% theta_prop = [3 1 2 0.5];
priorValues = [10, 5, 10, 5];
lowest_acceptance_rate = 0.0075;

%load posterior thetas
load('approximate_posterior_thetas.mat');



%iteration
for abc_run = 1:num_thetas
%run.m script (fit theta_d to model)
    selected_posterior_theta = approximate_posterior_thetas(:,randi(length(approximate_posterior_thetas)));
    selected_posterior_thetas(:,abc_run) = selected_posterior_theta;
%     x = simulate_gk(10000, approximate_posterior_thetas(:,randi(length(approximate_posterior_thetas))));
    x =  simulate_gk(10000, selected_posterior_theta);
    obj = gmdistribution.fit(x,numComp,'Options',statset('MaxIter', 100000,'TolFun',1e-10));
    theta_d = [obj.PComponents(1:(numComp-1)) obj.mu' reshape(obj.Sigma,numComp,1)'];  
    weight_matrix = compute_obs_inf(theta_d,x,obj,numComp);
    weight_matrix = inv(weight_matrix);


%typical ABC run
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
while((lowest_acceptance_rate < acceptanceRate)) %(epsilonTarget <= epsilonMax) && 
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
    
    phis_ind = phis(:,1:N_a);
    
    for j = (N - N_a + 1):N
        proposed_phi = mvnrnd(phis_ind(:,randi(N_a))' , sigma);
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
%     if accepted == 0
%         accepted = 1;
%     end

    acceptanceRate = accepted/(N - N_a) ;
    if acceptanceRate < lowest_acceptance_rate
        break;
    end   
    R_t = round((log(c))/(log(1 - acceptanceRate)));
  
%   acceptanceRate = 0.1;
    
     for j = (N - N_a + 1):N
        for k = 1:R_t-1
            proposed_phi = mvnrnd(phis_ind(:,randi(N_a))' , sigma);
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
     epsilonThreshold
     epsilonMax
     acceptanceRate
     abc_run
     estimated_thetas(:,abc_run) = mean(thetas');
end
%calculate mean




%end iteration
end

%save for later plot
% save('
% load('no_recycling_results.mat');
% selected_posterior_thetas_all = [selected_posterior_thetas_all, selected_posterior_thetas];
% estimated_thetas_all = [estimated_thetas_all, estimated_thetas];


% save('no_recycling_results.mat', 'selected_posterior_thetas_all', 'estimated_thetas_all');


%plot 
% symbols = ['a', 'b', 'g', 'k'];
for i = 1:numParams
    figure;
    scatter(selected_posterior_thetas(i,:), estimated_thetas(i,:));
    refline;
    xlabel('estimated value');
    ylabel('actual value');
%     title(["Difference between approximate posterior theta and estimated theta for paramater" num2str(i)]);
end
% save('no_recycling_results_5.mat', 'estimated_thetas', 'selected_posterior_thetas')
