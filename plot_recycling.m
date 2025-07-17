num_thetas = 40;
load('gandk_750.mat');
approximate_posterior_thetas = zeros(numParams, num_thetas);
% selected_posterior_thetas = zeros(numParams, num_thetas);
estimated_thetas = zeros(numParams, num_thetas);
% posteriors = zeros(1,num_thetas);

% load('gandk_600.mat');


%recycling
%randomly select theta from approximated posterior
% theta_r = datasample(thetas',1);

for abc_run = 1:num_thetas
%     theta_r = transpose(posteriorThetas(:,randi(length(posteriorThetas))));
%     selected_posterior_theta = posteriorThetas(:,randi(length(posteriorThetas)));
%     selected_posterior_thetas(:,abc_run) = selected_posterior_theta;
    theta_r = transpose(posteriorThetas(:,randi(length(posteriorThetas))));
    approximate_posterior_thetas(:, abc_run) = theta_r;
    y_r = simulate_gk(10000, theta_r);
%     grad_theta_r = compute_grad(theta_d,y_r, obj, numComp);
    
    %test re-fitting theta_d
%     x = simulate_gk(10000, approximate_posterior_thetas(:,randi(length(approximate_posterior_thetas))));
%     obj = gmdistribution.fit(y_r,numComp,'Options',statset('MaxIter', 100000,'TolFun',1e-10));
%     theta_d = [obj.PComponents(1:(numComp-1)) obj.mu' reshape(obj.Sigma,numComp,1)'];  
%     weight_matrix = compute_obs_inf(theta_d,y_r,obj,numComp);
%     weight_matrix = inv(weight_matrix);
    
    grad_theta_r = compute_grad(theta_d,y_r, obj, numComp);
    
    discrepancies = zeros(1, length(thetasAll));
    for i = 1:length(thetasAll)
%         y_s = simulate_gk(simulation_size, thetasAll(:,i));
%         grad = compute_grad(theta_d, y_s, obj, numComp);
        %discrepancies(i) = norm(grad_theta_r - scoresAll(:,i));
        discrepancies(i) = sqrt((grad_theta_r' - scoresAll(:,i))'*weight_matrix*(grad_theta_r' - scoresAll(:,i)));
    end

    %keep small portion of thetas with lowest discrepancy 
    proportion_keep = 0.001;
    [discrepancies, indices] = sort(discrepancies);
    thetasAll = thetasAll(:, indices);
    proposalsAll = proposalsAll(indices);
    reducedThetas = thetasAll(:,1:round(proportion_keep * length(thetasAll)));
    discrepancies = discrepancies(1:round(proportion_keep * length(thetasAll)));
    proposals = proposalsAll(1:round(proportion_keep * length(thetasAll)));
    indices = indices(1:round(proportion_keep * length(thetasAll)));

    % loop_numbers = zeros(0);
    weights = zeros(1,length(reducedThetas));
    normalised_weights = zeros(1, length(reducedThetas));
    weight_set = zeros(1,length(reducedThetas));
    for i = 1:length(indices)
        k = 1;
%         k = 1;
        while (sum(R_t_all(1:k)) * (N - N_a)) < indices(i) %determine what iteration theta came from
            k = k + 1;
        end 
    %     loop_numbers(i) = k;
%         posteriorDensity = computeProposalDensity( reducedThetas(:,i)', posteriorThetas', 2*cov(posteriorThetas'));
         posteriorDensity = computeProposalDensity( reducedThetas(:,i)', reducedThetas', 2*cov(reducedThetas'));


%         thetasDensity = thetasAllAccepted(:, ((N - N_a)*(k-1))+1:((N - N_a) * k));%thetas used for independent proposal for specific theta
%         qDensity = computeProposalDensity(reducedThetas(:,i)', thetasDensity', 2*cov(thetasDensity'));
        %qDensity = proposals(i);
        weights(i) = posteriorDensity/proposals(i);
        weight_set(i) = k;
    end

    [weight_set, weight_indices] = sort(weight_set);
    weights = weights(weight_indices);

    %count number from each distribution
    set_size = zeros(1, max(weight_set));
    ESS_t = zeros(1, max(weight_set));
    for i = 1:max(weight_set)
        sum_previous_iteration = sum(set_size);
        set_size(i) = sum(i == weight_set);
        if set_size(i) ~= 0
            normalised_weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)) = weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i))./sum(weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)));
            ESS_t(i) = 1/(sum( normalised_weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)).^2));
        end
    end

    normalised_ESS_t = ESS_t/sum(ESS_t);

    final_weights = zeros(1, length(weights));
    counter = 1;
    for i = 1:max(weight_set)
        for k = 1:set_size(i)
            final_weights(counter) = normalised_weights(counter) * normalised_ESS_t(i);
            counter = counter + 1;
        end
    end
    
    
%     estimated_theta = zeros(numParams, 1);
%     for i = 1:max(weight_set)
%         sum_previous_iteration = 0;
%         if i > 1
%             sum_previous_iteration = sum(set_size(1:i));
%         end 
%         sum_weights = sum(weights(sum_previous_iteration+1:sum_previous_iteration + set_size(i)));
%         
%     end
    estimated_thetas(:,abc_run) = transpose(sum(transpose(final_weights.*reducedThetas)));
end
    


%plot
for i = 1:numParams
    figure;
    scatter(approximate_posterior_thetas(i,:), estimated_thetas(i,:));
    refline;
end





