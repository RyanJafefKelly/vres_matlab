trueTheta = 0.5;
numTrialsBin = 100;
N = 15000; %number of particles
load('bin_data.mat');

% Y = binornd(numTrialsBin, trueTheta); 

alpha = 0.5;
N_a = round(N*alpha); %Must be integer 
thetas = zeros(1,N);
thetasAfterOneIteration = zeros(1, N);
phis = zeros(1, N);
rhos = zeros(1, N);
epsilonTarget = 1;
epsilonInitial = 10;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 10; %set for first loop 
lowest_acceptance_rate = 0.005;
num_sims = 0;

%(initialisation)
%Perform the rejection sampling algorithm with e1. Produces a set of
%particles 
i = 1;
notAccepted = 0;
while (i <= N) 
    %Draw theta from prior
    theta = rand();
    %Simulate x 
    % simulate auxillary model Binomial...
    x = binornd(numTrialsBin, theta); %expected value binomial
    
    %Discrepancy < epsilon
    if( abs((Y-x)) < epsilonInitial) 
        thetas(i) = theta;
        rhos(i) = abs(Y - x);
        i = i+1;
    else 
        notAccepted = notAccepted + 1;
    end
end

%initial R_t
% acceptanceRate = 1000/(notAccepted + 1000);
% 
% R_t = round((log(c))/(log(1 - acceptanceRate))) + 100;


%Sort the particles set by ro and set et and e max
[rhos, indices] =  sort(rhos);
for i = 1:N
    indice = indices(i);
    thetas(i) = thetas(indice);
end

%  phis(j) = log((thetas(j))/(1 - thetas(j)));

    
epsilonThreshold = rhos(N - N_a);
epsilonMax = rhos(N);
counter = 0;
acceptanceRate = 1;
while(lowest_acceptance_rate <= acceptanceRate )
    accepted = 0;
    [rhos, indices] =  sort(rhos);
    thetas = thetas(indices);
%     for i = 1:N-1
%         indice = indices(i);
%         thetas(i) = thetas(indice);
%     end
    
    epsilonThreshold = rhos(N - N_a);
    epsilonMax = rhos(N);
    
    
    [thetas((N_a + 1):N), indices_datasample] = datasample(thetas(1:N_a), N - N_a);
    rhos(N_a+1:N) = rhos(indices_datasample);
    
    phis = arrayfun( @(x) log((x) ./ (1 - x)), thetas);

    counter = counter + 1;
    %Compute the tuning parameters of the MCMC kernel using theta particle set
    tuning_data = phis(1:N-N_a);
    tuning_param = cov(tuning_data);
    %for duplicated particles
  
        for j = (N - N_a + 1):N
          %Resample (duplicate exactly twice)
%         thetas(j) = thetas(j - N_a);
%         phis(j) = log((thetas(j))/(1 - thetas(j)));
%         if j == N - N_a + 1
%            R_t  
%         end            
          proposed_phi = phis(j) + normrnd(0, tuning_param);

            %early rejection.
            if rand() < (computePhiPrior(proposed_phi)/computePhiPrior(phis(j)))
                proposed_theta = 1/(1 + exp(-proposed_phi));
                x =  binornd(numTrialsBin, proposed_theta);
                num_sims = num_sims + 1;
                %compute psi (within tolerance)
%                 if epsilonThreshold < epsilonTarget 
%                     epsilonThreshold = epsilonTarget;
%                 end
                psi = computePsi(Y,x, epsilonThreshold);
                if psi == 1
                    thetas(j) = proposed_theta;
                    phis(j) = proposed_phi;
                    accepted = accepted + 1;
                    rhos(j) = abs(Y-x);
                end    
            end
    
        end 
       
            %Compute Rt based on overall MCMC acceptance rate of previous iteration 
    %compute acceptance rate
    acceptanceRate = accepted/(N - N_a);
    if acceptanceRate < lowest_acceptance_rate
        break;
    end
    acceptanceRate
    R_t = round((log(c))/(log(1 - acceptanceRate)));
    
    
    for j = (N - N_a + 1):N

        for k = 1:R_t
            proposed_phi = phis(j) + normrnd(0, tuning_param);

            %early rejection.
            if rand() < (computePhiPrior(proposed_phi)/computePhiPrior(phis(j)))
                proposed_theta = 1/(1 + exp(-proposed_phi));
                x =  binornd(numTrialsBin, proposed_theta);
                num_sims = num_sims + 1;
                %compute psi (within tolerance)
%                 if epsilonThreshold < epsilonTarget 
%                     epsilonThreshold = epsilonTarget;
%                 end
                psi = computePsi(Y,x, epsilonThreshold);
                if psi == 1
                    phis(j) = proposed_phi;
                    thetas(j) = proposed_theta;
                    rhos(j) = abs(Y-x);
                end
            end
            
        end

    end 


end

% figure;
% plot(thetas);
% figure;
symbols = ['a','b','g','k'];
for i = 1:4
    figure;
    hold on;
    ksdensity(i_thetas(i,:));
    % title('Comparison of random walk and independent proposals');
    xlabel(symbols(i));
    ylabel('density');
    ksdensity(rw_thetas(i,:));
    legend('Independent Proposal', 'Random Walk Proposal');
end

for i = 1:4
    figure;
    ksdensity(thetas(i,:));
end




