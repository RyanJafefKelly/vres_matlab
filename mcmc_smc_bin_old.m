trueTheta = 0.5;
numTrialsBin = 100;
N = 10000; %number of particles
Y = binornd(numTrialsBin, trueTheta); 
 
alpha = 0.5;
N_a = round(N*alpha); %Must be integer 
thetas = zeros(1,N-1);
thetasAfterOneIteration = zeros(1, N-1);
phis = zeros(1, N-1);
ros = zeros(1, N-1);
epsilonTarget = 2;
epsilonInitial = 10;
epsilonThreshold = 0;
epsilonMax = 0;
c = 0.01;
R_t = 10; %set for first loop 
 
%(initialisation)
%Perform the rejection sampling algorithm with e1. Produces a set of
%particles 
i = 1;
notAccepted = 0;
while (i < N) 
    %Draw theta from prior
    theta = rand();
    %Simulate x 
    % simulate auxillary model Binomial...
    x = binornd(numTrialsBin, theta); %expected value binomial
     
    %Discrepancy < epsilon
    if( abs((Y-x)) < epsilonInitial) 
        thetas(i) = theta;
        ros(i) = abs(Y - x);
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
[ros, indices] =  sort(ros);
for i = 1:N-1
    indice = indices(i);
    thetas(i) = thetas(indice);
end
     
epsilonThreshold = ros(N - N_a);
epsilonMax = ros(N-1);
counter = 0;
while(epsilonTarget < epsilonMax)
    [ros, indices] =  sort(ros);
    for i = 1:N-1
        indice = indices(i);
        thetas(i) = thetas(indice);
    end
     
    epsilonThreshold = ros(N - N_a);
    epsilonMax = ros(N-1);
     
    oldThetas = thetas;
     
    counter = counter + 1;
    %Compute the tuning parameters of the MCMC kernel using theta particle set
    tuning_data = thetas(1:N-N_a);
    tuning_param = cov(tuning_data);
 
    %for duplicated particles
   
    for j = (N - N_a + 1):N-1 
          %Resample (duplicate exactly twice)
        thetas(j) = thetas(j - N_a);
        oldThetas(j) = thetas(j);
        phis(j) = log((thetas(j))/(1 - thetas(j)));
%         if j == N - N_a + 1
%            R_t  
%         end
 
        for k = 1:R_t
            proposed_phi = phis(j) + normrnd(0, tuning_param);
 
            %early rejection.
            if rand() < (computePhiPrior(proposed_phi)/computePhiPrior(phis(j)))
                proposed_theta = 1/(1 + exp(-proposed_phi));
                x =  binornd(numTrialsBin, proposed_theta);
             
                %compute psi (within tolerance)
                if epsilonThreshold < epsilonTarget 
                    epsilonThreshold = epsilonTarget;
                end
                psi = computePsi(Y,x, epsilonThreshold);
                if psi == 1
                    thetas(j) = proposed_theta;
                    ros(j) = abs(Y-x);
                end
            end
             
            if k == 1
                thetasAfterOneIteration(j) = thetas(j);
            end
             
        end
 
    end 
 
     
    %Compute Rt based on overall MCMC acceptance rate of previous iteration 
    %compute acceptance rate
    notAccepted = 0;
    for i = (N-N_a + 1):N-1
        if oldThetas(i) == thetasAfterOneIteration(i)
            notAccepted = notAccepted + 1;
        end 
    end 
    acceptanceRate = (N_a - notAccepted)/N_a ;
    R_t = round((log(c))/(log(1 - acceptanceRate)));
%     if R_t < 1
%         R_t = 1;
%     end
end
 
figure;
plot(thetas);
figure;
ksdensity(thetas);