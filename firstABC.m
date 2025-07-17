%Set up data
numTrialsBin = 1000;
trueTheta = 0.5;
Y = binornd(numTrialsBin, trueTheta); 

%ABC rejection 
N = 100000;

%Obtain theta^0, x^0 using a burnin or from one sample of rejection ABC
i = 1; %starts at 1 not 0 :(
%Repeat until N samples drawn
accepted_thetas = zeros(1,N+1);
while (i < N) 
    %Draw theta from prior
    theta = rand();
    %Simulate x 
    % simulate auxillary model Binomial...
    x = theta * numTrialsBin; %expected value binomial
    
    %Discrepancy < epsilon
    if( abs((Y-x)) < 1) 
        accepted_thetas(i) = theta;
        i = i+1;
    end
end

%obtain theta0, x0
theta0 = mean(accepted_thetas);
phi0 = computePhi(theta0);

x0 = numTrialsBin * theta0;


%Compute psi^0 = K epsilon -> discrepancy function
psi = computePsi(Y,x0);

%for i=1 to N do
MCMCAcceptedThetas = zeros(1, N-1);
MCMCAcceptedPhis = zeros(1, N-1);
psiVector = zeros(1, N-1);
MCMCAcceptedThetas(1) = theta0;
MCMCAcceptedPhis(1) = phi0;
psiVector(1) = psi;
phi = phi0;
for i = 2:N 
%draw theta * q 
oldPhi = MCMCAcceptedThetas(i-1);
phi = oldPhi + normrnd(0, 0.5);
phiPrior = computePhiPrior(phi);

%Simulate x*
theta = phiToTheta(phi);

xStar = theta * numTrialsBin;

%Compute psi*
%oldPsi = psi;
psi = computePsi(Y,xStar);

%Compute r
% because prior = 1 and normal symmetric...
% but if use phi prior....
r = (computePhiPrior(phi)/computePhiPrior(oldPhi))*psi;
%if U(0,1) <R ...


if rand() < r 
    MCMCAcceptedThetas(i) = theta;
    MCMCAcceptedPhis(i) = phi;
    psiVector(i) = psi;
    %string('test')

else 
    MCMCAcceptedThetas(i) = MCMCAcceptedThetas(i-1);
    MCMCAcceptedPhis(i) = MCMCAcceptedPhis(i-1);
    psiVector(i) = psiVector(i-1);
end

end 
plot(MCMCAcceptedThetas)
ksdensity(MCMCAcceptedThetas);

function output = computePsi(Y, xStar) 
    epsilon = 10;
    %discrepancy function
    ro = Y - xStar;  
    if(abs(ro) < epsilon) 
        output = 1;
    else 
        output = 0;
    end
end

function phi = computePhi(theta) 
    phi = log((theta)/(1 - theta));
end

function theta = phiToTheta(phi)
    theta = 1/(1 + exp(-phi));
end

function priorDistribution = computePhiPrior(phi)
    priorDistribution = (exp(phi))/((exp(phi) + 1 )^2);
end

