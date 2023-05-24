close all; clc;
C1 = 1e-4;
lambda = linspace(1,1.5);

I1      = 2*lambda.^2 + lambda.^-4;
W       = C1 * (I1-3);
biax_s  = 2*C1 * (lambda.^2 - lambda.^-4);

sig_E = cumtrapz(lambda,biax_s);

hold on
plot(lambda,W,'b-')
plot(lambda,2*cumtrapz(log(lambda),biax_s)','r-.')