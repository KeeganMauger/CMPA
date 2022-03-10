
%--------------------------------------------------------------------------
% Initialization
%--------------------------------------------------------------------------

clear all
clc
close all
set(0,'DefaultFigureWindowStyle','docked')

Is = 0.01e-12;  % Forward bias saturation current
Ib = 0.1e-12;   % Breakdown saturation current
Vb = 1.3;       % Breakdown voltage
Gp = 0.1;       % Parasitic parallel conductance

V = linspace(-1.95,0.7,200);
I = zeros(1,200);
Ivar = zeros(1,200);

for i = 1:width(V)
    I(i) = Is*((exp((1.2/0.025)*(V(i))))-1) + Gp*V(i)...
        - Ib*((exp(-(1.2/0.025)*(V(i)+Vb)))-1);
    Ivar(i) = (I(i)-I(i)*0.2) + ((I(i)+I(i)*0.2) - (I(i)-I(i)*0.2)).*rand(1,1);
end


%--------------------------------------------------------------------------
% Polynomial Fitting
%--------------------------------------------------------------------------

I_pf4 = polyfit(V,I,4);          % 4th degree polynomial fitting
I_pf8 = polyfit(V,I,8);
Ivar_pf4 = polyfit(V,Ivar,4);
Ivar_pf8 = polyfit(V,Ivar,8);    % 8th degree polynomial fitting
I_pv4 = polyval(I_pf4,V);
I_pv8 = polyval(I_pf8,V);
Ivar_pv4 = polyval(Ivar_pf4,V);
Ivar_pv8 = polyval(Ivar_pf8,V);



figure('Name','Polynomial Fitting')

subplot(2,2,1);
plot(V,I)
hold on
plot(V,I_pv4)
plot(V,I_pv8)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, No Noise')
legend('I Data','I poly 4th','I poly 8th')

subplot(2,2,2);
semilogy(V,I)
hold on
semilogy(V,I_pv4)
semilogy(V,I_pv8)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Semilog, No Noise')
legend('I Data','I poly 4th','I poly 8th')

subplot(2,2,3);
plot(V,Ivar)
hold on
plot(V,Ivar_pv4)
plot(V,Ivar_pv8)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, Noise')
legend('I Data','I poly 4th','I poly 8th')

subplot(2,2,4);
semilogy(V,Ivar)
hold on
semilogy(V,Ivar_pv4)
semilogy(V,Ivar_pv8)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Semilog, Noise')
legend('I Data','I poly 4th','I poly 8th')

%--------------------------------------------------------------------------
% Curve Fitting
%--------------------------------------------------------------------------

% Fitting A C
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(I,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_AC = If_reshape;

% Fitting A B C
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(I,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_ABC = If_reshape;

% Fitting A B C D
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(I,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_ABCD = If_reshape;

figure('Name','Curve Fitting')
subplot(2,1,1);
plot(V,I)
hold on
plot(V,CF_AC)
plot(V,CF_ABC)
plot(V,CF_ABCD)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, No Noise')
legend('I Data', 'I fit A C', 'I fit A B C', 'I fit A B C D')

% Fitting A C
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(Ivar,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_AC = If_reshape;

% Fitting A B C
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(Ivar,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_ABC = If_reshape;

% Fitting A B C D
fo = ...
fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
V_reshape = reshape(V,[200 1]);
I_reshape = reshape(Ivar,[200 1]);
ff = fit(V_reshape,I_reshape,fo);
If = ff(V);
If_reshape = reshape(If,[1 200]);
CF_ABCD = If_reshape;

subplot(2,1,2);
plot(V,Ivar)
hold on
plot(V,CF_AC)
plot(V,CF_ABC)
plot(V,CF_ABCD)
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, Noise')
legend('I Data', 'I fit A C', 'I fit A B C', 'I fit A B C D')

%--------------------------------------------------------------------------
% Neural Net Fitting
%--------------------------------------------------------------------------


inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs;
Inn_reshape = reshape(Inn, [1 200]);

figure('Name','Neural Network Fitting')

subplot(2,1,1);
plot(V,I)
hold on
plot(V,Inn_reshape);
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, No Noise')
legend('I Data','I Neural Network')

inputs = V.';
targets = Ivar.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs)
view(net)
Inn = outputs;
Inn_reshape = reshape(Inn, [1 200]);

subplot(2,1,2);
plot(V,I)
hold on
plot(V,Inn_reshape);
xlabel('V')
ylabel('I')
title('Diode I-V Curve: Linear, Noise')
legend('I Data','I Neural Network')




