load('ex4data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% hidden neurons, output neurons
% even with hidden=8 and only 1000 input samples, out of memory
hidden_neurons = 16; 
output_neurons = numel(unique(y));

epochs = 200;
train_size = 80;

% train, test, validation indices
ntrain = 0;
idx_train = ntrain+1:ntrain+train_size;
ntest = 4000;
idx_test = ntest+1:ntest+train_size;
nvali = 4800;
idx_vali = nvali+1:nvali+train_size;

% classification (instead of regression), so.. 
% convert 1 column to 10 columns target with binary 0/1 values.
y10 = zeros(size(y,1), output_neurons);
y10( sub2ind(size(y10), [1:numel(y)]', y) ) = 1;

% attach target columns, and temporary y
X = [X y10 y];

% randomize order of rows, to blend the training data
X = X( randperm(size(X,1)),: );
% get back re-ordered y
y = X(:,end);
% remove y
X(:,end) = [];
% row based
X = X';

% create train, test, validate data, randomized order
fprintf('\nSUBSETTING train, test, validate data.\n');
drawnow('update');
%[X_train, X_test, X_vali] = subset(X', output_neurons, 1);
X_train = X(:, idx_train);
X_test = X(:, idx_test);
X_vali = X(:, idx_vali);
y_test = y(idx_test);   % same indices as X_test

% indices of input & output data
idx_in = size(X_train,1) - output_neurons;
idx_out = idx_in + 1;

% feed forward network, 
fprintf('\nCREATE feed forward network...\n');
drawnow('update');
R = minmax(X_train(1:idx_in,:));    % Rx2
S = [hidden_neurons output_neurons]; 
net = newff(R, S, {'logsig', 'purelin'}, 'trainlm', 'learngdm', 'mse'); 

% create randomized weights for symmetry breaking.
epsilon_init = 0.12;
InW = rand(hidden_neurons, size(R, 1)) * 2 * epsilon_init - epsilon_init;
LaW = rand(output_neurons, hidden_neurons) * 2 * epsilon_init - epsilon_init;

net.IW{1, 1} = InW;
net.LW{2, 1} = LaW;
net.b{1, 1}(:) = 1;
net.b{2, 1}(:) = 1;

net.trainParam.epochs = epochs;

% define validation data new, for matlab compatibility
VV.P = X_vali(1:idx_in,:);
VV.T = X_vali(idx_out:end,:);

% train
fprintf('\nTRAIN the network...\n');
drawnow('update');
[net_train, tr] = train(net, X_train(1:idx_in,:),...
    X_train(idx_out:end,:), [], [], VV);

% simulation
fprintf('\nSIMULATION...\n');
drawnow('update');
sim_out = sim(net_train, X_test(1:idx_in,:));

sim_out1 = round(sim_out);

% convert back 10 outputs to 1
[val,idx] = max(sim_out);

xlim([0 max(y)]);
ylim([0 max(y)]);
scatter(idx', y_test);
title({'Handwriting Digit Scatter, ANN Simulation, Octave nnet';... 
      sprintf('(epochs=%d, train-size=%d)', epochs, train_size)});
xlabel('Result Digit');
ylabel('Test Digit');

fprintf('\nTraining Set Accuracy: %f\n', mean(double(idx' == y_test)) * 100);
drawnow('update');

% weight matrix of trained model
%fprintf('weight matrix of trained model:\n');
%net.IW(1)
%net.LW(2,1)

%{
train=100
TRAINLM, Epoch 0/200, MSE 1.00043/0, Gradient 940.962/1e-010
TRAINLM, Epoch 16/200, MSE 3.81393e-006/0, Gradient 0.348274/1e-010
TRAINLM, Validation stop.
Training Set Accuracy: 48.000000

train=120 - octave
TRAIN the network...
TRAINLM, Epoch 0/200, MSE 0.748301/0, Gradient 936.635/1e-010
TRAINLM, Epoch 10/200, MSE 1.24535e-005/0, Gradient 1.66512/1e-010
TRAINLM, Validation stop.
Training Set Accuracy: 63.333333

train=120 - matlab


%}