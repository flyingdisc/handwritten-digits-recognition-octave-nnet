%% ./04/mlclass-ex4-008/mlclass-ex4/validationCurve.m
%% cross validation set approach to find good value of lambda
%%    (regularization parameter) for handwritten digit recognition
%%    neural network model. 
% repetition
num_rep = 10

% Selected values of lambda
%lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 2 3 4 5 6 7 8 9 10]';
lambda_vec = [0 0.001 0.004 0.007 0.01 0.03 0.05 0.07 0.1 0.15 0.2 0.25...
              0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1 2 3]';

all_error_train = zeros(length(lambda_vec), num_rep);
all_error_val = zeros(length(lambda_vec), num_rep);

for k = 1:num_rep
    fprintf('--- REP: %d ---\n', k);

    % data stored in arrays X, y
    load('ex4data1.mat'); 
    m = size(X, 1);

    % Load the weights into variables Theta1 and Theta2
    load('ex4weights.mat');
    % Unroll parameters 
    initial_nn_params = [Theta1(:) ; Theta2(:)];

    input_layer_size  = 400;  % 20x20 Input Images of Digits
    hidden_layer_size = 25;   % 25 hidden units
    output_neurons = numel(unique(y));   % 10 

    % attach target/output columns
    X = [X y];

    % randomize order of rows, to blend the data
    X = X(randperm(size(X,1)), :);

    % get back re-ordered y
    y = X(:, end);
    % remove y
    X(:, end) = [];

    % split into train (80%), validation (20%)
    idx_mid = 0.8 * m; 
    X_train = X(1:idx_mid, :);
    y_train = y(1:idx_mid, :);
    X_vali = X(idx_mid+1:end, :);
    y_vali = y(idx_mid+1:end, :);

    error_train = zeros(length(lambda_vec), 1);
    error_val = zeros(length(lambda_vec), 1);

    % suppress warning, or to change all | and & in fmincg to || and && 
    warning('off', 'Octave:possible-matlab-short-circuit-operator');

    for i = 1:length(lambda_vec)
        % Create "short hand" for the cost function to be minimized
        costFunction = @(p) nnCostFunction(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       output_neurons, X_train, y_train, lambda_vec(i));

        options = optimset('MaxIter', 50);   % around 95% accuracy
                                       
        % Now, costFunction is a function that takes in only one argument (the
        % neural network parameters)
        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

        [error_train(i), grad_train] =...
            nnCostFunction(nn_params, input_layer_size, hidden_layer_size,...
            output_neurons, X_train, y_train, 0); 
        [error_val(i), grad_val] =...
            nnCostFunction(nn_params, input_layer_size, hidden_layer_size,...
            output_neurons, X_vali, y_vali, 0); 
    endfor

    all_error_train(:, k) = error_train;
    all_error_val(:, k) = error_val;
endfor

mean_error_train = mean(all_error_train, 2);
mean_error_val = mean(all_error_val, 2);

plot(lambda_vec, mean_error_train, lambda_vec, mean_error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');
