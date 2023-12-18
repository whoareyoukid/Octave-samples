% Load the dataset named gyro.csv
data=dlmread('gyro1.csv',',');
% All the independent features/ inputs are assigned to X i.e we consider all the rows and columns 1 to 15 (i.e 15 inputs)
X = data(:, 1:15); 
% # All the dependent features/ outputs are assigned to y i.e we consider all the rows and columns 16 to 21 (i.e 6 outputs)
y = data(:, 16:21);


% Computing the distance between x and xu by calling the function distance
function d = distance(x, xu)
    d = abs(x - xu);
end


%Implementing the formula for RBF: 
% Ku(d(x,xu))=e^(d2(x-xu)*(1/2*var(u)))
function ku = Kernel(x, xu)
    ku = exp(-(distance(x, xu).^2)./(2*var(xu)));
end

%Implementing the sigmoid function:
% Formula: 1/(1+e^(-x))
function s = sigmoid(x)
    s = (1./(1 + exp(-x)));
end


%Consider the entire dataset for training the data
%consider 20% of the dataset for testing the data



function [X_test, y_test] = split_data(X, y)
% p = randperm(n,k) returns a row vector containing k unique integers selected randomly from 1 to n. 
% For testing the data, we have considered approx 20% i.e 20% of 159=32
    train_off = randperm(159,32);
% Initailly, setting X_test and y_test as the first values in train_off  
    X_test = X(train_off(1),:);
    y_test = y(train_off(1),:);
% Out of 32, since we have already taken the first index, we will start from 2 and iterate till 32
    for idx = 2:32
% cat is a function used to concatenate. We iterate from 2 to 32 i.e the remaining values from train_off and concatenate each of those values as rows to X_test and y_test 
        X_test = cat(1, X_test, X(train_off(idx),:));
        y_test = cat(1, y_test, y(train_off(idx),:));
    end
end

% At the end of this loop, we will get the entire X_test and y_test with 32 rows.

[X_test, y_test] = split_data(X, y)

% w0 - we are assigning a random fractional value between 0 and 1
w0 = rand();

% Creates a matrix w which is 1*15 in size. rand() assigns random values between 0 and 1. 
w = rand(1,15);

% Creates a matrix hidden which is 15*6 in size with each of the values between 0 and 1
hidden = rand(15,6);

% Learning rate - eta
eta = 0.1;
f = w0;

% epoch is the number of iterations
epoch = 1000;

% Training
for i=1:epoch
% As we consider the entire dataset for training the data, we iterate the for loop from 1 to 159 i.e the total no of rows in the dataset
    for r = 1:159
    % row - considers one row at a time with r ranging from 1 to 159 and has columns from 1 to 15
    % t - considers one row at a time with r ranging from 1 to 159 and has columns from 16 to 21
        row = X(r,:); t = y(r,:);
    % Each time when one row is considered, we will be left with 158 rows which is denoted by other
        other = cat(1, X(1:r-1,:),X(r+1:159,:));
    % Iterate j from 1 to 158 for the remaining rows
        for j = 1:158
    % ku is a kernel value that we would store
            ku = Kernel(row,other(j,:));
            
    % Implementing this formula: f(x)=w0+summation(wu*Ku(d(x,xu)))
            f = f + ku.*w;
            
    % To get 6 outputs, we are multiplying f with hidden and fx will store 6 column output for each row
            fx = f * hidden;
            
    % We use sigmoid function as we get huge values and we need to scale down these values between 0 and 1
    % We use round function to get values as either 0 or 1
            fx = sigmoid(fx);
            
            error = fx .* (1 - fx) .* (t - fx);
    % error cantains the value which is the difference between target and the calculated output
            %error = t - fx;
            #error=round(error);
            
     % Formula for backpropogation:
     
            hidden = (hidden + row'*error).*eta;
            w = (w + row .* (error * hidden')).*eta;
        end
    end
end

% Testing

% correct variable is the no of correctly predicted outputs
correct = 0;
% As we have considered 20% for testing i.e 20% of 159=32. Hence, we iterate the loop from 1 to 32 for testing the data
for r = 1:32
    
% row - considers one row at a time with r ranging from 1 to 32 and has columns from 1 to 15
% t - considers one row at a time with r ranging from 1 to 32 and has columns from 16 to 21
    row = X_test(r,:); t = y_test(r,:);
    
    f = f + row.*w;
    
% To get 6 outputs, we are multiplying f with hidden and fx will store 6 column output for each row
    fx = f * hidden;
    
% We use sigmoid function as we get huge values and we need to scale down these values between 0 and 1
% We use round function to get values as either 0 or 1
    fx = sigmoid(fx)
    
% error cantains the value which is the difference between target and the calculated output
    error = t - fx
    error=round(error)
    
% As the error=[0,0,0,0,0,0], then target and the calculated output is equal and the correct variable is incremented by 1
    if error == [0,0,0,0,0,0]
        correct = correct + 1;
    end
end
% We divide the estimated correct values by the total number of test values through which we predict the accuracy.
correct = correct/32