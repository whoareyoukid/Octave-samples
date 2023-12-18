% Training using neural networks with 15 input units, 30 hidden units (1 layer), 6 output units
function fx = predict(row)
  % b - bias value for output layer
  b = dlmread('b.csv',',');

  % Weight matrix for output layer
  w = dlmread('w.csv',','); 

  % bh - bias value for hidden layer
  bh = dlmread('bh.csv',',');

  % Weight matrix wh for hidden layer 
  wh = dlmread('wh.csv',','); 
  
  % Formula: f(x)=b+w*x, i.e., f(x) = bias + weight * input
  % Hidden layer computations
  f = bh + (row * wh);
  f = sigmoid(f);
  %Output layer computations
  fx = f * w + b;
  fx = sigmoid(fx);
  fx = round(fx);
  % We use sigmoid function as we get huge values and we need to scale down these values between 0 and 1
% We use round function to get values as either 0 or 1
end