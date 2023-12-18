% Load the dataset named gyro1.csv
data=dlmread('gyro1.csv',',');
% All the independent features/ inputs are assigned to X i.e we consider all the rows and columns 1 to 15 (i.e 15 inputs)
X = data(:, 1:15); 
% All the dependent features/ outputs are assigned to y i.e we consider all the rows and columns 16 to 21 (i.e 6 outputs)
y = data(:, 16:21);

function k = rbf_kernel(gamma, x, y)
  k = exp(-gamma * sum((y-x') .^ 2));
end

function clipped = clip(x)
  if x < 0
    clipped = 0;
  elseif x > 1
    clipped = 1;
endif

function t = restrict_to_square(t, v0, u)
  t = (clip(v0 + t*u) - v0)(1)/u(1)
  t = (clip(v0 + t*u) - v0)(0)/u(0)
end

C = 10;
max_iter = 1000;

function fit(x, y)
  y = y * 2 *1;
  lambdas = zeros(159,1);
  K = rbf_kernel(0.1, x, y) * y * y;
  for i = 1:max_iter
    for idxM = 1:159
      idxL = randi([1 159], 1, 1)
      Q = ((K(idxM,idxM), K(idxL, idxL)), (K(idxM, idxL), K(idxM, idxL)))
      v0 = ((lambdas(idxM,:)), (lambdas(idxL,:)))
      k0 = 1 - sum(lambdas * ((K(idxM,:)), (K(idxL,:)))
      u = ((y(idxL,:)), (y(idxM,:)))
      t_max = dot(k0, u) / (dot(dot(Q,u), u) + 10^-15)
      lambdas(idxM,:) = v0 + u * restrict_to_square(t_max, v0, u)
      lambdas(idxL,:) = v0 + u * restrict_to_square(t_max, v0, u)
    endfor
  endfor
  idx = 
end  
%{
    
    idx, = np.nonzero(self.lambdas > 1E-15)
    self.b = np.sum((1.0-np.sum(self.K[idx]*self.lambdas, axis=1))*self.y[idx])/len(idx)
  
  def decision_function(self, X):
    return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b
%}
