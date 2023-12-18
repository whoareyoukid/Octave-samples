%Returns the accuracy of the model
function accuracy = score(X, y)
  correct = 0;
  n = size(X,1);
  for r = 1:n
  % row - input value; considers one row at a time with r ranging from 1 to 40 and has columns from 1 to 15
  % t - target value; considers one row at a time with r ranging from 1 to 40 and has columns from 16 to 21
      row = X(r,:); t = y(r,:);

      fx = predict(row);
      
  % error cantains the value which is the difference between target and the calculated output
      error = t - fx;

  % As the error=[0,0,0,0,0,0], then target and the calculated output is equal and the correct variable is incremented by 1
      if error == [0,0,0,0,0,0]
          correct = correct + 1;
      end
    end
    correct
    correct = correct/n*100
end