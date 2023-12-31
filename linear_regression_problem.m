%Regression

#x = [1850,2190,2100,1930,2300,1710,1550,1920,1840,1720,1660,2405,1525,2030,2240]
#y = [229500,273300,247000,195100,261000,179700,168500,234400,168800,180400,156200,288350,186750,202100,256800]

#x = [46.8,52.1,55.1,59.2,61.9,66.2,69.9,76.8,79.3,79.7,80.2,83.3]
#x = x.^2
#y = [12530,10800,10180,9730,9750,10230,11160,13910,15690,15110,17020,17880]
#y = y.^2

x = [59,52,44,51,42,42,41,45,27,63,54,44,50,47]
y = [56,63,55,50,66,48,58,36,13,50,81,56,64,50]

sum(x)
sum(y)

xmean = mean(x)
ymean = mean(y)

a = x .- xmean
b = y .- ymean
sa = sum(a)
sb = sum(b)

a_sq = a.^2
b_sq = b.^2

sum(a_sq)
sum(b_sq)

axb = a .* b
sum(axb)
r = corr(x,y)
r_sq = r.^2

sdx = std(x)
sdy = std(y)

b_slope = r * sdy / sdx 
%slope
a_yintercept = ymean - b_slope * xmean 
%y-intercept