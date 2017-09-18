import random
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

def create_dataset(ndata, variance, step=3, correlation=False):
    val = 1
    ys = []
    for i in range(ndata):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
        xs = [i for i in range(len(ys))]
        
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64) 

def best_fit_slope_intercept(xs, ys):
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
          (mean(xs)*mean(xs) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs) 
    return m, b

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sq_error_regr = squared_error(ys_orig, ys_line)
    sq_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (sq_error_regr / sq_error_y_mean)
                
xs, ys = create_dataset(40, 10, 2, correlation='pos')
m, b = best_fit_slope_intercept(xs, ys)

regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=100, color='b')
plt.plot(xs, regression_line)
plt.show()
