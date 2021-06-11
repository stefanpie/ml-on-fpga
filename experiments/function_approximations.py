import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial
from numpy.polynomial.chebyshev import chebfit, chebval, cheb2poly
from numpy.polynomial.polynomial import Polynomial as P

from pprint import pprint as pp

x = np.linspace(-6, 6, 100)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y = sigmoid(x)

fig, axs = plt.subplots(1,2)

axs[0].plot(x, y, label='sigmoid')

# for degree in range(1, 10, 2):
#     y_taylor = approximate_taylor_polynomial(sigmoid, 0, degree, 10, degree)
#     # pp(y_taylor)
#     axs[0].plot(x, y_taylor(x), label=f'taylor_{degree}')
#     axs[1].plot(x, y-y_taylor(x), label=f'taylor_{degree}')



for degree in range(10, 15, 1):
    def y_cheb(point):
        c = chebfit(x, sigmoid(x), degree)
        return chebval(point, c)
    pp(P(cheb2poly(chebfit(x, sigmoid(x), degree))))
    axs[0].plot(x, y_cheb(x), label=f'cheb_{degree}')
    axs[1].plot(x, y_cheb(x)-y, label=f'cheb_{degree}')


# def make_lookup_table(func, a, b, n):
#     table = func(np.array([(x+0.5)/n*(b-a)+a for x in range(0, n)]))
#     print(table)

#     def tablefunc(x):
#         i = np.int64(np.floor((x-a)/(b-a)*n))
#         i[i==n] = n-1
#         return table[i]
#     return tablefunc

# def make_lookup_table_2(func, a, b, n):
#     table = func(np.array([float(x)/n*(b-a)+a for x in range(0, n+1)]))

#     def tablefunc(x):
#         ii = (x-a)/(b-a)*n
#         i = np.int64(np.floor(ii))
#         i[i == n] = n-1
#         u = ii-i
#         return table[i]*(1-u) + table[i+1]*u
#     return tablefunc

# for sample_points in range(60, 91, 10):
#     y_lut = make_lookup_table(sigmoid, -10, 10, sample_points)
#     axs[0].plot(x, y_lut(x), label=f'lut_{sample_points}')
#     axs[1].plot(x, y-y_lut(x), label=f'lut_{sample_points}')

# for sample_points in range(60, 91, 10):
#     y_lut_2 = make_lookup_table_2(sigmoid, -10, 10, sample_points)
#     axs[0].plot(x, y_lut_2(x), label=f'lut2_{sample_points}')
#     axs[1].plot(x, y-y_lut_2(x), label=f'lut2_{sample_points}')

# axs[0].set_xlim([-10, 10])
# axs[1].set_xlim([-10, 10])

# axs[0].set_ylim([-5,5])
# axs[1].set_ylim([-5, 5])

axs[0].legend()
axs[1].legend()

plt.show()
