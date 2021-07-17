import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import approximate_taylor_polynomial


from pprint import pprint as pp


x = np.linspace(0.001, 1, 1000)


def inverse(x):
    return 1 / x


y = inverse(x)

fig, axs = plt.subplots(1, 2)

axs[0].plot(x, y, label='inverse')


# def make_lookup_table(func, a, b, n):
#     table = func(np.array([(x+0.5)/n*(b-a)+a for x in range(0, n)]))
#     print(table)

#     def tablefunc(x):
#         i = np.int64(np.floor((x-a)/(b-a)*n))
#         i[i == n] = n-1
#         return table[i]
#     return tablefunc

# for sample_points in range(100, 1000, 100):
#     y_lut = make_lookup_table(inverse, 0, 1, sample_points)
#     axs[0].plot(x, y_lut(x), label=f'lut_{sample_points}')
#     axs[1].plot(x, y-y_lut(x), label=f'lut_{sample_points}')

for degree in range(19, 20, 1):
    def y_poly(point):
        c = np.polyfit(x, inverse(x), degree)
        return np.polyval(c, point)
    axs[0].plot(x, y_poly(x), label=f'poly_{degree}')
    axs[1].plot(x, y-y_poly(x), label=f'poly_{degree}')

axs[0].legend()
axs[1].legend()

# axs[0].set_xlim([-10, 10])
# axs[1].set_xlim([-10, 10])

axs[0].set_ylim([-10, 50])
# axs[1].set_ylim([-5, 5])

plt.show()
