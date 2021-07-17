
import logistic_function
from fxpmath import Fxp

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, cheb2poly
from numpy.polynomial.polynomial import Polynomial as P
import matplotlib.pyplot as plt


from nmigen import *
from nmigen.cli import main
from nmigen.sim import Simulator, Delay, Settle



BIT_DEPTH=32
dut_logistic_function = logistic_function.LogisticFunction(bit_depth=BIT_DEPTH)

m = Module()
m.submodules.dut_logistic_function = dut_logistic_function

sim = Simulator(m)




def process_main():
    x_vals = np.linspace(-6, 6, 100)
    x_vals_fpx = [Fxp(x.tolist(), True, BIT_DEPTH, BIT_DEPTH//2) for x in x_vals]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def y_cheb(point):
        c = chebfit(x_vals, sigmoid(x_vals), 10)
        return chebval(point, c)

    y_vals_true = sigmoid(x_vals)
    y_values_cheb = y_cheb(x_vals)
    y_vals = []
    for x in x_vals_fpx:
        yield dut_logistic_function.x.eq(int(x.base_repr(10)))
        # yield Delay(1e-9)
        yield Settle()
        output = yield dut_logistic_function.y
        
        out_fpx = Fxp(output, True, BIT_DEPTH, BIT_DEPTH//2, raw=True)
        print(x, out_fpx)
        y_vals.append(out_fpx)

    print("Logistic Function Results")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x_vals, y_vals, label="sim")
    axs[0].plot(x_vals, y_values_cheb, label="cheb")
    axs[0].plot(x_vals, y_vals_true, label="real")
    axs[0].legend()
    axs[1].plot(x_vals, y_vals_true-y_vals, label="sim")
    axs[1].plot(x_vals, y_vals_true-y_values_cheb, label="cheb")
    axs[1].legend()
    axs[1].set_ylim([-0.01, 0.01])
    plt.tight_layout()
    plt.show()
    # print(out_fpx)
    print()


sim.add_process(process_main)


traces = [*dut_logistic_function.ports()]

with sim.write_vcd("logistic_function_tb.vcd", "logistic_function_tb.gtkw", traces=traces):
    sim.run()
