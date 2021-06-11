from nmigen import *
from nmigen.cli import main

from math_ops import Multiplier, Adder

from fxpmath import Fxp

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, cheb2poly
from numpy.polynomial.polynomial import Polynomial, polyval

import matplotlib.pyplot as plt


class LogisticFunction(Elaboratable):
    def __init__(self, design="cheb", bit_depth=32):
        self.design = design
        self.x = Signal(signed(bit_depth), name="x")
        self.y = Signal(signed(bit_depth), name="y")

    def elaborate(self, platform):
        m = Module()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        x = np.linspace(-8, 8, 100)
        y = sigmoid(x)

        if self.design == "cheb":
            degree = 15
            c_cheb = chebfit(x, sigmoid(x), degree)
            polynomial = Polynomial(cheb2poly(c_cheb))
            c_polynomial = polynomial.coef
            # fig, axs = plt.subplots(1,2)
            # axs[0].plot(x, y, label="logistic")
            # axs[0].plot(x, polyval(x, c_polynomial), label="cheb")
            # axs[1].plot(x, polyval(x, c_polynomial)-y, label="error")
            # axs[0].legend()
            # axs[1].legend()
            # plt.tight_layout()
            # plt.show()
        elif self.design == "taylor":
            raise NotImplementedError(
                f"Design type not implemented: \"{self.design}\"")
        else:
            raise Exception(f"Design type not supported: \"{self.design}\"")

        return m

    def ports(self):
        return [self.x, self.y]


if __name__ == "__main__":
    logistic_function = LogisticFunction(design="cheb")
    main(logistic_function, ports=logistic_function.ports())
