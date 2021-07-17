from nmigen import *
from nmigen.cli import main

from math_ops import Multiplier, Adder
from power_function import PowerFunction

from fxpmath import Fxp

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval, cheb2poly
from numpy.polynomial.polynomial import Polynomial, polyval

import matplotlib.pyplot as plt


class LogisticFunction(Elaboratable):
    def __init__(self, design="cheb", bit_depth=32):
        self.design = design
        self.bit_depth = bit_depth
        self.x = Signal(signed(bit_depth), name="x")
        self.y = Signal(signed(bit_depth), name="y")

    def elaborate(self, platform):
        m = Module()

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        x = np.linspace(-6, 6, 100)
        y = sigmoid(x)

        if self.design == "cheb":
            degree = 10
            c_cheb = chebfit(x, sigmoid(x), degree)
            polynomial = Polynomial(cheb2poly(c_cheb))
            c_polynomial = polynomial.coef

            # print(c_polynomial)
            # print(polynomial)
            # fig, axs = plt.subplots(1,2)
            # axs[0].plot(x, y, label="logistic")
            # axs[0].plot(x, polyval(x, c_polynomial), label="cheb")
            # axs[1].plot(x, polyval(x, c_polynomial)-y, label="error")
            # axs[0].legend()
            # axs[1].legend()
            # plt.tight_layout()
            # plt.show()

            power_functions = [PowerFunction(
                i, bit_depth=self.bit_depth) for i in range(degree+1)]
            for pf in power_functions:
                m.submodules += pf

            for pf in power_functions:
                m.d.comb += pf.x.eq(self.x)

            multipliers = [Multiplier(self.bit_depth) for i in range(degree+1)]
            for mult in multipliers:
                m.submodules += mult

            for i in range(degree+1):
                m.d.comb += multipliers[i].a.eq(power_functions[i].y)
                m.d.comb += multipliers[i].b.eq(int(Fxp(c_polynomial[i], True,
                                                self.bit_depth, self.bit_depth//2).base_repr(10)))

            adders = [Adder(self.bit_depth) for i in range(degree+1)]
            for adder in adders:
                m.submodules += adder

            for i in range(degree+1):
                if i == 0:
                    m.d.comb += adders[i].a.eq(int(Fxp(0, True, self.bit_depth, self.bit_depth//2).base_repr(10)))
                    m.d.comb += adders[i].b.eq(multipliers[i].o)
                else:
                    m.d.comb += adders[i].a.eq(adders[i-1].o)
                    m.d.comb += adders[i].b.eq(multipliers[i].o)

            min_val = -4
            max_val = 4
            min_val_fpx = int(Fxp(min_val, True, self.bit_depth,
                              self.bit_depth//2).base_repr(10))
            max_val_fpx = int(Fxp(max_val, True, self.bit_depth,
                              self.bit_depth//2).base_repr(10))

            with m.If(self.x < min_val_fpx):
                m.d.comb += self.y.eq(int(Fxp(0, True, self.bit_depth, self.bit_depth//2).base_repr(10)))
            with m.Elif(self.x > max_val_fpx):
                m.d.comb += self.y.eq(int(Fxp(1, True, self.bit_depth, self.bit_depth//2).base_repr(10)))
            with m.Else():
                m.d.comb += self.y.eq(adders[-1].o)
                # m.d.comb += self.y.eq(0)

        # m.d.comb += self.y.eq(adders[-1].o)

        elif self.design == "taylor":
            raise NotImplementedError(f"Design type not implemented: \"{self.design}\"")
        else:
            raise Exception(f"Design type not supported: \"{self.design}\"")

        return m

    def ports(self):
        return [self.x, self.y]


if __name__ == "__main__":
    logistic_function = LogisticFunction(design="cheb")
    main(logistic_function, ports=logistic_function.ports())
