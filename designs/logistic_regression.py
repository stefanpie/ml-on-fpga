from nmigen import *
from nmigen.cli import main

from linear_regression import LinearRegression
from logistic_function import LogisticFunction

# from fxpmath import Fxp


class LogisticRegression(Elaboratable):
    def __init__(self, weights, bias, bit_depth=32):
        self.weights = weights
        self.bias = bias
        self.bit_depth = bit_depth

        self.x = [Signal(signed(self.bit_depth), name=f"x_{i}")
                  for i in range(len(self.weights))]

        self.y = Signal(signed(bit_depth), name="y")

    def elaborate(self, platform):
        m = Module()
        
        linear_regression = LinearRegression(self.weights, self.bias, bit_depth=self.bit_depth)
        m.submodules += linear_regression

        for i in range(len(self.x)):
            m.d.comb += linear_regression.x[i].eq(self.x[i])

        logistic_function = LogisticFunction(bit_depth=self.bit_depth)
        m.submodules += logistic_function

        m.d.comb += logistic_function.x.eq(linear_regression.y)
        m.d.comb += self.y.eq(logistic_function.y)

        return m

    def ports(self):
        return [*self.x, self.y]


if __name__ == "__main__":
    logistic_regression = LogisticRegression([1, 2, 3, 4, 5], 5, 32)
    main(logistic_regression, ports=logistic_regression.ports())
