from nmigen import *
from nmigen.cli import main

from math_ops import Multiplier, Adder

from fxpmath import Fxp


class LinearRegression(Elaboratable):
    def __init__(self, weights, bias, bit_depth=32):
        self.weights = weights
        self.bias = bias
        self.bit_depth = bit_depth

        self.x = Array([Signal(signed(self.bit_depth))
                       for _ in range(len(self.weights))])
        self.y = Signal(signed(bit_depth))

    def elaborate(self, platform):
        m = Module()

        # Make the Multipliers and output signals to do w_i*x_i
        multipliers = [Multiplier(self.bit_depth)
                       for _ in range(len(self.weights))]
        for mult in multipliers:
            m.submodules += mult
        mult_out_signals = Array([Signal(signed(self.bit_depth))
                                 for _ in range(len(self.weights))])

        # Assign input values for multipliers
        for x_i, w, mult in zip(self.x, self.weights, multipliers):
            # print(x_i, w, mult)
            m.d.comb += mult.a.eq(x_i)
            w_fixed_point = Fxp(w, True, 32, 16)
            w_fixed_point_base_10 = int(w_fixed_point.base_repr(10))
            m.d.comb += mult.b.eq(w_fixed_point_base_10)

        # A dookie unbalnced adder tree bc im too lazy to make a real one right now
        # Should spend some time build a balanced adder tree module at some point, not to hard

        # mult_sum = Array([Signal(signed(self.bit_depth)) for _ in range(len(self.weights))])

        adders = [Adder(self.bit_depth)
                  for _ in range(len(self.weights))]
        for adder in adders:
            m.submodules += adder

        for i in range(len(self.weights)):
            if i == 0:
                m.d.comb += adders[i].a.eq(0)
                m.d.comb += adders[i].b.eq(multipliers[i].o)
            else:
                m.d.comb += adders[i].a.eq(adders[i-1].o)
                m.d.comb += adders[i].b.eq(multipliers[i].o)

        mult_sum = Signal(signed(self.bit_depth))
        m.d.comb += mult_sum.eq(adders[-1].o)

        # Now add bias term
        bias_sum = Signal(signed(self.bit_depth))
        bias_adder = Adder(self.bit_depth)
        m.submodules += bias_adder

        m.d.comb += bias_adder.a.eq(mult_sum)
        bias_fixed_point = Fxp(self.bias, True, 32, 16)
        bias_fixed_point_base_10 = int(bias_fixed_point.base_repr(10))
        m.d.comb += bias_adder.b.eq(bias_fixed_point_base_10)
        m.d.comb += bias_sum.eq(bias_adder.o)

        # And were done
        m.d.comb += self.y.eq(bias_sum)
        return m

    def ports(self):
        return []


if __name__ == "__main__":
    x = LinearRegression([1, 2, 3, 4, 5], 25, 32)
    main(x)
