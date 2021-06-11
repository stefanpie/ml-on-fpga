from nmigen import *
from nmigen.cli import main
from math_ops import Multiplier

from fxpmath import Fxp


class PowerFunction(Elaboratable):
    def __init__(self, exponent, bit_depth=32):
        self.bit_depth = bit_depth
        self.exponent = exponent
        self.x = Signal(signed(bit_depth), name="x")
        self.y = Signal(signed(bit_depth), name="y")

    def elaborate(self, platform):
        m = Module()

        # Gotta build a multiplier tree now
        # Its definitely going to be unbalanced

        if self.exponent < 0:
            raise NotImplementedError("Exponent values less than 0 not supported")
        elif self.exponent == 0:
            m.d.comb += self.y.eq(int(Fxp(1, True, 32, 16).base_repr(10)))
        else:
            multipliers = [Multiplier(self.bit_depth)
                        for _ in range(self.exponent)]
            for mult in multipliers:
                m.submodules += mult
            for i in range(self.exponent):
                    if i == 0:
                        m.d.comb += multipliers[i].a.eq(int(Fxp(1, True, 32, 16).base_repr(10)))
                        m.d.comb += multipliers[i].b.eq(self.x)
                    else:
                        m.d.comb += multipliers[i].a.eq(multipliers[i-1].o)
                        m.d.comb += multipliers[i].b.eq(self.x)
        
            m.d.comb += self.y.eq(multipliers[-1].o)

        return m

    def ports(self):
        return [self.x, self.y]


if __name__ == "__main__":
    power_function = PowerFunction(4)
    main(power_function, ports=power_function.ports())
