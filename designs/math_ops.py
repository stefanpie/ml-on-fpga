from nmigen import *


class Adder(Elaboratable):
    def __init__(self, width):
        self.a = Signal(signed(width))
        self.b = Signal(signed(width))
        self.o = Signal(signed(width))

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a + self.b)
        return m

    def ports(self):
        return [self.a, self.b, self.o]


class Subtractor(Elaboratable):
    def __init__(self, width):
        self.a = Signal(signed(width))
        self.b = Signal(signed(width))
        self.o = Signal(signed(width))

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a - self.b)
        return m

    def ports(self):
        return [self.a, self.b, self.o]


class Multiplier(Elaboratable):
    def __init__(self, width):
        self.a = Signal(signed(width))
        self.b = Signal(signed(width))
        self.o = Signal(signed(width))

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a * self.b)
        return m

    def ports(self):
        return [self.a, self.b, self.o]
