from nmigen import *
from nmigen.cli import main


class FullAdder(Elaboratable):
    def __init__(self):
        self.a = Signal()
        self.b = Signal()
        self.c_in = Signal()

        self.x = Signal()
        self.c_out = Signal()

    def elaborate(self, platform) -> Module:
        m = Module()
        m.d.comb += self.x.eq(((self.a ^ self.b) ^ self.c_in))
        m.d.comb += self.c_out.eq(((self.a & self.b) |
                                  (self.b & self.c_in) | (self.c_in & self.a)))
        return m


if __name__ == "__main__":
    top = FullAdder()
    main(top, ports=(top.a, top.b, top.c_in, top.x, top.c_out))
