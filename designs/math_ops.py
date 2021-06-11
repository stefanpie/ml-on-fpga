from nmigen import *


class Adder(Elaboratable):
    def __init__(self, width):
        self.width = width
        self.a = Signal(signed(width), name="a")
        self.b = Signal(signed(width), name="b")
        self.o = Signal(signed(width), name="o")

    def elaborate(self, platform):
        m = Module()
        m.d.comb += self.o.eq(self.a + self.b)
        return m

    def ports(self):
        return [self.a, self.b, self.o]

class Multiplier(Elaboratable):
    def __init__(self, width):
        self.width = width
        self.a = Signal(signed(width), name="a")
        self.b = Signal(signed(width), name="b")
        self.o = Signal(signed(width), name="o")
        

    def elaborate(self, platform):
        m = Module()

        # larger output for mult op
        o_i = Signal(signed(self.width*2)) 
        
        # do mult
        m.d.comb += o_i.eq(self.a * self.b)
        
        # slice to real output which is same as input size (smaller)
        m.d.comb += self.o.eq(Cat(o_i[(self.width//2):self.width], o_i[self.width:(self.width + (self.width//2)-1)], o_i[-1]))         
        return m
    
    def ports(self):
        return [self.a, self.b, self.o]



