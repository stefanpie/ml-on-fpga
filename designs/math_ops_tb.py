
import math_ops
from fxpmath import Fxp

from nmigen import *
from nmigen.cli import main
from nmigen.back.pysim import Simulator, Delay, Settle

dut_adder = math_ops.Adder(32)
dut_multiplier = math_ops.Multiplier(32)

m = Module()
m.submodules.dut_adder = dut_adder
m.submodules.dut_multiplier = dut_multiplier

sim = Simulator(m)


def adder():
    
    a = Fxp(-6.78, True, 32, 16)
    b = Fxp(3.219, True, 32, 16)
    yield dut_adder.a.eq(int(a.base_repr(10)))
    yield dut_adder.b.eq(int(b.base_repr(10)))
    yield Delay(1e-6)
    yield Settle()
    output = yield dut_adder.o
    out_fpx = Fxp(output, True, 32, 16, raw=True)
    print("Adder Results")
    print(a)
    print(b)
    print("================== +")
    print(out_fpx)
    print()


def multiplier():
    a = Fxp(-2.5, True, 32, 16)
    b = Fxp(2.3568, True, 32, 16)
    yield dut_multiplier.a.eq(int(a.base_repr(10)))
    yield dut_multiplier.b.eq(int(b.base_repr(10)))
    yield Delay(1e-6)
    yield Settle()
    output = yield dut_multiplier.o
    out_fpx = Fxp(output, True, 32, 16, raw=True)
    print("Multiplier Results")
    print(a)
    print(b)
    print("================== x")
    print(out_fpx)
    print()

sim.add_process(adder)
sim.add_process(multiplier)


traces = [*dut_adder.ports(), *dut_multiplier.ports()]

with sim.write_vcd("math_ops_tb.vcd", "math_ops_tb.gtkw", traces=traces):
    sim.run()
