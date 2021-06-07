
import math_ops

from nmigen import *
from nmigen.cli import main
from nmigen.back.pysim import Simulator, Delay, Settle

dut_adder = math_ops.Adder(8)
dut_subtractor = math_ops.Subtractor(8)
dut_multiplier = math_ops.Multiplier(8)

m = Module()
m.submodules.dut_adder = dut_adder
m.submodules.dut_subtractor = dut_subtractor
m.submodules.dut_multiplier = dut_multiplier

sim = Simulator(m)


def adder():
    yield dut_adder.a.eq(1)
    yield dut_adder.b.eq(4)
    yield Delay(1e-6)
    yield Settle()


def subtractor():
    yield dut_subtractor.a.eq(1)
    yield dut_subtractor.b.eq(4)
    yield Delay(1e-6)
    yield Settle()


def multiplier():
    yield dut_multiplier.a.eq(1)
    yield dut_multiplier.b.eq(4)
    yield Delay(1e-6)
    yield Settle()


sim.add_process(adder)
sim.add_process(subtractor)
sim.add_process(multiplier)


traces = [*dut_adder.ports(), *dut_subtractor.ports(), *dut_multiplier.ports()]

with sim.write_vcd("math_ops_tb.vcd", "math_ops_tb.gtkw", traces=traces):
    sim.run()
