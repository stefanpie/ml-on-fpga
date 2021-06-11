
import power_function
from fxpmath import Fxp

from nmigen import *
from nmigen.cli import main
from nmigen.sim import Simulator, Delay, Settle

dut_power_function = power_function.PowerFunction(0)

m = Module()
m.submodules.dut_power_function = dut_power_function

sim = Simulator(m)


def process_main():

    x = Fxp(0.26589, True, 32, 16)
    yield dut_power_function.x.eq(int(x.base_repr(10)))
    yield Delay(1e-6)
    yield Settle()
    output = yield dut_power_function.y
    out_fpx = Fxp(output, True, 32, 16, raw=True)
    print("Power Function Results")
    print(x)
    print(out_fpx)
    print()


sim.add_process(process_main)


traces = [*dut_power_function.ports()]

with sim.write_vcd("power_function_tb.vcd", "power_function_tb.gtkw", traces=traces):
    sim.run()
