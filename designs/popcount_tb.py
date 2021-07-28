from nmigen import *
from nmigen.cli import main
from nmigen.sim import Simulator, Delay, Settle

import tqdm

import popcount

INPUT_SIZE = 32
OUTPUT_SIZE = 6
dut_popcount = popcount.Popcount(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, method="tree")

m = Module()
m.submodules.dut_popcount = dut_popcount

sim = Simulator(m)


def process_main():
    inputs = range(10000)

    results = []
    
    for i in tqdm.tqdm(inputs):
        popcount_sw = bin(i).count("1")
        yield dut_popcount.x.eq(i)
        yield Delay(1e-9)
        yield Settle()
        popcount_hw = yield dut_popcount.y

        # print(popcount_sw)
        # print(popcount_hw)

        # print(popcount_sw == popcount_hw)
        results.append((popcount_sw, popcount_hw, popcount_sw == popcount_hw))
    
    results_check = [r[2] for r in results]
    print((sum(results_check)/len(results_check))*100)


sim.add_process(process_main)


traces = [*dut_popcount.ports]

with sim.write_vcd("popcount_tb.vcd", "popcount_tb.gtkw", traces=traces):
    sim.run()
