import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pprint import pprint as pp
from itertools import tee
from copy import deepcopy, copy

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from fxpmath import Fxp

from nmigen import *
from nmigen.cli import main
from nmigen.sim import Simulator, Delay, Settle
from nmigen_boards.icebreaker import *

import decision_tree


print("### Creating Decision Tree Model")


dataset = load_breast_cancer()

x = dataset["data"]
y = dataset["target"]

# pp(x.shape)
# pp(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)

# acc = clf.score(x_test, y_test)
# print(acc)

y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))

print("### Building HW Testbench ###")

BIT_DEPTH = 32

dut_decision_tree = decision_tree.DecisionTreeClassifierHW(clf, bit_depth=BIT_DEPTH)
print(dut_decision_tree)

m = Module()
m.submodules.decision_tree = dut_decision_tree


def process_main():
    prediction_results = []
    for sample_idx in range(x_test.shape[0]):
        x_test_0 = x_test[sample_idx]
        y_test_0 = y_test[sample_idx]
        y_pred_0 = clf.predict(x_test_0.reshape(1, -1))[0]

        x_test_0 = x_test_0.tolist()


        for i, x in enumerate(x_test_0):
            x_val_fpx = Fxp(x, True, BIT_DEPTH, BIT_DEPTH//2)
            x_val_fpx_base_10 = int(x_val_fpx.base_repr(10))
            yield dut_decision_tree.x[i].eq(x_val_fpx_base_10)

        yield Delay(1e-6)
        yield Settle()

        y_output = yield dut_decision_tree.y
        valid_output = yield dut_decision_tree.valid

        prediction_results.append(y_pred_0 == y_output)


        print(f"Sample {sample_idx}")
        print(f"x_sample: {x_test_0}")
        print(f"y_predicted: {y_pred_0}")
        print(f"y_predicted_hw: {y_output}")
        print(f"hardware_correct: {y_pred_0 == y_output}")
        print()

    print( sum(prediction_results) / x_test.shape[0] )



sim = Simulator(m)
sim.add_process(process_main)

traces = [*dut_decision_tree.ports()]

print("Done")
print()

print("### Running Sumliation ###")
with sim.write_vcd("decision_tree_tb.vcd", "decision_tree_tb.gtkw", traces=traces):
    sim.run()
print("Simualtion Complete")
print()
