
from nmigen import *
from nmigen.cli import main
from nmigen.lib.coding import Encoder

from fxpmath import Fxp

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from pprint import pprint as pp
from itertools import tee
from copy import deepcopy, copy
import math
from operator import and_, or_
from functools import reduce
from typing import OrderedDict


class DecisionTreeClassifierHW(Elaboratable):
    def __init__(self, trained_model: DecisionTreeClassifier, bit_depth=32, y_bit_depth=None):
        self.model = trained_model
        self.x_size = trained_model.n_features_

        self.n_classes = self.model.n_classes_.item()

        self.bit_depth = bit_depth
        min_y_bit_depth = math.ceil(math.log2(self.n_classes))
        if y_bit_depth:
            assert(y_bit_depth >= min_y_bit_depth)
            self.y_bit_depth = y_bit_depth
        else:
            self.y_bit_depth = min_y_bit_depth

        # print(self.y_bit_depth)

        self.x = [Signal(signed(self.bit_depth), name=f"x_{i}") for i in range(self.x_size)]
        self.y = Signal(unsigned(self.y_bit_depth), name="y")
        self.valid = Signal(unsigned(1), name="valid")

    def elaborate(self, platform):

        m = Module()

        clf = self.model
        clf_tree = clf.tree_
        # print(clf.classes_)

        # print(clf_tree.node_count)
        # for i in range(clf_tree.node_count):
        #     print(f"node_id: {i}")
        #     print(f"|-children_left: {clf_tree.children_left[i]}")
        #     print(f"|-children_right: {clf_tree.children_right[i]}")
        #     print(f"|-feature: {clf_tree.feature[i]}")
        #     print(f"|-threshold: {clf_tree.threshold[i]}")
        #     print(f"|-value: {clf_tree.value[i]}")

        tree = nx.DiGraph()
        for i in range(clf_tree.node_count):
            node_data = {}
            node_data["node_id"] = i
            node_data["left"] = clf_tree.children_left[i]
            node_data["right"] = clf_tree.children_right[i]
            node_data["leaf"] = clf_tree.children_left[i] == -1 and clf_tree.children_right[i] == -1
            node_data["feature"] = clf_tree.feature[i]
            node_data["threshold"] = clf_tree.threshold[i]
            if node_data["leaf"]:
                node_data["value"] = clf_tree.value[i]
                node_data["label_idx"] = np.argmax(clf_tree.value[i])
                node_data["label_value"] = clf.classes_[np.argmax(clf_tree.value[i])]
            else:
                node_data["value"] = None
                node_data["label_idx"] = None
                node_data["label_value"] = None
            tree.add_node(i, **copy(node_data))

        for id, data in tree.nodes(data=True):
            if data["left"] != -1:
                tree.add_edge(id, data["left"], comparison_result=True)
            if data["right"] != -1:
                tree.add_edge(id, data["right"], comparison_result=False)

        comparisons = {}
        for id, data in tree.nodes(data=True):
            if not data["leaf"]:
                comparisons[id] = Signal(unsigned(1), name=f"comp_{id}")
                feature_index_to_compare = data["feature"]
                comparison_value = data["threshold"]
                comparison_value_fixed_point = Fxp(comparison_value, True, self.bit_depth, self.bit_depth//2)
                comparison_value_fixed_point_base_10 = int(comparison_value_fixed_point.base_repr(10))
                m.d.comb += comparisons[id].eq(self.x[feature_index_to_compare] <= comparison_value_fixed_point_base_10)

        paths = []
        for node in tree:
            if tree.out_degree(node) == 0:  # it's a leaf
                paths.append(nx.shortest_path(tree, 0, node))

        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return list(zip(a, b))

        logic_paths = {}
        for path in paths:
            # print(path[-1])
            # print(path)
            logic_path = [(edge[0], tree.edges[edge]['comparison_result']) for edge in pairwise(path)]
            logic_paths[path[-1]] = logic_path

        # pp(logic_paths)

        # pp(logic_paths)
        leaf_reaches = {}
        for id, data in tree.nodes(data=True):
            if data["leaf"]:
                leaf_reaches[id] = Signal(unsigned(1), name=f"leaf_reach_{id}")
        # pp(leaf_reaches)

        for leaf_id in logic_paths:
            # print(leaf_id)
            # print(logic_paths[leaf_id])
            path_of_signals = []
            for step in logic_paths[leaf_id]:
                if step[1] == False:
                    path_of_signals.append(~comparisons[step[0]])
                else:
                    path_of_signals.append(comparisons[step[0]])
            # leaf_path_expression = reduce(and_, path_of_signals)
            leaf_path_expression = Cat(path_of_signals).all()
            # print(leaf_path_expression)
            m.d.comb += leaf_reaches[leaf_id].eq(leaf_path_expression)

        label_selects = {}
        for label_idx in range(self.n_classes):
            label_selects[label_idx] = Signal(unsigned(1), name=f"label_select_{label_idx}")
        # pp(label_selects)

        for label_idx in range(self.n_classes):
            label_leaves = [leaf_id for leaf_id, leaf_data in tree.nodes(
                data=True) if leaf_data["leaf"] and leaf_data["label_idx"] == label_idx]
            label_leaves_reach_signals = [leaf_reaches[leaf_id] for leaf_id in label_leaves]
            # pp(label_leaves_reach_signals)
            # label_select_expression = reduce(or_, label_leaves_reach_signals)
            label_select_expression = Cat(label_leaves_reach_signals).any()
            # input("...")
            m.d.comb += label_selects[label_idx].eq(label_select_expression)

        # input("...")
        # print(type(self.n_classes))
        encoder = Encoder(self.n_classes)
        m.submodules += encoder
        for label_idx, label_select in label_selects.items():
            m.d.comb += encoder.i[label_idx].eq(label_select)

        for i in range(self.y_bit_depth):
            m.d.comb += self.y[i].eq(encoder.o[i])

        m.d.comb += self.valid.eq(~encoder.n)

        return m

    def ports(self):
        return [*self.x, self.y, self.valid]


if __name__ == "__main__":

    dataset = load_breast_cancer()

    x = dataset["data"]
    y = dataset["target"]

    pp(x.shape)
    pp(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)

    acc = clf.score(x_test, y_test)
    print(acc)

    decision_tree_classifier_hw = DecisionTreeClassifierHW(clf, 32)
    main(decision_tree_classifier_hw, ports=decision_tree_classifier_hw.ports())
