from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


from pprint import pprint as pp
from itertools import tee
from copy import deepcopy, copy

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

clf_tree = clf.tree_

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
        node_data["label"] = clf.classes_[np.argmax(clf_tree.value[i])]
    else:
        node_data["value"] = None
        node_data["label"] = None
    tree.add_node(i, **copy(node_data))

for id, data in tree.nodes(data=True):
    if data["left"] != -1:
        tree.add_edge(id, data["left"], comparison_result=True)
    if data["right"] != -1:
        tree.add_edge(id, data["right"], comparison_result=False)


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
    logic_path = [(edge[0], tree.edges[edge]['comparison_result']) for edge in pairwise(path)]
    logic_paths[path[-1]] = logic_path

pp(logic_paths)

pos = nx.drawing.nx_pydot.graphviz_layout(tree, prog="dot")


def get_node_label(node):
    if node["leaf"]:
        return node["label"]
    else:
        return f'x[{node["feature"]}] <= {round(node["threshold"], 2)}'


node_labels = {n: get_node_label(data) for n, data in tree.nodes(data=True)}
edge_labels = nx.get_edge_attributes(tree, 'comparison_result')
nx.draw(tree, pos=pos, labels=node_labels, node_color='none')
nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels)
plt.show()
