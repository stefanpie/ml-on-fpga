from nmigen import *
from nmigen.cli import main

import math
from fxpmath import Fxp

import networkx as nx
import matplotlib.pyplot as plt

class Popcount(Elaboratable):
    def __init__(self, input_size=32, output_size=None, method="tree"):
        self.method = method
        self.input_size = input_size
        self.min_output_size = math.ceil(math.log2(self.input_size)) + 1
        if output_size:
            assert(output_size >= self.min_output_size)
            self.output_size = output_size
        else:
            self.output_size = self.min_output_size
        self.x = Signal(unsigned(self.input_size), name="x")
        self.y = Signal(unsigned(self.output_size), name="y")
    
    def elaborate(self, platform):
        m = Module()
        if self.method == "tree":
            tree = nx.balanced_tree(2, self.min_output_size-1,create_using=nx.DiGraph)
            
            for node in tree.nodes():
                tree.nodes[node]["using"] = False
                node_level = nx.shortest_path_length(tree, 0, node)
                signal_size = self.min_output_size-node_level+1
                tree.nodes[node]["level"] = node_level
                tree.nodes[node]["signal"] = Signal(unsigned(signal_size), name=f"{node}")
            
            leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x) == 0 and tree.in_degree(x) == 1]
            for i in range(self.input_size):
                leaf_node = leaf_nodes[i]
                tree.nodes[leaf_node]["using"] = True
                m.d.comb += tree.nodes[leaf_node]["signal"].eq(self.x[i])
            # print(leaf_nodes)

            for level in range(self.min_output_size):
                leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x) == 0 and tree.in_degree(x) == 1]
                for leaf_node in leaf_nodes:
                    if not tree.nodes[leaf_node]["using"]:
                        tree.remove_node(leaf_node)
            
            tree = tree.reverse()

            leaf_nodes = [x for x in tree.nodes() if tree.out_degree(x) == 1 and tree.in_degree(x) == 0]
            node_op_order = list(nx.topological_sort(tree))
            # print(leaf_nodes)
            # print(node_op_order)
            
            for n in node_op_order:
                if tree.out_degree(n) == 1 and tree.in_degree(n) == 0:
                    # print(f"leaf node: {n}")
                    ...
                elif tree.in_degree(n) == 1 and tree.out_degree(n) == 1:
                    # print(f"chain link: {n}")
                    in_node = list(tree.predecessors(n))[0]
                    # out_node = list(tree.successors(n))[0]
                    m.d.comb += tree.nodes[n]["signal"].eq(tree.nodes[in_node]["signal"])
                elif tree.in_degree(n) >= 2:
                    in_nodes = list(tree.predecessors(n))
                    in_node_signals = [tree.nodes[i]["signal"] for i in in_nodes]
                    # print(in_node_signals)
                    m.d.comb += tree.nodes[n]["signal"].eq(sum(in_node_signals))
                    # print(f"summing link: {n}")
                
            m.d.comb += self.y.eq(tree.nodes[0]["signal"])
                

            # pos = nx.nx_pydot.graphviz_layout(tree, prog="dot")
            # labels = dict(tree.nodes(data="signal"))
            # # print(labels)
            # nx.draw(tree, pos=pos, labels=labels)
            # plt.show()

        elif self.method == "ladder":
            m.d.comb += self.y.eq(sum(iter(self.x)))
        else:
            raise NotImplementedError(f"Method \"{self.method}\" not implemented")
        return m
    
    @property
    def ports(self):
        return [self.x, self.y]


if __name__ == "__main__":
    popcount = Popcount(input_size=32, method="tree")
    main(popcount, ports=popcount.ports)
