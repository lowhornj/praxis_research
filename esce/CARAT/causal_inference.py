from sortedcontainers import SortedSet
import networkx as nx
import matplotlib.pyplot as plt
from dowhy import gcm 
import numpy as np
import pandas as pd

class causal_inference:
    def __init__(self, adj_matrix, new_data, old_data):
        super(causal_inference, self).__init__()
        self.adj_matrix=adj_matrix
        self.adj_matrix[self.adj_matrix < 0.5] = 0
        np.fill_diagonal(self.adj_matrix, 0, wrap=False)
        self.new_data = new_data
        self.old_data = old_data
        self.cols = list(new_data.columns)
        self.causal_graph = nx.DiGraph(self.adj_matrix)
        self.topological_remove_cycles(self.causal_graph)   
        self.build__causal_model()

    def topological_remove_cycles(self, g):

        # Taken from: https://stackoverflow.com/questions/78348739/converting-cyclic-digraph-to-acyclic-digraph-dag
        # Dictionary of sets of incoming edges. We want to pick nodes with few of them.
        incoming = {}
        for node in g.nodes:
            incoming[node] = set()
        for node in g.nodes():
            for next_node in g.neighbors(node):
                incoming[next_node].add(node)
    
        # Sorted set of (count_incoming, -count_outgoing, node) triplets.
        # The first item in the set will have as few incoming nodes as it can, and as many outgoing.
        # In other words, it is a greedy choice for a good node to get rid of cycles.
        todo = SortedSet()
        for node, to_node in incoming.items():
            todo.add((len(to_node), -len(g.adj[node]), node))
    
        # And now let's start removing cycles.
        while 0 < len(todo):
            # Get the best node.
            _, _, node = todo.pop(0)
            to_node = incoming[node]
            for prev_node in to_node:
                # Each of these edges is in, or comes from, a cycle.
                if prev_node != node:
                    # Do bookkeeping in todo.
                    len_in = len(incoming[prev_node])
                    len_out = len(g.adj[prev_node])
                    todo.remove((len_in, -len_out, prev_node))
                    todo.add((len_in, -len_out + 1, prev_node))
                g.remove_edge(prev_node, node)
    
            for next_node in g.neighbors(node):
                # Do bookkeeping in todo for forgetting about node.
                len_in = len(incoming[next_node])
                len_out = len(g.adj[next_node])
                todo.remove((len_in, -len_out, next_node))
                todo.add((len_in - 1, -len_out, next_node))
                incoming[next_node].remove(node)
            # And now we've guaranteed that node is cycle free and gone from our bookkeeping.

    def build__causal_model(self):
        mapping = {}
        for key, value in zip(list(self.causal_graph.nodes),self.cols):
            mapping[key] = value
        
        self.causal_graph = nx.relabel_nodes(self.causal_graph, mapping)
        self.causal_model = gcm.ProbabilisticCausalModel(self.causal_graph)
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.new_data,override_models=True)
        gcm.fit(self.causal_model, self.new_data)

    def infer_causal_path(self,anomaly):
        #self.summary_evaluation = gcm.evaluate_causal_model(self.causal_model, self.new_data, 
        #                                                    compare_mechanism_baselines=True)

        self.causal_attributions = gcm.distribution_change(self.causal_model, self.old_data, self.new_data, anomaly)
        abs_attributions = {}
        for k, v in self.causal_attributions.items():
            if k != anomaly:
                abs_attributions[k] = abs(v)
        
        self.causal_factors = sorted(abs_attributions.items(), key=lambda item: item[1], reverse=True)
                
    def draw_causal_dag(self):
        nx.draw(self.causal_graph, with_labels=True, node_size=800, node_color="skyblue", 
            font_size=10, font_weight="bold", arrowsize=20)
            
        
        