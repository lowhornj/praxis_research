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
        #self.adj_matrix[self.adj_matrix < 0.5] = 0
        np.fill_diagonal(self.adj_matrix, 0, wrap=False)
        self.new_data = new_data
        self.old_data = old_data
        self.cols = list(new_data.columns)
        self.causal_graph = nx.DiGraph(self.adj_matrix)
        self.num_cycles = len(list(nx.simple_cycles(self.causal_graph)))
        print('Number of cycles: '+ str(len(list(nx.simple_cycles(self.causal_graph))) ))
        if len(list(nx.simple_cycles(self.causal_graph))) > 0:
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
        print('Building Causal Models')
        print('-------------------------------------------------------------------------')
        mapping = {}
        for key, value in zip(list(self.causal_graph.nodes),self.cols):
            mapping[key] = value
        
        self.causal_graph = nx.relabel_nodes(self.causal_graph, mapping)
    
        self.causal_model_prob = gcm.ProbabilisticCausalModel(self.causal_graph)
        self.causal_model_struct = gcm.StructuralCausalModel(self.causal_graph)
        self.causal_model_inv = gcm.InvertibleStructuralCausalModel(self.causal_graph)
        gcm.auto.assign_causal_mechanisms(self.causal_model_prob, self.new_data,override_models=True)
        gcm.auto.assign_causal_mechanisms(self.causal_model_struct, self.new_data,override_models=True)
        gcm.auto.assign_causal_mechanisms(self.causal_model_inv,self. new_data,override_models=True)
        gcm.fit(self.causal_model_prob, self.new_data)
        gcm.fit(self.causal_model_struct, self.new_data)
        gcm.fit(self.causal_model_inv, self.new_data)

    def infer_causal_path(self,anomaly):
        print('Infering Causality')
        print('-------------------------------------------------------------------------')
        self.potential_causes = list(self.causal_graph.predecessors(anomaly))
        # Infer from adjacency weights
        adj_data = pd.DataFrame(self.adj_matrix,index=self.cols,columns=self.cols)
        candidates = adj_data[anomaly].sort_values(ascending=False)
        candidates = candidates[candidates.index.isin(self.potential_causes)]
        candidates_dict = candidates.to_dict()
        self.candidates_causes = {}
        for k, v in candidates_dict.items():
            if k not in non_causal_columns:
                self.candidates_causes[k] = v/np.max(list(candidates_dict.values()))

        # Infer from Dist Changes
        dist_changes = gcm.distribution_change(self.causal_model_prob, self.old_data, self.new_data, anomaly)
        if anomaly in dist_changes.keys():
            del dist_changes[anomaly]
        dist_changes_vals = np.abs(np.array(list(dist_changes.values())))
        self.dist_changes_dict = {}
        for k, v in dist_changes.items():
            if k != anomaly:
                self.dist_changes_dict[k] = np.abs(np.mean(v)/np.max(dist_changes_vals))
        
        # Infer from Average Causal Effects        
        causal_effects = {}
        for cause in self.potential_causes:
            causal_effects[cause] = np.abs(gcm.average_causal_effect(self.causal_model_prob, anomaly,
                                      interventions_alternative={cause:lambda x:1},
                                      interventions_reference={cause:lambda x:0},
                                      num_samples_to_draw=1000
                                     ))
        
        causal_effects_vals = np.array(list(causal_effects.values()))
        causal_effects_dict = {}
        for k, v in causal_effects.items():
            if k != anomaly:
                causal_effects_dict[k] = np.abs(np.mean(v)/np.max(causal_effects_vals))
        
        causal_effects_vals = np.array(list(causal_effects.values()))
        self.causal_effects_dict = {}
        for k, v in causal_effects.items():
            if k != anomaly:
                self.causal_effects_dict[k] = np.abs(np.mean(v)/np.max(causal_effects_vals))

        # Infer from Parent Relevance
        parent_relevance = gcm.parent_relevance(self.causal_model_struct,target_node=anomaly)
        parent_rel_vals = np.array(list(parent_relevance[0].values()))
        self.parent_rel_dict = {}
        for k, v in parent_relevance[0].items():
            if k != anomaly:
                self.parent_rel_dict[k[0]] = np.abs(np.mean(v)/np.max(parent_rel_vals))

        # Infer from Anomaly Attribution
        attributions  = gcm.attribute_anomalies(self.causal_model_inv,target_node=anomaly,anomaly_samples=self.new_data)
        if anomaly in attributions.keys():
            del attributions[anomaly]
        attributions_vals = np.mean(np.array(list(attributions.values())),axis=1)
        self.attributions_dict = {}
        for k, v in attributions.items():
            if k != anomaly:
                self.attributions_dict[k] = np.mean(v)/np.max(attributions_vals)

        # Infer from Intrinsic Influence
        intrinsic_influence = gcm.intrinsic_causal_influence(self.causal_model_struct,target_node=anomaly)
        if anomaly in intrinsic_influence.keys():
            del intrinsic_influence[anomaly]
        self.intrinsic_influence = {}
        for k, v in intrinsic_influence.items():
            if k != anomaly:
                self.intrinsic_influence[k]= v/np.max(list(intrinsic_influence.values()))

        # Infer from Arrow Strength
        arrow_strengths = gcm.arrow_strength(
            causal_model=self.causal_model_prob,
            target_node= anomaly)
    
        if anomaly in arrow_strengths.keys():
            del arrow_strengths[anomaly]
        
        self.out_arrow_strengths = {}
        for k, v in arrow_strengths.items():
            self.out_arrow_strengths[k[0]]= v/np.max(list(arrow_strengths.values()))
        
        # Infer from average Effect
        dictionaryList = [self.parent_rel_dict,self.attributions_dict,
                          self.out_arrow_strengths,self.candidates_causes,
                          self.causal_effects_dict,self.dist_changes_dict,
                         self.intrinsic_influence]
        self.average_effect = {}
        
        for key in self.candidates_causes.keys():
            self.average_effect[key] = np.mean([d[key] for d in dictionaryList ])
                                   
        self.top_causal_factors = sorted(self.average_effect.items(), key=lambda item: item[1], reverse=True)
                
    def draw_causal_dag(self):
        nx.draw(self.causal_graph, with_labels=True, node_size=800, node_color="skyblue", 
            font_size=10, font_weight="bold", arrowsize=20)
        