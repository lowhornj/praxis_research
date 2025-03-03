from dowhy import gcm
import networkx as nx
import numpy as np
import pandas as pd

class CausalInference:
    def __init__(self, adj_matrix, new_data, old_data, excluded_nodes=None):
        self.adj_matrix = adj_matrix
        np.fill_diagonal(self.adj_matrix, 0)
        self.new_data = new_data
        self.old_data = old_data
        self.cols = list(new_data.columns)
        self.excluded_nodes = set(excluded_nodes) if excluded_nodes else set()

        # Build initial graph
        self.causal_graph = nx.DiGraph(self.adj_matrix)
        self.remove_excluded_nodes()
        self.ensure_dag()
        self.build_causal_model()

    def remove_excluded_nodes(self):
        """Remove explicitly excluded nodes from the causal graph."""
        for node in list(self.causal_graph.nodes):
            if node in self.excluded_nodes:
                self.causal_graph.remove_node(node)

    def ensure_dag(self):
        """
        Fully removes all cycles using Tarjan's SCC Algorithm.
        It continues removing edges until no cycles remain.
        """
        print("🔍 Checking for cycles using Tarjan’s SCC Algorithm...")

        while True:
            sccs = list(nx.strongly_connected_components(self.causal_graph))
            cycles = [scc for scc in sccs if len(scc) > 1]  # Only keep actual cycles

            if not cycles:
                print("✅ No cycles detected! Graph is now a DAG.")
                return

            #print(f"⚠️ Cycles detected: {cycles}. Removing edges...")

            for cycle in cycles:
                cycle_edges = [(a, b) for a in cycle for b in self.causal_graph.neighbors(a) if b in cycle]

                # Remove the edge that has the highest out-degree (most influential)
                edge_to_remove = max(cycle_edges, key=lambda e: self.causal_graph.out_degree(e[0]))
                self.causal_graph.remove_edge(*edge_to_remove)

                #print(f"🛑 Removed edge {edge_to_remove} to break cycle.")

    def build_causal_model(self):
        """Constructs the probabilistic causal model using DoWhy."""
        mapping = {i: col for i, col in enumerate(self.cols)}
        self.causal_graph = nx.relabel_nodes(self.causal_graph, mapping)
        self.causal_model_prob = gcm.ProbabilisticCausalModel(self.causal_graph)
        gcm.auto.assign_causal_mechanisms(self.causal_model_prob, self.new_data, override_models=True)
        gcm.fit(self.causal_model_prob, self.new_data)

    def infer_causal_effect(self, cause, effect):
        """Handles standard causal effect inference."""
        return gcm.average_causal_effect(
            self.causal_model_prob, effect,
            interventions_alternative={cause: lambda x: 1},
            interventions_reference={cause: lambda x: 0},
            observed_data=self.new_data
        )

    def infer_counterfactual(self, interventions, num_runs=10):
        """
        Generates counterfactual samples for given interventions.
        
        - **Key Fix:** Resets noise data before each run to prevent identical counterfactuals.
        """
        intervention_dict = {var: (lambda x: value) for var, value in interventions.items()}
        
        # Run counterfactual simulations multiple times
        all_results = []
        for _ in range(num_runs):
            # Deep copy noise data to introduce variability
            noise_data_sampled = self.new_data.sample(frac=1, replace=True).reset_index(drop=True)

            cf_result = gcm.counterfactual_samples(
                self.causal_model_prob,
                intervention_dict,
                noise_data=noise_data_sampled
            )
            all_results.append(cf_result)

        # Average the results across multiple runs
        avg_results = pd.concat(all_results).groupby(level=0).mean()
        return avg_results


    def rank_causal_factors(self, target):
        """Ranks the top causal factors influencing the target variable."""
        causal_effects = {
            cause: self.infer_causal_effect(cause, target)
            for cause in self.causal_graph.predecessors(target)
            if cause not in self.excluded_nodes
        }
        return sorted(causal_effects.items(), key=lambda item: item[1], reverse=True)

    def draw_causal_dag(self):
        """Visualizes the causal DAG."""
        nx.draw(self.causal_graph, with_labels=True, node_size=800, node_color="skyblue",
                font_size=10, font_weight="bold", arrowsize=20)

class CausalPipeline:
    def __init__(self, causal_inference):
        self.causal_inference = causal_inference

    def run_pipeline(self, target_variable, num_runs=10):
        """
        Runs separate counterfactual analyses for each top causal factor.
        """
        # 1️⃣ Rank the most important causal factors
        ranked_causes = self.causal_inference.rank_causal_factors(target_variable)

        if not ranked_causes:
            print(f"⚠️ No causal factors found for {target_variable}. Exiting pipeline.")
            return None

        # Take the top 3 causes (or fewer if not available)
        top_causes = ranked_causes

        # 2️⃣ Estimate the direct causal effects for the top causes
        estimated_effects = {
            cause: self.causal_inference.infer_causal_effect(cause, target_variable)
            for cause, _ in top_causes
        }

        # Filter out None values (due to instantaneous effects)
        estimated_effects = {k: v for k, v in estimated_effects.items() if v is not None}

        if not estimated_effects:
            print(f"⚠️ No valid causal effects found for {target_variable}. Exiting pipeline.")
            return None

        # 3️⃣ Generate counterfactual outcomes for EACH cause separately
        counterfactual_results = {}
        observed_value = self.causal_inference.new_data[target_variable].mean()

        for cause in estimated_effects.keys():
            intervention = {cause: 1}  # Apply intervention only on this cause

            # Generate counterfactual results with multiple runs for variability
            cf_result = self.causal_inference.infer_counterfactual(intervention, num_runs=num_runs)

            if target_variable in cf_result:
                counterfactual_value = np.mean(cf_result[target_variable])
                counterfactual_results[cause] = {
                    "counterfactual_value": counterfactual_value,
                    "difference_due_to_intervention": counterfactual_value - observed_value
                }
            else:
                print(f"⚠️ Counterfactual results missing for {target_variable} under intervention {cause}.")
                counterfactual_results[cause] = None

        # 4️⃣ Store structured results
        results = {
            "target_variable": target_variable,
            "top_causes": list(estimated_effects.keys()),
            "estimated_effects": estimated_effects,
            "observed_value": observed_value,
            "counterfactual_results": counterfactual_results
        }

        return results
