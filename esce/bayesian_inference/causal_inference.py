import pyro
from pyro.distributions import Normal, Categorical, Independent, OneHotCategorical
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from matplotlib import pyplot as plt
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference.CausalInference import CausalInference
from IPython.display import Image
import torch
import pandas as pd

class bayesian_network_inference:
    def __init__(self,anomaly,vae_model,model_dataframe,input_numpy_data, scoring_method = 'bdeuscore'):
        self.anomaly = anomaly
        self.vae_model = vae_model
        self.mod_df = model_dataframe
        self.column_names = self.mod_df.columns
        self.input_numpy_data = input_numpy_data
        self.scoring_method=scoring_method
        self.latent_df = None
        self.bayesian_model = None
        self.rca_scores = None
        self.potential_causes = []

    def _get_latent_variables(self):
        self.vae_model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(self.input_numpy_data, dtype=torch.float32)
            mu, _ = self.vae_model.encode(data_tensor)
            return mu.numpy()

    def _prep_latent_data(self):
        latent_data = self._get_latent_variables()
        self.latent_df = pd.DataFrame(latent_data, columns=[f"z{i}" for i in range(latent_data.shape[1])])
        self.latent_df.columns = self.mod_df.columns

    def _train_bayes(self):
        # Train Bayesian Network
        hc = HillClimbSearch(self.latent_df)
        best_model = hc.estimate(scoring_method=self.scoring_method)
        self.bayesian_model = BayesianNetwork(best_model.edges())
        self.bayesian_model.fit(self.latent_df, estimator=MaximumLikelihoodEstimator)

    def draw_graph(self,name):
        viz = bayesian_model.to_graphviz()
        viz.draw( name + '.png', prog='neato')
        Image(name + '.png')

    def causal_inference(self,n_causes=3):
        self._prep_latent_data()
        self._train_bayes()

        inference = CausalInference(self.bayesian_model)
        scores = []
        columns = self.latent_df.columns
        test_columns = []
        for col in columns:
            if col != self.anomaly:
                try:
                    inference.get_all_backdoor_adjustment_sets(X=col, Y=self.anomaly)
                except:
                    None
                try:
                    inference.get_all_frontdoor_adjustment_sets(X=col, Y=self.anomaly)
                except:
                    None
                try:
                    test_columns.append(col)
                    scores.append(inference.estimate_ate(X=col, Y=self.anomaly, data=self.latent_df, estimator_type="linear"))
                except:
                    None

        rca_indices = sorted(range(len(scores)), key=lambda i: scores[i])[-n_causes:]
        self.rca_scores = dict(zip(test_columns,scores))
        for index in rca_indices:
            self.potential_causes.append( (test_columns[index]))
        self.potential_causes.sort(key=lambda tup: tup[1],reverse=True)


 
        