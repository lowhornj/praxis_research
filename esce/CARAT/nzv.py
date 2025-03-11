 """
    NZV Removal
    """
    var_removal = mod_df.var()<1e-6
    var_removal = var_removal.reset_index()
    var_removal.columns = ['index','var']
    var_removal[var_removal['index']==anomaly] = [anomaly,False]
    nzv_vars = var_removal[var_removal['var']==True]['index'].to_list()
    mod_df = mod_df.drop(nzv_vars,axis=1)
    num_nodes = len(mod_df.columns)

    priors_drops = []
    for col in nzv_vars:
        priors_drops.append(test_df.columns.get_loc(col))
        
    priors = np.delete(A0, priors_drops, axis=0)
    priors = np.delete(priors, priors_drops, axis=1)
    priors = torch.tensor(priors,dtype=torch.float32)
    curr_non_causal = non_causal_indices.copy()
    for var in sorted(priors_drops,reverse=True):
        try:
            curr_non_causal.remove(var)
        except:
            None

    cols = mod_df.columns.tolist()
