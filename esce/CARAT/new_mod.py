from utils.utils import set_seed
set_seed()
n_correct = 0
total_checked = 0
for i, row in enumerate(cats_rows_list):
    print('Model: '+ str(i))
    total_checked +=1

    anomaly_time = datetime.strptime(row['start_time'],"%Y-%m-%d %H:%M:%S")
    #start_time = datetime.strptime(row['start_time'],"%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(row['end_time'],"%Y-%m-%d %H:%M:%S")
    start_time = anomaly_time- timedelta(seconds=500)
    anomaly = eval(row['affected'])[0]
    root_cause = row['root_cause']
    
    mod_df = test_df[(test_df.index>= anomaly_time) & (test_df.index<= end_time)]
    mod_df = mod_df[['aimp', 'amud', 'arnd', 'asin1', 'asin2', 'adbr', 'adfl', 'bed1',
       'bed2', 'bfo1', 'bfo2', 'bso1', 'bso2', 'bso3', 'ced1', 'cfo1', 'cso1']]
    num_nodes = len(mod_df.columns)

    start_len = mod_df.shape[0]
    if start_len >1000:
        start_len = 1000

    start_time = anomaly_time- timedelta(seconds=start_len)

    normal_df = test_df[(test_df.index>= start_time) & (test_df.index< anomaly_time)]

    sample_data = TKGNGCDataProcessor(mod_df,device,num_timestamps=20, lags=1)
    model_data = create_granger_gat_data(pretrained_tkg,sample_data)
    model_data.retrain_tkg()
    features = model_data.time_series_data
    model_name = 'model_category_'+str(row['category'])
    """model = CausalGraphVAE(input_dim=num_nodes, hidden_dim=128,
                   latent_dim=16, 
                   num_nodes=num_nodes, 
                   embed_dim=17,
                   prior_adj_matrix=A0.to(torch.float))"""
    model = CausalGraphVAE(input_dim=num_nodes, hidden_dim=128,
                   latent_dim=16, 
                   num_nodes=num_nodes, 
                   embed_dim=17,
                   prior_adj_matrix=A0.to(torch.float))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data = create_lagged_features(features, 3, pad_value=0)
    ee = create_lagged_features(model_data.entity_emb, 3, pad_value=0)
    tt = create_lagged_features(model_data.timestamp_emb, 3, pad_value=0)
        
    dataset = TimeSeriesDataset(data, ee, tt)
    dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, shuffle=False)
    loss = train_causal_vae(model, optimizer, dataloader, A0, num_epochs=500)
    model.eval()
    with torch.no_grad():
        _, _, _, learned_adj_matrix = model(data,ee,tt, num_nodes=num_nodes)
        causal_graph = learned_adj_matrix.cpu().numpy()

    ci = causal_inference(causal_graph, mod_df, normal_df)
    ci.infer_causal_path(anomaly)    
    adj_data = pd.DataFrame(causal_graph,index=cols,columns=cols)
    candidates = adj_data[anomaly].sort_values(ascending=False)
    candidates = candidates[candidates.index.isin( potential_causes)]

    if len(ci.causal_factors) >= 3:
        potential_cause1 = ci.causal_factors[0][0]
        potential_cause2 = ci.causal_factors[1][0]
        potential_cause3 = ci.causal_factors[2][0]
    elif len(ci.causal_factors) == 2:
        potential_cause1 = ci.causal_factors[0][0]
        potential_cause2 = ci.causal_factors[1][0]
        potential_cause3 = 'NA'
    elif len(ci.causal_factors) == 1:
        potential_cause1 = ci.causal_factors[0][0]
        potential_cause2 = "NA"
        potential_cause3 = 'NA'
    else:
        potential_cause1 = candidates.index[0]
        potential_cause2 = candidates.index[1]
        potential_cause3 = candidates.index[2]

    if root_cause in [potential_cause1,potential_cause2,potential_cause3]:
        n_correct+=1

    if root_cause == potential_cause1:
        row['cause_1'] = 1
    if root_cause == potential_cause2:
        row['cause_2'] = 1
    if root_cause == potential_cause3:
        row['cause_3'] = 1
    new_metadata.append(row)
    if root_cause in [potential_cause1 , potential_cause2 , potential_cause3]:
        print('ROOT CAUSE FOUND!')
    print(root_cause + '-->' + potential_cause1 + ' | ' + potential_cause2 + ' | ' + potential_cause3)

    print('Model Accuracy: '+ str(round((n_correct/total_checked)*100,2)) + '%')
    
    print('--------------------------------------------------------------------------------------------------------------------------------')
