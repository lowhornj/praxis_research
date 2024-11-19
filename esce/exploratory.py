# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import polars as pl
import altair as alt
alt.data_transformers.disable_max_rows()

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
   
   
cats_df = pl.read_csv("data/data.csv", separator=",")  


alt.Chart(cats_df).transform_density(
    'aimp',
    as_=['aimp', 'density'],
).mark_area().encode(
    x="aimp:Q",
    y='density:Q',
)