# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import polars as pl
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
   
   
cats_df = pl.read_csv("data/data.csv", separator=",")  
