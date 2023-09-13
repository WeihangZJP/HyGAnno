#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:43:07 2023

@author: weihangzhang
"""

### CONFIGS ###
import torch
import os
import numpy as np
import pandas as pd
dir_path=os.path.dirname(os.path.abspath(__file__))

PATH_feature=[dir_path+"/HyGAnno_inputs/Feature_matrices/GEM.mtx",
              dir_path+"/HyGAnno_inputs/Feature_matrices/PM.mtx",
              dir_path+"/HyGAnno_inputs/Feature_matrices/GAM.mtx"]

PATH_graph=[dir_path+"/HyGAnno_inputs/Graphs/RNA_graph.csv",
            dir_path+"/HyGAnno_inputs/Graphs/ATAC_graph.csv",
            dir_path+"/HyGAnno_inputs/Graphs/Anchor_graph.csv"]


taget_label=False

use_GPU=torch.cuda.is_available()

hidden_hyg_dim1=128
hidden_hyg_dim2=len(set(np.array(pd.read_csv(dir_path+"/Raw_RNA/reference_label.csv")["cluster_id"]))) # cell type number
    

hidden_atac_dim1=128
hidden_atac_dim2=len(set(np.array(pd.read_csv(dir_path+"/Raw_RNA/reference_label.csv")["cluster_id"])))

learning_rate=0.0001
epoch=10