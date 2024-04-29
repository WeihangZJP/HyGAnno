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

PATH_reference_label=dir_path+"/Raw_RNA/reference_label.csv"
PATH_target_label=dir_path+"/Raw_ATAC/target_label.csv"
taget_label=True

use_GPU=torch.cuda.is_available()

hidden_hyg_dim1=128
hidden_hyg_dim2=len(set(np.array(pd.read_csv(PATH_reference_label)["cluster_id"]))) # cell type number
    

hidden_atac_dim1=128
hidden_atac_dim2=len(set(np.array(pd.read_csv(PATH_reference_label)["cluster_id"])))

learning_rate=0.0001
epoch=800
