import scipy
from scipy import io
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing
import configs
import os
dir_path=os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------------------------------------------------------#
# function name: feature_matrix_import
# Description: Import the feature matrices as the inputs of HyGAnno

## inputs: automatically detect the raw count matrices in "/Feature_matrices" folder,
##---------which are "GEM.mtx", "GAM.mtx" and "PM.mtx"

## return: A dict of feature matrix objects: key=["GEM","GAM","PM"]
##---------A dict of label list key=["ref_label","tar_label]

## output files: No output files

def feature_matrix_import(PATH_feature):
    rna_features=scipy.io.mmread(PATH_feature[0]).todense()# import the gene expression matrix (GEM) of reference
    atac_features=scipy.io.mmread(PATH_feature[1]).todense()# import the peak matrix (PM) of target
    gam_features=scipy.io.mmread(PATH_feature[2]).todense()# import the gene activity matrix (GAM) of target 
    print("#--------------------Feature matrices are imported--------------------#")
    print("Shape of GEM (genes-by-cells):",rna_features.shape)
    print("Shape of PM (peaks-by-cells):",atac_features.shape)
    print("Shape of GAM (genes-by-cells):",gam_features.shape)
    ##import reference labels
    reference_label = np.array(pd.read_csv(dir_path+"/Raw_RNA/reference_label.csv")["cluster_id"])
    
    ##import target labels
    if configs.target_label:
        if os.path.isfile(dir_path+"/Raw_ATAC/target_label.csv"):
            target_label = np.array(pd.read_csv(dir_path+"/Raw_ATAC/target_label.csv")["cluster_id"])
        else:
            raise Exception("Target labels should be provided or set target_label to False in configs.py.")

    else:
        target_label=None
        print("Target labels are unknown, no validation during training process!!")
    
    #check if the matrix is proper for training 
    if rna_features.shape[0]!=gam_features.shape[0]:
      raise Exception("The numbers of gene features of GAM and GEM are inconsistent !!! ")
      
    if configs.target_label:
        if len(reference_label)!=rna_features.shape[1] or len(target_label)!=atac_features.shape[1]:
            raise Exception("The numbers of cells in label list and feature matrix is inconsistent !!!")
    
    return {"GEM":rna_features,"GAM":gam_features,"PM":atac_features},{"ref_label":reference_label,"tar_label":target_label}
#------------------------------------------------------------------------------------------------------------------------#








#------------------------------------------------------------------------------------------------------------------------#
# function name: graph_processing
# Description: Process the input ".csv" graphs to normalized graphs
# norm A=D^(-1/2)(A)D^(1/2)


## inputs: 1. Array format graph (two columns),
##---------2. "True": return the nomalized graph; 
##------------"False": return label graph (all elements are set to 1)

## return: nomalized graph or label graph

## output files: No output files

def graph_processing(graph,node_number,norm=True):
    
    data=[1 for i in range(len(graph))]
    graph_coo=sp.coo_matrix((data, (graph[:,0], graph[:,1])), shape=(node_number, node_number))
    graph_coo=graph_coo+graph_coo.transpose()
    graph_coo =graph_coo + sp.eye(graph_coo.shape[0])
    #set diag to be 0
    graph_mat=graph_coo.toarray()
    # set all elements larger than 0 to 1
    graph_mat=np.int64(graph_mat>0)
    graph_coo=sp.coo_matrix(graph_mat)
    if norm==False:
        #return graph_coo-sp.eye(graph_coo.shape[0])
        return graph_coo
    
    #normalize
    rowsum = np.array(graph_coo.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    graph_normalized = graph_coo.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return graph_normalized
#------------------------------------------------------------------------------------------------------------------------#








#------------------------------------------------------------------------------------------------------------------------#
# function name: graph_matrix_import
# Description: Import the graphs as the inputs of HyGAnno and construct the hybrid graph

## inputs: automatically detect the graphs in "/Graphs" folder,
##---------which are "RNA_graph.csv", "ATAC_graph.csv" and "Anchor_graph.csv"

## return: 1. A dict of graph objects: key=["hybrid","ATAC","initial","initial_label"]
##---------2. A dict recording the new index and the original index of ATAC anchor cells

## output files: No output files

def graph_matrix_import(feature_dict,PATH_graph):
    
    f1=open(PATH_graph[0],"rb")
    rna_rna_g=np.loadtxt(f1,delimiter=',',skiprows=1)
    f1.close()
    
    f1=open(PATH_graph[1],"rb")
    atac_atac_g=np.loadtxt(f1,delimiter=',',skiprows=1)
    f1.close()
    
    f1=open(PATH_graph[2],"rb")
    rna_atac_g=np.loadtxt(f1,delimiter=',',skiprows=1)
    f1.close()
    
    rna_cell_num=feature_dict["GEM"].shape[1]
    atac_cell_num=feature_dict["PM"].shape[1]

    print("#--------------------Graphs are imported--------------------#")
    
    #check if the graph is proper for training 
    if len(set(rna_rna_g[:,0]))!=rna_cell_num:
      raise Exception("Cell numbers in graph and feature matrix of GEM are inconsistent !!! ")
      
    if len(set(atac_atac_g[:,0]))!=atac_cell_num:
      raise Exception("Cell numbers in graph and feature matrix of PM are inconsistent !!! ")

    #ATAC graph construction
    atac_graph_norm=graph_processing(atac_atac_g,atac_cell_num)
    print("ATAC graph is constructed")
    
    #each atac index plus the total number of rna cell indexs
    for i in range(len(rna_atac_g)):
        rna_atac_g[i][1]+=rna_cell_num
    for i in range(len(atac_atac_g)):
        atac_atac_g[i][0]+=rna_cell_num
        atac_atac_g[i][1]+=rna_cell_num
    
    #the whole initial graph
    initial_graph=np.vstack((np.vstack((rna_atac_g,rna_rna_g)),atac_atac_g))
    initial_graph_norm=graph_processing(initial_graph,rna_cell_num+atac_cell_num)
    initial_graph_label=graph_processing(initial_graph,rna_cell_num+atac_cell_num,norm=False)

    #hybrid graph construction
    anchor_atac_ind=sorted(set(rna_atac_g[:,1]))
    anchor_atac_dict1=dict(zip(range(len(anchor_atac_ind)),anchor_atac_ind))# eg. key=[0,1,2...,number of ATAC anchor] <-> value=[ATAC anchor ind1,ATAC anchor ind2,...]
    anchor_atac_dict2=dict(zip(anchor_atac_ind,range(len(anchor_atac_ind))))# eg. key=[ATAC anchor ind1,ATAC anchor ind2,...] <-> value=[0,1,2...,number of ATAC anchor]
    for i in range(len(rna_atac_g)):
        rna_atac_g[i][1]=anchor_atac_dict2[rna_atac_g[i][1]]+rna_cell_num
    hybrid_graph=np.vstack((rna_rna_g,rna_atac_g))
    hybrid_graph_norm=graph_processing(hybrid_graph,rna_cell_num+len(anchor_atac_ind))
    print("hybrid graph is constructed")

    

    
    print("Shape of hybrid graph (RNA and anchor ATAC cells):",hybrid_graph_norm.shape)
    print("Shape of ATAC graph (ATAC cells):",atac_graph_norm.shape)
    print("Shape of initial graph (RNA and ATAC cells):",initial_graph_norm.shape)
    
    

    #normalization of GAM and GEM
    norm_rna_features=preprocessing.scale(np.asarray(rna_features))
    norm_gam_features=preprocessing.scale(np.asarray(gam_features))
    
    return {"GEM":norm_rna_features,"GAM":norm_gam_features,"PM":atac_features},{"ref_label":reference_label,"tar_label":target_label}
#------------------------------------------------------------------------------------------------------------------------#








