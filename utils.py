import scipy.sparse as sp
import numpy as np
import torch



#------------------------------------------------------------------------------------------------------------------------#
# function name: To_Sparse_tensor
# Description: Change graph and feature data into tuple

## inputs: 1. dictionaries of sparse feature matrices
##---------2. graph adjacency matrices 

## return: 1. dictionary of sparse feature matrix with tensor format
##---------2. dictionary of sparse graph matrix with tensor format


## output files: No output files


def  To_Sparse_tensor(graph_dict,feature_dict):
    for key in graph_dict.keys():
        tup=sparse_to_tuple(graph_dict[key].tocoo())
        graph_dict[key]=torch.sparse.FloatTensor(torch.LongTensor(tup[0].T), 
                            torch.FloatTensor(tup[1]), 
                            torch.Size(tup[2]))
    for key in feature_dict.keys():
        tup=sparse_to_tuple(sp.csr_matrix(feature_dict[key].T).tocoo())
        feature_dict[key] = torch.sparse.FloatTensor(torch.LongTensor(tup[0].T), 
                            torch.FloatTensor(tup[1]), 
                            torch.Size(tup[2]))
    return feature_dict,graph_dict

#------------------------------------------------------------------------------------------------------------------------#




#------------------------------------------------------------------------------------------------------------------------#
# function name: sparse_to_tuple
# Description: Change graph and feature data into tuple

## inputs: a sparse matrix (feature matrix or graph adjacency matrix)

## return: 1. Ccoordinates of elements in matrix
##---------2. Value of elements
##---------3. Shape of matrix

## output files: No output files


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

#------------------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------------------#
# function name: hybrid_feature
# Description: Construct the hybrid feature matrix by concatenate normalized GEM and GAM (containing only ATAC anchor cells)

## inputs: 1. feature dictionary  
##---------2. atac index dictionary

## return: feature dictionary with new hybrid feature matrix

## output files: No output files


def hybrid_feature(feature_dict,anchor_atac_dict2):
    rna_cell_num=feature_dict["GEM"].shape[1]
    overlap_ind=[int(i-rna_cell_num) for i in list(anchor_atac_dict2.keys())]
    hybrid_features=np.hstack((feature_dict["GEM"],feature_dict["GAM"][:,overlap_ind]))
    feature_dict["hybrid"]=hybrid_features
    return feature_dict
#------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------#
# function name: edges_diff
# Description: Evaluate the difference between the initial graph and reconstructed graph 

## inputs: 1. initial graph
##---------2. reconstructed graph

## return: ratio of overlapped edges 

## output files: No output files


def edges_diff(adj_pred, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_pred > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy

#------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------------------------------------------------------------------#
# function name: nodes_diff
# Description: Evaluate the difference between the predicted and the ground truth of reference cells 

## inputs: 1. predicted labels of reference cells 
##---------2.ground truth labels of reference cells 

## return: accuracy of predicted reference cells

## output files: No output files
def nodes_diff(pred_ref, true_ref):
    preds = pred_ref.max(1)[1].type_as(true_ref)
    correct = preds.eq(true_ref).double()
    correct = correct.sum()
    return correct / len(true_ref)
#------------------------------------------------------------------------------------------------------------------------#
