#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:17:02 2023

@author: weihangzhang
"""
    
import os

import torch.nn.functional as F
import sklearn
import torch
import configs
import model
import time
import pandas as pd

from torch.optim import Adam
from sklearn.metrics import accuracy_score
from datapreprocessing import *
from utils import *


dir_path=os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':
    
    #import feature matrices and label list for both reference and target dataset
    feature_dict,label_dict=feature_matrix_import(configs.PATH_feature)
    
    #import graph adjacency matrices, at the same time generate the index dictionarys which records
    #the original index and new index of the ATAC anchor cells
    graph_dict,anchor_atac_dict1,anchor_atac_dict2=graph_matrix_import(feature_dict,configs.PATH_graph)
    
    #generate the hybrid feature matrix by combining the normalized GEM and GAM(only includes ATAC anchor cells) 
    feature_dict=hybrid_feature(feature_dict,anchor_atac_dict2)
    
    #change the numpy format of feature and graph matrices to tensor format
    feature_tensor,graph_tensor=To_Sparse_tensor(graph_dict,feature_dict)
    
    #normalize the whole initial graph 
    norm = graph_dict["initial_label"].shape[0] * graph_dict["initial_label"].shape[0] / float((graph_dict["initial_label"].shape[0] * graph_dict["initial_label"].shape[0] - graph_dict["initial_label"].sum()) * 2)
    weight = float(graph_dict["initial_label"].shape[0] * graph_dict["initial_label"].shape[0] - graph_dict["initial_label"].sum()) / graph_dict["initial_label"].sum()
    weight_mask = graph_tensor["initial_label"].to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)) 
    weight_tensor[weight_mask] = weight
    
    
    ref_label_ind=np.array([i for i in range(feature_dict["GEM"].shape[0])])
    ref_label_ind_ten=torch.tensor(ref_label_ind)
    
    #import paramaters to model, if configs.use_GPU is True, the mdoel will be trained on GPU node (faster, recommended)
    #otherwise the model will run on CPU node (slower)
    #by default configs.use_GPU is False
    if configs.use_GPU:
        device = torch.device("cuda:0")
        hyg_graph_norm_tens=graph_tensor["hybrid"].to(device)
        atac_graph_norm_tens=graph_tensor["ATAC"].to(device)
        
        Model = model.multi_VGAE(hyg_graph_norm_tens,atac_graph_norm_tens).to(device)
        optimizer = Adam(model.parameters(), lr=configs.learning_rate)
        
        #dynamic learning rate is applied via CosineAnnealingWarmRestarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=1)
        
        hyg_features_tens=feature_tensor["hybrid"].to(device)
        atac_features_tens=feature_tensor["ATAC"].to(device)
        
        true_ref_label_tens = torch.tensor(label_dict["ref_label"]).to(device)
        initial_graph_label_tens = graph_tensor["initial_label"].to(device)
        
        weight_tensor=weight_tensor.to(device)
        ref_label_ind_ten=ref_label_ind_ten.to(device)
    else:
        hyg_graph_norm_tens=graph_tensor["hybrid"]
        atac_graph_norm_tens=graph_tensor["ATAC"]
        
        Model = model.multi_VGAE(hyg_graph_norm_tens,atac_graph_norm_tens,feature_dict["GEM"].shape[1],feature_dict["PM"].shape[1])
        optimizer = Adam(Model.parameters(), lr=configs.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100,T_mult=1)
        hyg_features_tens=feature_tensor["hybrid"]
        atac_features_tens=feature_tensor["PM"]
        
        true_ref_label_tens = torch.tensor(label_dict["ref_label"])
        initial_graph_label_tens = graph_tensor["initial_label"]
        
    
    #begin training!
    for epoch in range(1,configs.epoch):
        t = time.time()
        
        #forward
        #loss_align minimize the distance of ATAC anchor cells viewed by hybrid and atac graphs
        A_pred_m,Z_m,Z_h,Mean_atac,Logstd_atac,output,loss_align = Model(hyg_features_tens,atac_features_tens,feature_dict["GEM"].shape[0],anchor_atac_dict1)
        
        #set grad zero
        optimizer.zero_grad()
        
        #loss1: kl_dvergence to make the embedding of cells in hybrid and atac graphs are satisfied with gaussian distribution
        kl_dvergence_hyg=0.5/ feature_dict["hybrid"].shape[0] * (1 + 2*Model.logstd_hyg - Model.mean_hyg**2 - torch.exp(Model.logstd_hyg)**2).sum(1).mean()
        kl_dvergence_atac=0.5/ feature_dict["PM"].shape[0] * (1 + 2*Model.logstd_atac - Model.mean_atac**2 - torch.exp(Model.logstd_atac)**2).sum(1).mean()
        loss1=-(kl_dvergence_hyg+kl_dvergence_atac)
        
        #loss2: initial graph reconstruction loss
        loss2=norm*F.binary_cross_entropy(A_pred_m.view(-1), initial_graph_label_tens.to_dense().view(-1),weight = weight_tensor)
        
        #loss3:reference prediction loss
        loss3=F.nll_loss(output[ref_label_ind_ten], true_ref_label_tens)
    
    
        loss = loss1+loss2+loss3+loss_align
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        edge_similarity = edges_diff(A_pred_m,initial_graph_label_tens)
        node_similarity = nodes_diff(output[ref_label_ind_ten], true_ref_label_tens)
        preds = output.max(1)[1].type_as(ref_label_ind_ten).cpu().detach().numpy()
        if configs.taget_label:
            print('%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()),"ali_loss=", "{:.5f}".format(loss_align.item()),
                  "node_acc=", "{:.5f}".format(node_similarity), "edge_acc=", "{:.5f}".format(edge_similarity),
                  "time=", "{:.5f}".format(time.time() - t),"NMI=","{:.5f}".format(sklearn.metrics.normalized_mutual_info_score(label_dict["tar_label"],preds[feature_dict["GEM"].shape[0]:])),
                  "ACC=","{:.5f}".format(accuracy_score(label_dict["tar_label"], preds[feature_dict["GEM"].shape[0]:])),"lr=","{:.5f}".format(optimizer.param_groups[-1]['lr']))
        else:
             print('%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()),"ali_loss=", "{:.5f}".format(loss_align.item()),
              "node_acc=", "{:.5f}".format(node_similarity), "edge_acc=", "{:.5f}".format(edge_similarity),
              "time=", "{:.5f}".format(time.time() - t),"lr=","{:.5f}".format(optimizer.param_groups[-1]['lr']))

    print("training finished!")
    preds = output.max(1)[1].type_as(ref_label_ind_ten).cpu().detach().numpy()
    tar_preds=list(preds[feature_dict["GEM"].shape[0]:])
    

    if not os.path.exists(dir_path+"/outputs"):
        os.makedirs(dir_path+"/outputs")
    tar_preds_df=pd.DataFrame({"pred_cell_id":tar_preds})
    tar_preds_df.to_csv(dir_path+"/outputs/cell_type_prediction.csv",index=False)
        
    


    
    
    
