import torch
import torch.nn.functional as F
import torch.nn as nn
import configs 
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import configs 
import numpy as np

#------------------------------------------------------------------------------------------------------------------------#
# class name: multi_VGAE
# Description: hybrid VGAE model
#------------ base_gcn_hyg -> gcn_mean_hyg 
#------------------------- -> gcn_logstddev_hyg

#------------ base_gcn_atac -> gcn_mean_atac
#-------------------------  -> gcn_logstddev_atac

class multi_VGAE(nn.Module):
    def __init__(self, adj1,adj2,input_hyb_dim,input_atac_dim):
        super(multi_VGAE,self).__init__()
        self.input_hyb_dim=input_hyb_dim
        self.input_atac_dim=input_atac_dim
        
        self.base_gcn_hyg = one_layerGCN(self.input_hyb_dim, configs.hidden_hyg_dim1, adj1)
        self.gcn_mean_hyg = one_layerGCN(configs.hidden_hyg_dim1, configs.hidden_hyg_dim2, adj1,activation=lambda x:x)
        self.gcn_logstddev_hyg = one_layerGCN(configs.hidden_hyg_dim1, configs.hidden_hyg_dim2, adj1,activation=lambda x:x)

        self.base_gcn_atac = one_layerGCN(self.input_atac_dim,configs.hidden_atac_dim1, adj2)
        self.gcn_mean_atac = one_layerGCN(configs.hidden_atac_dim1, configs.hidden_atac_dim2, adj2, activation=lambda x:x)
        self.gcn_logstddev_atac = one_layerGCN(configs.hidden_atac_dim1, configs.hidden_atac_dim2, adj2, activation=lambda x:x)
        #self.fn=nn.Linear(configs.hidden_atac_dim2,configs.hidden_atac_dim2)
        
    def forward(self, X1,X2,rna_cell_num,anchor_atac_dict1):
        Z_h,Z_m,loss_align = self.encode(X1,X2,rna_cell_num,anchor_atac_dict1)
        output=F.log_softmax(Z_m, dim=1)
        A_pred_m =  torch.sigmoid(torch.matmul(Z_m,Z_m.t()))
        return  A_pred_m,Z_m,Z_h,self.mean_atac,self.logstd_atac,output,loss_align

    def encode(self, X1,X2,rna_cell_num,anchor_atac_dict1):

        hidden_hyg = self.base_gcn_hyg(X1)
        self.mean_hyg = self.gcn_mean_hyg(hidden_hyg)
        self.logstd_hyg = self.gcn_logstddev_hyg(hidden_hyg)
        if configs.use_GPU:
            gaussian_noise = torch.randn(X1.size(0), configs.hidden_hyg_dim2).to(torch.device("cuda:0"))
        else:
            gaussian_noise = torch.randn(X1.size(0), configs.hidden_hyg_dim2)
        sampled_z_1 = gaussian_noise*torch.exp(self.logstd_hyg) + self.mean_hyg


        hidden_atac = self.base_gcn_atac(X2)
        self.mean_atac = self.gcn_mean_atac(hidden_atac)
        self.logstd_atac = self.gcn_logstddev_atac(hidden_atac)
        if configs.use_GPU:
            gaussian_noise = torch.randn(X2.size(0), configs.hidden_atac_dim2).to(torch.device("cuda:0"))
        else:
            gaussian_noise = torch.randn(X2.size(0), configs.hidden_atac_dim2)
        sampled_z_2 = gaussian_noise*torch.exp(self.logstd_atac) + self.mean_atac

        #self alignment: minimize the distance between ATAC anchor cells in hybrid graph and ATAC graph
        loss_self=0
        for i in range(len(sampled_z_1)-rna_cell_num):     
            sampled_z_2[int(anchor_atac_dict1[i])-rna_cell_num,:]=sampled_z_2[int(anchor_atac_dict1[i])-rna_cell_num,:]
            loss_self+=torch.norm(sampled_z_1[i+rna_cell_num,:]-sampled_z_2[int(anchor_atac_dict1[i])-rna_cell_num,:])
        loss_self=loss_self/(len(sampled_z_1)-rna_cell_num)
        
        
        sample_merge_z=torch.cat((sampled_z_1[0:rna_cell_num,:],sampled_z_2),0)

        return sampled_z_1,sample_merge_z,loss_self#+loss_anchor
#------------------------------------------------------------------------------------------------------------------------#




#------------------------------------------------------------------------------------------------------------------------#
# class name: one_layerGCN model
# Description: basic one layer GCN relu(AXW),
#-------------where A is the adjacency matricx, X is the feature matrix, W is the hidden paramater matrix intialized by glorot

class one_layerGCN(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation = F.relu,**kwargs):
        super(one_layerGCN, self).__init__(**kwargs)
        
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        self.weight = nn.Parameter(initial)
        self.adj = adj 
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs
#------------------------------------------------------------------------------------------------------------------------#




    

