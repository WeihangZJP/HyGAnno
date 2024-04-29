import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
import argparse
import configs

dir_path=os.path.dirname(os.path.abspath(__file__))


#------------------------------------------------------------------------------------------------------------------------#
# function name: first_decision
# Description: For each cell in target data, the D and W are calculated to find the cell type 
##-------------which has the highest connectivity with this cell. If the found cell type is 
##-------------inconsistent with the predicted label, we assume the prediction result for this
##-------------cell is ambiguous (unreliable prediction or lacking reference information). 
##-------------And treat this kind of cells as ambiguous cells.

## inputs: 1. reconstructed RNA-ATAC cell graph
##---------2. ground truth of reference cell labels

## return: 1. indices of candidate ambiguous cells
##---------2. umap dataframe of the cell embedding


## output files: UMAP visualization of the distribution of candidate ambiguous cells 
##---------------(saved in ./outputs/Ambiguous_cell_detection/)


def first_decision(reconstructed_graph,target_label_pred,reference_label):
    reference_counter = Counter(reference_label)
    print("Start calculating the metric D and W for each target cell")
    #calculate the metrics fo w and d for each cell in target data
    metric_W=[]
    metric_D=[]
    
    
    for i in range(len(target_label_pred)):
        if i%1000==0 and i!=0:
            print(i,"cells finished")
        d={}
        w={}
        each_cluster_sum={}
        for c in range(len(set(reference_label))):
            d[str(c)]=0
            w[str(c)]=0
            each_cluster_sum[str(c)]=0
        
        for j in range(len(reference_label)):
            if reconstructed_graph[i+len(reference_label),j]>=0.5:
                w[str(reference_label[j])]+=reconstructed_graph[i+len(reference_label),j]
                each_cluster_sum[str(reference_label[j])]+=1
                
        for k in list(w.keys()):
            if each_cluster_sum[k]!=0:
                w[k]=w[k]/each_cluster_sum[k]
            else:
                 w[k]=0
            if reference_counter[int(k)]!=0:
                d[k]=each_cluster_sum[k]/reference_counter[int(k)]
            else:
                d[k]=0
        metric_W.append(w)
        metric_D.append(d)
    
    #find the cell type with the highest d and w
    rank_W=[]
    for i in range(len(target_label_pred)):
        values=list(metric_W[i].values())
        rank_W.append(values.index(max(values)))
    rank_D=[]
    for i in range(len(target_label_pred)):
        values=list(metric_D[i].values())
        rank_D.append(values.index(max(values)))
    
    #detect inconsistent cells as candidate ambiguous cells
    inconsistent_obs_W=[]
    for i in range(len(target_label_pred)):
        if rank_W[i]!=target_label_pred[i]:
            inconsistent_obs_W.append(i)
    inconsistent_obs_D=[]
    for i in range(len(target_label_pred)):
        if rank_D[i]!=target_label_pred[i]:
            inconsistent_obs_D.append(i)
    
    #take the intersection of the inconsistent cell index of D and W
    candidate_ambiguous_cell=list(set(inconsistent_obs_W) & set(inconsistent_obs_D))
    print("First decision:",len(candidate_ambiguous_cell),"candidate ambiguous cells")
    
    # 0 is confident cell, 1 is candidate ambiguous cell
    cell_certainty=["confident" for i in range(len(target_label_pred))]
    for i in candidate_ambiguous_cell:
        cell_certainty[i]="ambiguous"
    
    umap_df=pd.read_csv(dir_path+"/outputs/umap_atac.csv")   
    umap_df["cell_certainty"]=cell_certainty
    
    if not os.path.exists(dir_path+"/outputs/Ambiguous_cell_detection"):
        os.makedirs(dir_path+"/outputs/Ambiguous_cell_detection")
    
    amb=umap_df[umap_df["cell_certainty"]=="ambiguous"]
    con=umap_df[umap_df["cell_certainty"]=="confident"]
    
    class_color20=["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
                 "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5"]
    color=[class_color20[i] for i in target_label_pred]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1) 
    plt.title("Cell type prediction")
    x=umap_df["UMAP1"]
    y=umap_df["UMAP2"]
    for label in set(umap_df["cell_label"]):
        mask = [l == label for l in umap_df["cell_label"]]
        plt.scatter([x[i] for i in range(len(x)) if mask[i]],
                    [y[i] for i in range(len(y)) if mask[i]],
                    label=label,c=[color[i] for i in range(len(color)) if mask[i]],s=1)
    

    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2) 
    plt.title("Ambiguous cell detection")
    
    plt.scatter(con["UMAP1"],con["UMAP2"], s=1,c="#BDBDBD", alpha=0.5,label="Confident cell")
    plt.scatter(amb["UMAP1"],amb["UMAP2"], s=1,c="#EF8535", alpha=0.5,marker='^',label="Ambiguous cell")
    
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(pad=3)
    plt.savefig(dir_path+"/outputs/Ambiguous_cell_detection/Ambiguous_cell_distribution.pdf", format="pdf", bbox_inches="tight")
    
    umap_df.to_csv(dir_path+"/outputs/Ambiguous_cell_detection/target_cell_meta.csv",index=False)
    
    return candidate_ambiguous_cell,umap_df
#------------------------------------------------------------------------------------------------------------------------#






#------------------------------------------------------------------------------------------------------------------------#
# function name: second_decision
# Description: an knn-based method to detect more candidate ambiguous cells

## inputs: 1. indices of candidate ambiguous cells from the first decision
##---------2. umap dataframe of the cell embedding
##---------3. k nearest neighbors 
##---------4. iteration for expanding the cell cluster
##---------5. expanding strategy: "hard" or "soft" 

## return: No return


## output files: 1. Visualization of final detected ambiguous cells 
##---------------2. the metadata of target cells 
##---------------(saved in ./outputs/Ambiguous_cell_detection/)

def second_decision(candidate_ambiguous_cell,umap_df,target_label_pred,n_neighbors=3,knn_iter=40,expand_strategy="soft"):
    
    
    #import cell embedding
    atac_cell_embed=pd.read_csv(dir_path+"/outputs/atac_cell_embedding.csv")


    #knn expanding these candidate cells
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(atac_cell_embed)
    knn_matrix = nbrs.kneighbors(atac_cell_embed, return_distance=False)  
    knn_matrix=knn_matrix[:,1:]
    
    cell_certainty_new=[]
    for i in range(len(target_label_pred)):
        if i in candidate_ambiguous_cell:
            cell_certainty_new.append("ambiguous")
        else:
            cell_certainty_new.append("confident")
            
    for t in range(knn_iter):
        knn_matrix_label=[]
        for i in range(len(knn_matrix)):
            label_distribution_one_cell=[]
            for j in knn_matrix[i]:
                label_distribution_one_cell.append(cell_certainty_new[j])
            knn_matrix_label.append(label_distribution_one_cell)
    
        #hard expanding strategy
        if expand_strategy=="hard":
            for i in range(len(knn_matrix)):
                dict_label=Counter(knn_matrix_label[i])
                if cell_certainty_new[i]!="ambiguous":
                    cell_certainty_new[i]=max(dict_label, key=dict_label.get)
        #soft expanding strategy
        if expand_strategy=="soft":
            for i in range(len(knn_matrix)):
                dict_label=Counter(knn_matrix_label[i])
                if cell_certainty_new[i]!="ambiguous":
                    if "ambiguous" in list(dict_label.keys()):
                        cell_certainty_new[i]="ambiguous"
        final_unknown_ind=[]
        for ind,i in enumerate(cell_certainty_new):
            if i=="ambiguous":
                final_unknown_ind.append(ind)
    
    print("Final decision:",len(final_unknown_ind),"ambiguous cells")
    final_ambiguous_cells=["confident" for i in range(len(target_label_pred))]
    for i in final_unknown_ind:
        final_ambiguous_cells[i]="ambiguous"
 
    target_cell_meta = {
      "UMAP1": umap_df["UMAP1"],
      "UMAP2": umap_df["UMAP2"],
      "precicted_label":umap_df["cell_label"],
      "prediction_certainty":final_ambiguous_cells 
    }
    target_cell_meta_df = pd.DataFrame(target_cell_meta)
    
    amb=target_cell_meta_df[target_cell_meta_df["prediction_certainty"]=="ambiguous"]
    con=target_cell_meta_df[target_cell_meta_df["prediction_certainty"]=="confident"]
    
    class_color20=["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
                 "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5"]
    color=[class_color20[i] for i in target_label_pred]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1) 
    plt.title("Cell type prediction")
    x=target_cell_meta["UMAP1"]
    y=target_cell_meta["UMAP2"]
    for label in set(target_cell_meta["precicted_label"]):
        mask = [l == label for l in target_cell_meta["precicted_label"]]
        plt.scatter([x[i] for i in range(len(x)) if mask[i]],
                    [y[i] for i in range(len(y)) if mask[i]],
                    label=label,c=[color[i] for i in range(len(color)) if mask[i]],s=1)
    
    
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2) 
    plt.title("Ambiguous cell detection (second step)")
    
    plt.scatter(con["UMAP1"],con["UMAP2"], s=1,c="#BDBDBD", alpha=0.5,label="Confident cell")
    plt.scatter(amb["UMAP1"],amb["UMAP2"], s=1,c="#EF8535", alpha=0.5,marker='^',label="Ambiguous cell")
    
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(pad=3)
    plt.savefig(dir_path+"/outputs/Ambiguous_cell_detection/Ambiguous_cell_distribution.pdf", format="pdf", bbox_inches="tight")
    
    target_cell_meta_df.to_csv(dir_path+"/outputs/Ambiguous_cell_detection/target_cell_meta.csv",index=False)
#------------------------------------------------------------------------------------------------------------------------#


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default = 3)
    parser.add_argument('--knn_iter', type=int, default=40)
    parser.add_argument('--expand_strategy', type=str, default="hard")
    args = parser.parse_args()

    

    #import data
    reconstructed_graph = sp.load_npz(dir_path+"/outputs/reconstructed_graph.npz")
    reconstructed_graph=reconstructed_graph.todense()
    
    target_label_pred=pd.read_csv(dir_path+"/outputs/cell_type_prediction.csv")
    reference_label=pd.read_csv(configs.PATH_reference_label)     
    target_label_pred=list(target_label_pred.iloc[:,0])
    reference_label=list(reference_label.iloc[:,1])
    
    candidate_ambiguous_cell,umap_df=first_decision(reconstructed_graph,target_label_pred,reference_label)
    second_decision(candidate_ambiguous_cell,umap_df,target_label_pred,n_neighbors=args.n_neighbors,knn_iter=args.knn_iter,expand_strategy=args.expand_strategy)
    