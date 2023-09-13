import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import os
dir_path=os.path.dirname(os.path.abspath(__file__))

class_color20=["#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
                 "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5"]



atac_label_pred=pd.read_csv(dir_path+"/outputs/cell_type_prediction.csv")
atac_label_pred=list(atac_label_pred.iloc[:,0])
rna_labels=pd.read_csv(dir_path+"/Raw_RNA/reference_label.csv")

label_dict={}
for i in range(len(rna_labels["cluster_id"])):
    label_dict[str(rna_labels["cluster_id"][i])]=rna_labels["cluster_name"][i]
atac_label_id=[label_dict[str(i)] for i in atac_label_pred]



atac_cell_embedding=pd.read_csv(dir_path+"/outputs/atac_cell_embedding.csv")

color=[class_color20[i] for i in atac_label_pred]

mapper = umap.UMAP(random_state=42,metric="euclidean",n_neighbors=50,min_dist=0.0)
embedding = mapper.fit_transform(atac_cell_embedding)

plt.figure(figsize=(5, 5))
x=embedding[:,0]
y=embedding[:,1]
for label in set(atac_label_id):
    mask = [l == label for l in atac_label_id]
    plt.scatter([x[i] for i in range(len(x)) if mask[i]],
                [y[i] for i in range(len(y)) if mask[i]],
                label=label,s=10)


plt.xticks([])
plt.yticks([])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(dir_path+"/outputs/umap_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()

umap_embedding = {
      "UMAP1": x,
      "UMAP2": y,
      "cell_label":atac_label_id,
    }
umap_embedding_df = pd.DataFrame(umap_embedding)
    
umap_embedding_df.to_csv(dir_path+"/outputs/umap_atac.csv",index=False)