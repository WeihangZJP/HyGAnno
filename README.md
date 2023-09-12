# HyGAnno
HyGAnno is an automated cell type annotation method designed to improve the annotation quality of single-cell ATAC-seq (scATAC-seq) data. HyGAnno transfers cell type information from well-annotated scRNA-seq references to unlabeled scATAC-seq targets, 
by utilizing inherent information derived from the original peaks of the data. HyGAnno provides not only cell type annotations but also a reference-target cell graph to assist in identifying ambiguous cells, thereby enhancing the reliability of the cell annotations. HyGAnno stands out for its accuracy in cell annotation and its capacity for interpretable cell embedding, exhibiting robustness against noisy reference data and adaptability to tumor tissues. For more information. please refer our manuscript in XXXXX.

HyGAnno is developed using R and Python. The required packages are provided by exported conda virtual environment.
## Prerequisite 
### Essential R packages: 
```
R-basic==4.2.1
Signac==1.7.0
Seurat==4.3.0
```
### Essential Python packages:  
```
Python==3.8.17
numpy==1.23.4
pandas==2.0.3
scikit_learn==1.2.0
scipy==1.9.3
torch==2.0.1 # For CUDA 11.7
matplotlib==3.7.2
umap-learn==0.5.3
pyclustering==0.10.1.2 
```
### Preparing input data for HyGAnno
HyGAnno takes count matrices of `.mtx.gz` format, feature name of `.tsv.gz` and cell label list of `.csv` format as inputs. The reference data of scRNA-seq and target data of scATAC-seq data should be contained in two folders named `Raw_RNA` and `Raw_ATAC`, respectively. 
```
$ tree Raw_RNA
Raw_RNA
├── barcodes.tsv.gz
├── features.tsv.gz
├── matrix.mtx.gz
└── reference_label.csv

$ tree Raw_ATAC
Raw_ATAC
├── barcodes.tsv.gz
├── features.tsv.gz
├── matrix.mtx.gz
└── target_label.csv 
```
Note that the `target_label.csv` is optional and will be only used for validation. Besides, the first column name of `reference(target)_label.csv` should be the`cluster_name` while the seond one should be the `cluster_id`.
## Running HyGAnno

### Constructing graphs
```
# your terminal
$ cd HyGAnno/
$ Rscript Graph_coonstruction.R
```
The processed count matrices and corresponding graphs will be automatically saved in the folder named `./HyGAnno_inputs`.
```
$ tree ./HyGAnno_inputs/Feature_matrices
./HyGAnno_inputs/Feature_matrices
├── GAM.mtx 
├── GEM.mtx 
└── PM.mtx 
```
Note that `GEM.mtx` is the gene expression matrix containing only high varibale genes; `GAM.mtx` is the gene activity matrix containing the same genes with gene expression matrix; `PM.mtx` is the peak matrix containing only high accessiblity peaks.
```
$ tree ./HyGAnno_inputs/Graphs
./HyGAnno_inputs/Graphs
├── ATAC_graph.csv
├── Anchor_graph.csv
└── RNA_graph.csv
```
Note that the first column in `Anchor_graph.csv` means the cell index of RNA cells while the second column means the cell index of ATAC cells. Each cell pair indicates an edge.

### Starting training
1. Modify the arguments in `configs.py` file.
2. Run `main.py` in terminal.
```
# your terminal
$ Python main.py
```
### Arguments for training function
- `dir_path`: Current working directory, default is `"./HyGAnno"`.
- `PATH_feature`: A path list of feature matrices, default searching files in `./HyGAnno/HyGAnno_inputs/Feature_matrices` .
- `PATH_feature`: A path list of graphs, default searching files in `./HyGAnno/HyGAnno_inputs/Graphs` .
- `taget_label`: Accessibilty of the ground truth of target labels, default is `"False"`.
- `use_GPU`: Usage of the GPU, automatically detect avaliable GPU device, if no GPU is detected, CPU will be used.
- `hidden_hyg_dim1`: Dimension number of the first hidden layer of hybrid graph embedding, default is `128`.
- `hidden_hyg_dim2`: Dimension number of the second hidden layer of hybrid graph embedding, default is the number of cell type in reference data.
- `hidden_atac_dim1`: Dimension number of the first hidden layer of atac graph embedding, default is `128`.
- `hidden_atac_dim2`: Dimension number of the second hidden layer of atac graph embedding, default is the number of cell type in reference data.
- `learning_rate`: Leanring rate of the network, default is `0.0001`.
- `epoch`: Epoch number of training, default is `500`.

## Outputs
HyGAnno will output three files in `./outputs`.
```
$ tree outputs/
outputs/
├── atac_cell_embedding.csv
├── cell_type_prediction.csv
└── reconstructed_graph.npz
```
Note that `cell_type_prediction.csv` is the predicted cell labels for target scATAC-seq data; `atac_cell_embedding.csv` is the cell embedding of scATAC-seq data; `reconstructed_graph.npz` is the reconstructed RNA-ATAC graph. 

## Visualizing the cell embeddings
To visualize the cell embedding of scATAC-seq provided by HyGAnno, we apply UMAP on the obtained embeddding space. The pdf figure will be save as `./outputs/UMAP_plot.pdf`.
```
# your terminal
$ Python visualization.py
```

## Detecting ambiguous cells
The RNA-ATAC cell graph reconstructed by HyGAnno can be futher used to detetct ambiguous cells (cells with uncertain prediction or cells without reference information). For the first step, for each ATAC cell, we evaluate the connectivity between this ATAC cell and other RNA cell clusters. If this ATAC cell shows highest connection with RNA cell cluster with an inconsistent cell type different from the predicted cell type, we record this cell as candidate ambiguous cell. For the second step, we apply x-means clustering algorithm on the embedding space of these candidate ambiguous cells, select cells in the largest cluster and filter out other cells. We then use KNN iteration strategy to inflate the largest cluster and obtain the final amiguous cells. We recommend the users remove these cells or care about the prediction results for these ambiguous cells.
```
# your terminal
$ Python ambiguous_cell_detection.py
```
### Arguments for detecting function
- `n_neighbors`: k nearest neighbors for finding the neighbors of the ATAC cells, default is 3. 
- `knn_iter`: iteration times to expand the candidate ambiuous cells
- `expand_strategy`: knn expanding strategy. "soft": ifcandidate ambiguous cell are included in neighbors of a confident cell, this confident cell is re-annotated as candidate ambiguous cell. "hard": only when candidate ambiguous cell number are larger than confident cell number in neighbors of a confident cell, this confident cell is re-annotated as candidate ambiguous cell.




































