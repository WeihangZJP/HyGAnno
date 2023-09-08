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
Python==3.8.13
numpy==1.20.3
pandas==1.3.3
scikit_learn==0.24.2
scipy==1.7.1
torch==2.0.1 # For CUDA 11.2
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

### Graph construction
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
Note that the first column in Anchor_graph.csv means the cell index of RNA cells while the second column means the cell index of ATAC cells. Each cell pair indicates an edge.







































