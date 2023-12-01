
library(Seurat)
library(Signac)
setwd(getwd())

#----------------------------------------------------#
# function name: data_object_generation
# Description: Import the raw count data of gene expression matrix, gene activity matrix 
##-------------and peak matrix. Construct the data object.

## inputs: automatically detect the raw count matrices in three folders,
##---------which are "/Raw_RNA", "/Raw_ATAC" and "/Gene_activity"

## return: A list of data objects: {"GEM","GAM","PM"}

## output files: No output files

#----------------------------------------------------#
data_object_generation<-function(){
  gene_count <- Read10X(paste0(getwd(),"/Raw_RNA"), gene.column = 1)
  peak_count <- Read10X(paste0(getwd(),"/Raw_ATAC"), gene.column = 1)
  gene_activity <- Read10X(paste0(getwd(),"/Gene_activity"), gene.column = 1)
  
  #write.table(colnames(pbmc_gam), file='barcodes.tsv', quote=FALSE, sep='\t')
  #write.table(rownames(pbmc_gam), file='features.tsv', quote=FALSE, sep='\t')
  
  #writeMM(obj = pbmc_gam@assays[["RNA"]]@counts, file="matrix.mtx")
  
  #our input graph and feature generation
  
  #rna preprocessing
  rna_seurat<-CreateSeuratObject(gene_count)
  rna_seurat <- NormalizeData(rna_seurat,verbose = FALSE)
  rna_seurat <- FindVariableFeatures(rna_seurat, nfeatures = 2000,verbose = FALSE)
  rna_seurat <- ScaleData(rna_seurat,verbose = FALSE)
  rna_seurat <- RunPCA(rna_seurat, npcs = 30,verbose = FALSE)
  #rna_seurat <- RunUMAP(rna_seurat, dims = 1:30, reduction.name = "umap.rna",seed.use = 42,verbose = FALSE)
  message("Constructing GEM object")
  #DimPlot(rna_seurat) + NoLegend() + ggtitle("RNA UMAP")
  
  #atac preprocessing
  atac_seurat<-CreateSeuratObject(assay = "ATAC",peak_count)
  atac_seurat <- FindTopFeatures(atac_seurat, min.cutoff = 10,verbose = FALSE) 
  atac_seurat <- RunTFIDF(atac_seurat)
  atac_seurat <- RunSVD(atac_seurat)
  #atac_seurat <- RunUMAP(atac_seurat, reduction = 'lsi', dims = 2:30, reduction.name = 'umap.atac',seed.use = 42,verbose = FALSE)
  #DimPlot(atac_seurat) + NoLegend() + ggtitle("ATAC UMAP")
  message("Constructing PM object")
  
  #gam preprocesssing
  gam_seurat<-CreateSeuratObject(counts = gene_activity)
  gam_seurat <- NormalizeData(gam_seurat,verbose = FALSE)
  gam_seurat<-FindVariableFeatures(gam_seurat, nfeatures = 2000,verbose = FALSE)
  gam_seurat <- ScaleData(gam_seurat,verbose = FALSE)
  gam_seurat <- RunPCA(gam_seurat, npcs = 30,verbose = FALSE)
  #gam_seurat <- RunUMAP(gam_seurat, dims = 1:30, reduction.name = "umap.gam",seed.use = 42,verbose = FALSE)
  #DimPlot(gam_seurat) + NoLegend() + ggtitle("Gene activity UMAP")
  message("Constructing GAM object")
  return(list(GEM=rna_seurat, GAM=gam_seurat, PM=atac_seurat))
}



#----------------------------------------------------#
# function name: HyGAnno_graph_generation
# Description: Generate RNA graph, ATAC graph and the anchor graph for HyGAnno inputs.

## inputs: data_objects: data objects, 
##---------k_rna: the k-nearest neighbor finding edges between RNA cells, 
##---------k_atac: the k-nearest neighbor finding edges between ATAC cells, 
##---------k_rna_atac: the k-nearest neighbor finding edges between RNA and ATAC cells (anchor detection)

## return: No return

## output files: Output graph files of RNA, ATAC, and Anchor cells to folder named "/Graphs" 

#----------------------------------------------------#
HyGAnno_graph_generation<-function(data_objects,k_rna=25,k_atac=25,k_rna_atac=5){
  #create the graph folder
  output_dir<-paste0(getwd(),"/HyGAnno_inputs")
  if (!dir.exists(output_dir)){
    dir.create(output_dir)
  } 
  
  output_dir<-paste0(getwd(),"/HyGAnno_inputs/Graphs")
  if (!dir.exists(output_dir)){
    dir.create(output_dir)
  } 
  
  #graph construction
  message("Start constructing graphs.")
  #construct rna-atac graph
  transfer.anchors <- FindTransferAnchors(reference = data_objects$GEM, query = data_objects$GAM, features = VariableFeatures(object =data_objects$GEM),
                                          reference.assay = "RNA", query.assay = "RNA", reduction = "cca",k.anchor= k_rna_atac,k.filter = NA,verbose = FALSE)
  rna_atac_g<-transfer.anchors@anchors[,1:2]-1
  write.csv(rna_atac_g,paste0(getwd(),"/HyGAnno_inputs/Graphs/Anchor_graph.csv"), row.names = FALSE)#first column contains RNA cell id; second contains ATAC cell id
  message("#--------------------RNA-ATAC graph construction finished--------------------#")
  message(paste0("Graph size (RNA anchor cells by ATAC anchor cells ): ",length(levels(as.factor(rna_atac_g[,1])))," by ",length(levels(as.factor(rna_atac_g[,2])))))
  message(paste0("Edge number: ",length(rna_atac_g[,1]) ))
  
  
  #construct rna-rna graph
  transfer.anchors <- FindTransferAnchors(reference =  data_objects$GEM, query =  data_objects$GEM, features = VariableFeatures(object = data_objects$GEM),
                                          reference.assay = "RNA", query.assay = "RNA", reduction = "pcaproject",k.anchor=k_rna,k.filter = NA,verbose = FALSE)
  
  rna_rna_g<-transfer.anchors@anchors[,1:2]-1
  write.csv(rna_rna_g,paste0(getwd(),"/HyGAnno_inputs/Graphs/RNA_graph.csv"), row.names = FALSE)
  message("#--------------------RNA-RNA graph construction finished--------------------#")
  message(paste0("Graph size (RNA cells by RNA cells ): ",length(levels(as.factor(rna_rna_g[,1])))," by ",length(levels(as.factor(rna_rna_g[,2])))))
  message(paste0("Edge number: ",length(rna_rna_g[,1]) ))
  
  #construct atac-atac graph
  transfer.anchors <- FindTransferAnchors(reference =  data_objects$PM, query = data_objects$PM, features = VariableFeatures(object =data_objects$PM),
                                          reference.assay = "ATAC", query.assay = "ATAC",reference.reduction="lsi", reduction = "lsiproject",dim=2:30, k.anchor=k_atac,k.filter = NA,verbose = FALSE)
  atac_atac_g<-transfer.anchors@anchors[,1:2]-1
  write.csv(atac_atac_g,paste0(getwd(),"/HyGAnno_inputs/Graphs/ATAC_graph.csv"), row.names = FALSE)
  message("#--------------------ATAC-ATAC graph construction finished--------------------#")
  message(paste0("Graph size (ATAC cells by ATAC cells ): ",length(levels(as.factor(atac_atac_g[,1])))," by ",length(levels(as.factor(atac_atac_g[,2])))))
  message(paste0("Edge number: ",length(atac_atac_g[,1]) ))
  
  
}



#----------------------------------------------------#
# function name: HyGAnno_feature_generation
# Description: Take the subset of feature matrices as the input of the HyGAnno method instead of using the 
#-------------whole matrices to accelerate the training process

## inputs: data_objects: data objects
##---------min.cutoff: Include features in over "min.cutoff" cells in the set of VariableFeatures.
##---------------------Higher cutoff will select less peaks.

## return: No return

## output files: Output feature matrices of gene expression, gene activity matrix
##---------------and peak matrix to folder named "/Feature matrices" 

#----------------------------------------------------#
HyGAnno_feature_generation<-function(data_objects,min.cutoff=1000){
  
  output_dir<-paste0(getwd(),"/HyGAnno_inputs")
  if (!dir.exists(output_dir)){
    dir.create(output_dir)
  } 
  
  output_dir<-paste0(getwd(),"/HyGAnno_inputs/Feature_matrices")
  if (!dir.exists(output_dir)){
    dir.create(output_dir)
  } 
  #take the intersection between the highly variable gene set of gene expression matrix and 
  #the whole gene set of gene activity matrix 
  shared_genes<-intersect(VariableFeatures(data_objects$GEM),row.names(data_objects$GAM)) 
  
  #save sub gene expression (shared genes by cells ) matrix in mtx file
  rna_sub<-data_objects$GEM@assays[["RNA"]]@data[shared_genes,]
  library(Matrix)
  rna_sub_sp<-Matrix(rna_sub,sparse = T)  
  writeMM(obj = rna_sub_sp, file=paste0(getwd(),"/HyGAnno_inputs/Feature_matrices/GEM.mtx"))
  message("#--------------------Gene expression matrix saved--------------------#")
  message(paste0("Matrix size (Genes by RNA cells ): ",dim(rna_sub_sp)[1]," by ",dim(rna_sub_sp)[2]))
  
  #save gene (activity) by cells in mtx file (shared by rna and atac)
  gam_sub<-data_objects$GAM@assays[["RNA"]]@data[shared_genes,]
  library(Matrix)
  gam_sub_sp<-Matrix(gam_sub,sparse = T)  
  writeMM(obj = gam_sub_sp, file=paste0(getwd(),"/HyGAnno_inputs/Feature_matrices/GAM.mtx"))
  message("#--------------------Gene activity matrix saved--------------------#")
  message(paste0("Matrix size (Genes by ATAC cells ): ",dim(gam_sub_sp)[1]," by ",dim(gam_sub_sp)[2]))
  
  #save peak (hvp) by cells in mtx file 
  data_objects$PM <- FindTopFeatures(data_objects$PM, min.cutoff = min.cutoff)
  atac_sub<-data_objects$PM@assays[["ATAC"]]@data[VariableFeatures(data_objects$PM),]
  library(Matrix)
  atac_sub_sp<-Matrix(atac_sub,sparse = T)
  writeMM(obj = atac_sub_sp, file=paste0(getwd(),"/HyGAnno_inputs/Feature_matrices/PM.mtx"))
  message("#--------------------Peak matrix saved--------------------#")
  message(paste0("Matrix size (Peaks by ATAC cells ): ",dim(atac_sub_sp)[1]," by ",dim(atac_sub_sp)[2]))

}



#import data matrix to objects
data_objects<-data_object_generation()

#save graph and feature matrices 
HyGAnno_graph_generation(data_objects,k_rna=25,k_atac=25,k_rna_atac=5)
HyGAnno_feature_generation(data_objects,min.cutoff=1000)


