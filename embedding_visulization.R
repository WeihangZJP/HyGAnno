library(ggplot2)
library(ggthemes)
library(ggpubr)
library(ggsci)
library(uwot)
setwd(getwd()) 
set.seed(42)
class_color20<-c("#1f77b4","#aec7e8","#ff7f0e","#ffbb78","#2ca02c","#98df8a","#d62728","#ff9896","#9467bd","#c5b0d5",
                 "#8c564b","#c49c94","#e377c2","#f7b6d2","#7f7f7f","#c7c7c7","#bcbd22","#dbdb8d","#17becf","#9edae5")

atac_cell_types<- read.csv(paste0(getwd(),"/outputs/cell_type_prediction.csv"))
atac_cell_embedding<-read.csv(paste0(getwd(),"/outputs/atac_cell_embedding.csv"))

#use for obtaining cell type names
rna_cell_types<-read.csv(paste0(getwd(),"/Raw_RNA/reference_label.csv"))

atac_level<-levels(factor(atac_cell_types$pred_cell_id))
atac_label<-c()
for (level in atac_level){
  atac_label<-c(atac_label,rna_cell_types[which(rna_cell_types$cluster_id==level),][1,1])
}


atac_umap_embedding <- umap(atac_cell_embedding, min_dist = 0.0, n_neighbors=50)

umap_df <- data.frame(UMAP1 = atac_umap_embedding[, 1], UMAP2 = atac_umap_embedding[, 2],
                 cell_label = factor(as.character(atac_cell_types$pred_cell_id),levels = atac_level,labels=atac_label))
p1<-ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = cell_label)) +
  xlab(NULL) +
  ylab(NULL)+
  geom_point(size = 0.5) +
  #theme_dr(xlength = 0.4,
  #         ylength = 0.4,)+
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.y =element_blank(),
        axis.text.y=element_blank(),
        
        panel.grid.major = element_blank(),
        #panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        #panel.border = element_blank()
        panel.border = element_rect(fill=NA,color="black", size=1, linetype="solid")
  )+
  scale_color_manual(values = class_color20)+
  theme(axis.title.y = element_text(size=15),axis.title.x = element_text(size=15))+
  guides(col= guide_legend(title= " ",override.aes = list(size=5)))

ggsave(file.path(paste0(getwd(),"/outputs/UMAP_plot.pdf")), width = 5, height = 4.5)

write.csv(umap_df,paste0(getwd(),"/outputs/UMAP_atac.csv"), row.names = FALSE)


