setwd('~/projects/redes-tf/results/gsva/')

# Load expression data
# ===
load('../../data/expression/expression_Raul.RData')

# Load tmod
# ===
require(tmod)
data("tmod")

modules = tmod$MODULES
immune = modules[which(modules$Annotated == 'Yes' & modules$SourceID == 'LI' & modules$Category == 'immune'), ]

# Load gsva
# ===
gsva = readRDS('msigDB_tmod_ssgsea.rds')
gsva = gsva[match(rownames(PREC.res), rownames(gsva)), ]

annot = data.frame(
  # ids = rownames(gsva),
  treatment = PREC.res$scaled,
  row.names = rownames(gsva)
)
annot$treatment[which(annot$treatment == 'moderate_responder')] = 'responder'


require(ComplexHeatmap)
toPlot = t(gsva)
toPlot = toPlot[match(immune$ID, rownames(toPlot)),]
toPlot = toPlot[complete.cases(toPlot),]
toPlot = toPlot - rowMeans(toPlot)
ComplexHeatmap::Heatmap(toPlot,
                        show_row_names = T, show_column_names = F,
                        # column_order = pats,
                        # row_labels = s,
                        show_heatmap_legend = T,
                        top_annotation = HeatmapAnnotation(df = annot),
                        # right_annotation = rowAnnotation(foo = anno_block(gp = gpar(fill = '#FFFFB3'),
                        #                                                   labels = names(sigs)[i],
                        #                                                   labels_gp = gpar(col = 'black', fontsize = 12)))
                        )

immune = li[which(li$category == 'immune'),]
toPlot[1:5,1:5]
