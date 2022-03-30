# mCSEA pipeline
require(mCSEA)

# Load data
# ===
load('~/projects/redes-tf/data/methylation/methylation_jose.RData')

PREC.res$scaled = as.factor(PREC.res$scaled)
meta = data.frame(
  expla = as.factor(PREC.res$scaled),
  row.names = rownames(PREC.res)
)

# Rank probes
# ===
rank = rankProbes(B.precisesads, meta, caseGroup = 'responder', refGroup = 'non_responder')

# Specify DMRs
# ===
res = mCSEATest(rank,
                B.precisesads,
                PREC.res,
                regionsTypes = 'promoters',
                platform = '450k')


gsea = res$promoters
up = rownames(gsea[which(gsea$NES > 2 & gsea$padj < 0.01), ])
down = rownames(gsea[which(gsea$NES < -2 & gsea$padj < 0.01), ])

genelist = list(
  up = up,
  down = down,
  genotype = rownames(gsea)
)
saveRDS(genelist, file = '~/projects/redes-tf/results/methylation/geneList_diffMethylation.rds')

# Plotting volcano
# ===
alpha = 0.001
gsea = res$promoters
gsea$log10adjpval = -log10(gsea$padj)
gsea$color = as.factor(ifelse(gsea$padj < alpha, 'padj < 0.001', 'padj > 0.001'))
gsea$dmr = rownames(gsea)
gsea$score = gsea$log10adjpval * sign(gsea$NES)


saveRDS(gsea, file = '~/projects/redes-tf/results/methylation/gsea.rds')

require(ggplot2)
require(ggpubr)
gsea$size = gsea$size * 0.01
ggscatter(gsea,
              x = 'NES',
              y = 'log10adjpval',
              size = 0.5,
              # mean.point = T,
              # mean.point.size = 0.5,
              color = 'color',
              palette = c('#440154FF', '#22A884FF'),
              font.label = c(6, 'plain'),
              label = 'dmr',
              label.select = gsea$dmr[which(gsea$log10adjpval > 8)],
              repel = T,
              label.rectangle = T,
              show.legend.text = F,
              ggtheme = theme_pubr(
                base_size = 8
              )) 

setwd('~/projects/redes-tf/results/methylation/plots/')
ggplot2::ggsave(filename = 'volcano.tiff', 
       width = 6.35,
       height = 6.6,
       units = 'cm',
       dpi = 300)


# mCSEAPlot(res, regionType = 'promoters',
#           dmrName = 'LTA',
#           transcriptAnnotation = 'symbol',
#           makePDF = F)

size = 5
lta = mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "LTA") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(),
        axis.text.x = element_text(size = size), axis.text.y = element_text(size = size),
        title = element_text(size = 5))
nnat = mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "NNAT") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(),
        axis.text.x = element_text(size = size), axis.text.y = element_text(size = size),
        title = element_text(size = 5))
blcap = mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "BLCAP") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(),
        axis.text.x = element_text(size = size), axis.text.y = element_text(size = size),
        title = element_text(size = 5))
sfrp2 = mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "SFRP2") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank(),
        axis.text.x = element_text(size = size), axis.text.y = element_text(size = size),
        title = element_text(size = 5))

ggarrange(lta, nnat, blcap, sfrp2, nrow = 2, ncol = 2, common.legend = T)

ggplot2::ggsave(filename = 'enrich.tiff', 
                width = 6.35,
                height = 6.35,
                units = 'cm',
                dpi = 300)
