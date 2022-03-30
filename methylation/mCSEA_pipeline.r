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
up = rownames(gsea[which(gsea$NES > 0 & gsea$padj < 0.01), ])
down = rownames(gsea[which(gsea$NES < 0 & gsea$padj < 0.01), ])

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
gsea$color = as.factor(ifelse(gsea$padj < alpha, 'YES', 'NO'))
gsea$dmr = rownames(gsea)
gsea$score = gsea$log10adjpval * sign(gsea$NES)


saveRDS(gsea, file = '~/projects/redes-tf/results/methylation/gsea.rds')


ggpubr::ggscatter(gsea,
                  x = 'NES',
                  y = 'log10adjpval',
                  size = 'size',
                  color = 'color',
                  palette = c('#440154FF', '#22A884FF'),
                  label = 'dmr',
                  label.select = gsea$dmr[which(gsea$padj < alpha)],
                  repel = T,
                  label.rectangle = T,
                  show.legend.text = F,
                  title = 'Methylation')


head(res[["promoters"]][,-7])

mCSEAPlot(res, regionType = 'promoters',
          dmrName = 'LTA',
          transcriptAnnotation = 'symbol',
          makePDF = F)


mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "LTA")
