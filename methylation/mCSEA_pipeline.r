# mCSEA pipeline
require(mCSEA)

# Load data
# ===
load('~/projects/redes-tf/data/methylation/methylation_jose.RData')

PREC.res$scaled = as.factor(PREC.res$scaled)
meta = data.frame(
  expla = PREC.res$scaled,
  id = rownames(PREC.res),
  row.names = rownames(PREC.res)
)


# Rank probes
# ===
rank = rankProbes(B.precisesads, meta, refGroup = 'responder')
head(rank)

# Specify DMRs
# ===
res = mCSEATest(rank,
                B.precisesads,
                PREC.res,
                regionsTypes = 'promoters',
                platform = '450k')

head(res[["promoters"]][,-7])

mCSEAPlot(res, regionType = 'promoters',
          dmrName = 'ZAP70',
          transcriptAnnotation = 'symbol',
          makePDF = F)


mCSEAPlotGSEA(rank, res, regionType = "promoters", dmrName = "ZAP70")
